import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dataloaders.dataloader import get_dataloader, save_iter
from models.model_voxel_loader import load_optim_sched,load_vqvae_voxel_diffusion
from models.train_utils import getGradNorm, set_seed, setup_output_subdirs
from utils.args import parse_args
import time
from models.evaluation import evaluate_sdf

def init_processes(rank: int | str, size: int, fn: callable, args: DictConfig) -> None:
    """Initialize the distributed environment.

    Args:
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        fn (function): Function to run.
        args (DictConfig): Configuration.
    """

    if rank != 0:
        time.sleep(1)
    torch.cuda.set_device(rank)
    args.local_rank = rank
    args.global_rank = rank
    args.global_size = size
    args.gpu = rank

    # usual env init
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)

    fn(args)

    dist.barrier()
    dist.destroy_process_group()

def train(cfg: DictConfig) -> None:
    is_main_process = cfg.local_rank == 0

    logger.remove()

    if is_main_process:
        (outf_syn,) = setup_output_subdirs(cfg.output_dir, "output")
        cfg.outf_syn = outf_syn
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            + "<level>{level: <8}</level> | "
            + "<level>{message}</level>"
        )
        logger.add(sys.stdout, level="INFO", format=fmt)

    set_seed(cfg)
    torch.cuda.empty_cache()

    train_loader, train_sampler = get_dataloader(cfg,phase='train')
    val_loader,val_sampler = get_dataloader(cfg,'test')

    model, ckpt = load_vqvae_voxel_diffusion(cfg)

    optimizer, lr_scheduler = load_optim_sched(cfg, model, ckpt)
    logger.info("Training with config {}", cfg.config)
    logger.info("Trunc_thres:{}",cfg.data.trunc_thres)
    if cfg.data.trunc_thres is not None:
        cfg.data.trunc_distance = cfg.data.trunc_thres
    #cfg.data.trunc_distance=cfg.data.trunc_thres
    ampscaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    train_iter = save_iter(train_loader, train_sampler)
    torch.cuda.empty_cache()
    logger.info("Setup training and evaluation iterators.")
    print(len(train_loader))
    for step in range(cfg.start_step, cfg.training.steps):
        optimizer.zero_grad()

        # update the sampler for multi-node training
        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(train_loader))

        loss_accum = torch.tensor(0.0, dtype=torch.float32, device=cfg.local_rank)

        for accum_iter in range(cfg.training.accumulation_steps):
            next_batch = next(train_iter)
            scan_id, input_sdf, gt_df = next_batch
            input_sdf=input_sdf.to(cfg.local_rank)
            gt_df=gt_df.unsqueeze(1).to(cfg.local_rank) #[B,1,64,64,64]
            #print(gt_df.shape)
            loss = model(x0=gt_df, x1=input_sdf)
            loss /= cfg.training.accumulation_steps
            loss_accum += loss.detach()
            ampscaler.scale(loss).backward()

        ampscaler.unscale_(optimizer)
        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        ampscaler.step(optimizer)
        ampscaler.update()
        lr_scheduler.step()
        #for name, param in model.named_parameters():
        #    if param.grad is None:
        #        print(f"Parameter {name} is not used in the computation graph.")
        #break
        if cfg.distribution_type == "multi":
            dist.all_reduce(loss_accum)

        if step % cfg.training.log_interval == 0 and is_main_process:
            loss_accum /= cfg.global_size
            loss_accum = loss_accum.item()
            netpNorm, netgradNorm = getGradNorm(model.model)

            logger.info(
                "[{:>3d}/{:>3d}]\tloss: {:>10.6f} \t" "netpNorm: {:>10.2f},\tnetgradNorm: {:>10.4f}\t",
                step,
                cfg.training.steps,
                loss_accum,
                netpNorm,
                netgradNorm,
            )

        if (step + 1) % cfg.training.save_interval == 0 and (step + 1)<=cfg.training.log_interval*200:
            if is_main_process:
                save_dict = {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(save_dict, "%s/step_%d.pth" % (cfg.output_dir, step + 1))
                logger.info("Saved checkpoint to {}", cfg.output_dir)

            if cfg.distribution_type == "multi":
                dist.barrier()
                map_location = {"cuda:%d" % 0: "cuda:%d" % cfg.local_rank}
                model.load_state_dict(
                    torch.load(
                        "%s/step_%d.pth" % (cfg.output_dir, step + 1),
                        map_location=map_location,
                    )["model_state"]
                )

        if (step + 1) % cfg.training.viz_interval == 0:
            if cfg.distribution_type == "multi":
                dist.barrier()

            model.eval()
            if is_main_process:
                try:
                    logger.info('Skip Evaluation.')
                    #logger.info('Starting to evaluate test data for generate net')
                    #test_save_result_path = os.path.join(cfg.save_dir, "%d_steps/step_%dpth" % (cfg.diffusion.sampling_timesteps, step+1))
                    #logger.info('Visualize results will be stored in %s' % test_save_result_path)
                    #evaluate_sdf(model, val_loader, cfg, test_save_result_path, cfg.v, step+1)
                except Exception as e:
                    # print traceback and continue
                    print(sys.exc_info())
                    logger.warning("Could not evaluate model. Skipping.")
                    logger.warning(e)

            torch.cuda.empty_cache()
            model.train()

if __name__ == "__main__":
    opt = parse_args()

    # save the opt to output_dir
    save_data = DictConfig({})
    save_data.data = opt.data
    save_data.diffusion = opt.diffusion
    save_data.model = opt.model
    save_data.training = opt.training
    OmegaConf.save(save_data, os.path.join(opt.output_dir, "opt.yaml"))

    opt.ngpus_per_node = torch.cuda.device_count()

    torch.set_float32_matmul_precision("high")

    if opt.distribution_type == "multi":
        # setup configurations
        opt.world_size = opt.ngpus_per_node * opt.world_size
        opt.training.bs = int(opt.training.bs / opt.ngpus_per_node)
        mp.spawn(init_processes, nprocs=opt.world_size, args=(opt.world_size, train, opt))
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        opt.gpu = 0
        train(opt)
