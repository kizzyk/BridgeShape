import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import time
from dataloaders.dataloader import get_dataloader, save_iter
from models.evaluation import evaluate_sdf
from models.model_voxel_loader import load_optim_sched,load_vqvae_voxel_diffusion
from models.train_utils import get_data_batch, getGradNorm, set_seed, setup_output_subdirs, to_cuda
from utils.args import parse_args
#os.environ["CUDA_VISIBLE_DEVICES"]='1'

def test(cfg: DictConfig) -> None:
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

    if cfg.tbs!=0:
        cfg.evaluation.bs=cfg.tbs
    logger.info('Evaluation batch_size: %d' % cfg.evaluation.bs)
    val_loader,val_sampler = get_dataloader(cfg,'test')
    if cfg.rs!=0:
        cfg.diffusion.sampling_timesteps= cfg.rs
        logger.info(f'Reverse_sample_timesteps: {cfg.rs}')
    logger.info("OT_ODE:{}",cfg.diffusion.ot_ode)
    cfg.data.trunc_distance=cfg.data.trunc_thres
    start_epoch=cfg.test_start_epoch*cfg.training.log_interval//cfg.training.save_interval #50 epoch
    end_epoch=cfg.training.steps//cfg.training.save_interval+1 #end epoch
    if cfg.test_one:
        end_epoch=start_epoch+1
    logger.info("Evaluation epoch : from %d to %d"%(start_epoch,end_epoch-1))
    for i in range(start_epoch,end_epoch):
        cfg.model_path = os.path.join(cfg.output_dir,f"step_{(i*cfg.training.save_interval)}.pth")#output/coarse/__step.pth
        model, ckpt = load_vqvae_voxel_diffusion(cfg)
        logger.info("Training with config {}", cfg.config)

        torch.cuda.empty_cache()
        logger.info("Setup evaluation iterators.")

        model.eval()

        try:
            logger.info('Starting to evaluate test data for generate net' )
            test_save_result_path = os.path.join(cfg.save_dir, "%d_steps/step_%dpth" % (cfg.diffusion.sampling_timesteps, i * cfg.training.save_interval))
            logger.info('Visualize results will be stored in %s' % test_save_result_path)
            evaluate_sdf(model, val_loader, cfg,test_save_result_path,cfg.v,i * cfg.training.save_interval)
        except Exception as e:
            # print traceback and continue
            logger.info(sys.exc_info())
            logger.warning("Could not evaluate model. Skipping.")
            logger.warning(e)

        torch.cuda.empty_cache()
    # wandb.finish()

if __name__ == "__main__":
    opt = parse_args()

    # save the opt to output_dir
    save_data = DictConfig({})
    save_data.data = opt.data
    save_data.diffusion = opt.diffusion
    save_data.model = opt.model
    save_data.sampling = opt.sampling
    save_data.training = opt.training
    OmegaConf.save(save_data, os.path.join(opt.output_dir, "opt.yaml"))

    opt.ngpus_per_node = torch.cuda.device_count()

    torch.set_float32_matmul_precision("high")

    torch.cuda.set_device(0)
    opt.global_rank = 0
    opt.local_rank = 0
    opt.global_size = 1
    opt.gpu = 0
    test(opt)

