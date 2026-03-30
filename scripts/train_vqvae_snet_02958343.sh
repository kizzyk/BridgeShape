# train_pvqvae_nembed-8192-z-3x16x16x16-snet_ngpus.sh
export OMP_NUM_THREADS=4
RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='../logs_home_32'

### set gpus ###
#gpu_ids=0          # single-gpu
gpu_ids=6  # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    # NGPU=2
    NGPU=2
    # NGPU=4
    PORT=11768
    echo "HERE"
fi
################

### hyper params ###
lr=1e-4
batch_size=32
####################

### model stuff ###
model="vqvae"
vq_cfg="../configs/vqvae_snet.yaml"
####################


### dataset stuff ###
max_dataset_size=10000000
dataset_mode='ControlledEPNDataset_32'
data_root="/workspace/EPN/control_data"
res=32
cat='all'
# cat='chair'
trunc_thres=3.0
#workers: 4
per_class='True'
class_id='02958343'
suffix='.pth'
log_df='False'
representation='tsdf'

###########################

### display & log stuff ###
display_freq=250000 # default: display_current_results
print_freq=2000 # default: print_current_errors
total_iters=250000 #100000000
save_steps_freq=5000 #print_current_metrics
###########################

today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

note="release"
name="${model}-${dataset_mode}-${class_id}car-${cat}-res${res}-LR${lr}-T${trunc_thres}-${note}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=3
	max_dataset_size=12
    total_iters=1000000
    save_steps_freq=3
	display_freq=2
	print_freq=2
	# update_html_freq=$(( 1 *"$batch_size" ))
	# display_freq=$(( 1 *"$batch_size" ))
	# print_freq=$(( 1 *"$batch_size" ))
    name="DEBUG-${name}"
fi

cmd="../train_vq.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --vq_cfg ${vq_cfg} \
                --dataset_mode ${dataset_mode} --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} --max_dataset_size ${max_dataset_size} --per_class ${per_class} --representation ${representation} --suffix ${suffix} --class_id ${class_id} --per_class ${per_class} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
                --debug ${debug}"

if [ ! -z "data_root" ]; then
    cmd="${cmd} --data_root ${data_root}"
    echo "setting data_root to: ${data_root}"
fi

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi


multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.run --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Training with command: "
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

# exit

CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
