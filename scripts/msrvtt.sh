# mkdir data
# cd data
# echo "download data"
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/MSRVTT.tar.gz
# tar -zxvf MSRVTT.tar.gz
# cd ..
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/EMCL-Net/tvr/models/ViT-B-32.pt
# mv ViT-B-32.pt ./tvr/models


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
# split_hosts=($split_hosts)

master_addr=127.0.0.4
master_port=29504

DATA_PATH=/comp_robot/caomeng/code/SaLIP/data/MSRVTT
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 \
--master_addr $master_addr \
--master_port $master_port \
main.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 512 \
--batch_size_val 256 \
--anno_path ${DATA_PATH}/msrvtt_data \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir outputs/msrvtt_v2w \
--embd_mode wti \
--interact_mode FGW \
--do_gauss 0 \
--sal_predictor mlp \
--sample_num 4 \
--pseudo_ret_weight 10 \
# --freeze_clip 1
# --ot_weight 0.1

echo "test model"
DATA_PATH=/comp_robot/caomeng/code/SaLIP/data/MSRVTT
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 \
--master_addr $master_addr \
--master_port $master_port \
main.py \
--do_eval 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 512 \
--batch_size_val 256 \
--anno_path ${DATA_PATH}/msrvtt_data \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--init_model outputs/msrvtt_v2w/best.bin \
--output_dir outputs/msrvtt_v2w \
--embd_mode wti \
--interact_mode FGW \
--do_gauss 0 \
--sal_predictor mlp \
--sample_num 4 \
--pseudo_ret_weight 10 \
# --freeze_clip 1
# --ot_weight 0.1

