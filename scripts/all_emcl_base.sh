mkdir data
cd data
# echo "download anet"
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/anet.tar.gz
# tar -zxvf anet.tar.gz
# echo "download didemo"
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/didemo.zip 
# unzip didemo.zip  -d ./
echo "download msrvtt"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/MSRVTT.tar.gz
tar -zxvf MSRVTT.tar.gz
# echo "download lsmdc"
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/LSMDC.zip
# unzip LSMDC.zip
cd ..

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/EMCL-Net/tvr/models/ViT-B-32.pt
mv ViT-B-32.pt ./tvr/models

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
split_hosts=($split_hosts)

# echo "train anet"
# DATA_PATH=./data/anet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_train 1 \
# --do_eval 1 \
# --workers 8 \
# --n_display 5 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path ${DATA_PATH}/ \
# --video_path ${DATA_PATH}/Activity_Videos \
# --datatype activity \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --output_dir outputs/activity \
# --embd_mode emcl \
# --do_gauss 0 \
# --do_gauss 0 \
# --video_mask_rate 0.1 \
# --text_mask_rate 0.1 \
# --rec_trans_num_layers1 4 \
# --rec_trans_num_layers2 4 \
# --sal_predictor mlp \
# --sample_num 2 \
# --distribution 0

# echo "test anet"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_eval 1 \
# --workers 8 \
# --n_display 5 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path ${DATA_PATH}/ \
# --video_path ${DATA_PATH}/Activity_Videos \
# --datatype activity \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --init_model outputs/activity/best.bin \
# --output_dir outputs/activity \
# --embd_mode emcl \
# --do_gauss 0 \
# --video_mask_rate 0.1 \
# --text_mask_rate 0.1 \
# --rec_trans_num_layers1 4 \
# --rec_trans_num_layers2 4 \
# --tmp_trans_num_layers 4 \
# --sal_predictor mlp \
# --sample_num 2 \
# --distribution 0

# echo "train didemo"
# DATA_PATH=./data/didemo
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_train 1 \
# --workers 8 \
# --n_display 1 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path ${DATA_PATH}/ \
# --video_path ${DATA_PATH}/videos \
# --datatype didemo \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --output_dir outputs/didemo \
# --embd_mode emcl \
# --do_gauss 0 \
# --video_mask_rate 0.1 \
# --text_mask_rate 0.1 \
# --temp_loss_weight 1.0 \
# --rec_loss_weight 1.0 \
# --ret_loss_weight 1.0 \
# --sal_predictor mlp \
# --sample_num 2 \
# --distribution 0

# echo "test didemo"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_eval 1 \
# --workers 8 \
# --n_display 1 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path ${DATA_PATH}/ \
# --video_path ${DATA_PATH}/videos \
# --datatype didemo \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --init_model outputs/didemo/best.bin \
# --output_dir outputs/didemo \
# --embd_mode emcl \
# --do_gauss 0 \
# --video_mask_rate 0.1 \
# --text_mask_rate 0.1 \
# --sal_predictor mlp \
# --sample_num 2 \
# --distribution 0

echo "train msrvtt"
DATA_PATH=./data/MSRVTT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
main.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/msrvtt_data \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir outputs/msrvtt \
--embd_mode emcl \
--do_gauss 0 \
--video_mask_rate 0.7 \
--text_mask_rate 0.7 \
--temp_loss_weight 1.0 \
--rec_loss_weight 1.0 \
--ret_loss_weight 1.0 \
--sal_predictor mlp \
--sample_num 2 \
--distribution 0

echo "test msrvtt"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
main.py \
--do_eval 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/msrvtt_data \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--init_model outputs/msrvtt/best.bin \
--output_dir outputs/msrvtt \
--embd_mode emcl \
--do_gauss 0 \
--video_mask_rate 0.7 \
--text_mask_rate 0.7 \
--temp_loss_weight 1.0 \
--rec_loss_weight 1.0 \
--ret_loss_weight 1.0 \
--sal_predictor mlp \
--sample_num 2 \
--distribution 0

# echo "train lsmdc"
# DATA_PATH=./data/Clip_LSMDC
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_train 1 \
# --workers 8 \
# --n_display 5 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path ${DATA_PATH}/ \
# --video_path ${DATA_PATH}/LSMDC_Videos \
# --datatype lsmdc \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/lsmdc \
# --embd_mode emcl \
# --do_gauss 0 \
# --video_mask_rate 0.7 \
# --text_mask_rate 0.7 \
# --rec_trans_num_layers1 4 \
# --rec_trans_num_layers2 4 \
# --sal_predictor mlp \
# --sample_num 2 \
# --distribution 0

# echo "test lsmdc"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_eval 1 \
# --workers 8 \
# --n_display 5 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path ${DATA_PATH}/ \
# --video_path ${DATA_PATH}/LSMDC_Videos \
# --datatype lsmdc \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --init_model outputs/lsmdc/best.bin \
# --output_dir outputs/lsmdc \
# --embd_mode emcl \
# --do_gauss 0 \
# --video_mask_rate 0.7 \
# --text_mask_rate 0.7 \
# --rec_trans_num_layers1 4 \
# --rec_trans_num_layers2 4 \
# --sal_predictor mlp \
# --sample_num 2 \
# --distribution 0

hdfs dfs -put -f outputs hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/PMTVR_model/emcl