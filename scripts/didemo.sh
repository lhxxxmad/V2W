master_addr=127.0.0.4
master_port=29504

DATA_PATH=/comp_robot/caomeng/data/Didemo
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr $master_addr \
--master_port $master_port \
main.py \
--do_train 1 \
--workers 8 \
--n_display 1 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ \
--video_path ${DATA_PATH}/videos \
--datatype didemo \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir outputs/didemo \
--embd_mode wti \
--interact_mode FGW \
--do_gauss 0 \
--sal_predictor mlp \
--sample_num 4 \
--dist_weight 0.5


echo "test model"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr $master_addr \
--master_port $master_port \
main.py \
--do_eval 1 \
--workers 8 \
--n_display 10 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ \
--video_path ${DATA_PATH}/videos \
--datatype didemo \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--init_model outputs/didemo/best.bin \
--output_dir outputs/didemo \
--embd_mode wti \
--interact_mode FGW \
--do_gauss 0 \
--sal_predictor mlp \
--sample_num 4 \
--dist_weight 0.5
