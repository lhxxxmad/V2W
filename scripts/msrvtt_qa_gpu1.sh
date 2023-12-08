cd data
echo "download data"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/MSRVTT.tar.gz
tar -zxvf MSRVTT.tar.gz
cd ..
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/EMCL-Net/tvr/models/ViT-B-32.pt
mv ViT-B-32.pt ./tvr/models

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
split_hosts=($split_hosts)

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
main_qa.py \
--do_train \
--num_thread_reader=0 \
--epochs=5 \
--batch_size=32 \
--n_display=50 \
--train_csv data/MSRVTT/train.jsonl \
--val_csv data/MSRVTT/test.jsonl \
--data_path data/MSRVTT/train_ans2label.json \
--features_path data/MSRVTT/MSRVTT_Videos \
--output_dir ckpts/msrvtt_qa \
--lr 1e-4 --max_words 32 \
--max_frames 12 \
--batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--n_gpu 1 \
--linear_patch 2d \
--sal_predictor mlp \
--sample_num 2 \
--output_dir outputs/msrvtt_qa \




