export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

echo "fint_tune model"
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node=2 \
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
--batch_size_val 16 \
--anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSRVTT/msrvtt_data \
--video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSRVTT/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir outputs/msrvtt_ft \
--embd_mode cyc \
--do_gauss 0 \
--video_mask_rate 0.7 \
--text_mask_rate 0.7 \
--temp_loss_weight 1.0 \
--rec_loss_weight 1.0 \
--ret_loss_weight 1.0 \
--sal_predictor mlp \
--sample_num 2 \
--mergeclip=False \
--base_encoder ViT-L/14 \
--clip_eval_path /mnt/bd/cxx-dataset/InternVideo/Downstream/Video-Text-Retrieval/InternVideo-MM-L-14.ckpt
