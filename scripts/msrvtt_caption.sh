echo "download weight"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/weight
mv weight video_caption/

echo "download data"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/MSRVTT_CLIP4Clip_features.pickle
mv MSRVTT_CLIP4Clip_features.pickle video_caption/extracted_feats/msrvtt/

echo "download bert"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/bert-base-uncased
mv bert-base-uncased video_caption/modules/
cd video_caption/scripts

bash train_msrvtt.sh