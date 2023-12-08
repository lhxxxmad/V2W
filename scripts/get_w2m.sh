cd /mnt/bd/dataset-1017
git clone https://github.com/m-bain/webvid.git
cd webvid
cd data
wget http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv
wget http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv

pip install pandas numpy requests mpi4py

cd ..
python3 download.py --csv_path data/results_2M_train.csv --partitions 1 --part 0 --data_dir ./data --processes 8
python3 download.py --csv_path data/results_2M_val.csv --partitions 1 --part 0 --data_dir ./data --processes 8