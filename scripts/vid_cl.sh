export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128
export no_proxy=code.byted.org



git clone https://github.com/lhxxxmad/AIM.git

# pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

sudo pip install git+https://github.com/openai/CLIP.git

cd AIM

# install other requirements
sudo pip install -r requirements.txt

# install mmaction2
sudo python3 setup.py develop


git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
sudo pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
sudo pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ..
ln -s /mnt/bd/dataset0131/MIL_data  data
sudo bash run_exp.sh