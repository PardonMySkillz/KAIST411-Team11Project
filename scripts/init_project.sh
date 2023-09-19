#!/bin/bash

apt-get update -y 
apt-get install -y wget make gcc python3 python3-dev python3-pip

# download cuda toolkit installer script
wget -c https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
chmod +x cuda_11.6.0_510.39.01_linux.run

# download python library for project
python3 -m pip install --upgrade pip
python3 -m pip install torch==1.13.1 jupyter numpy image

cd $(dirname ${BASH_SOURCE[0]})/..

# download yolov2 weight
mkdir -p ./data/weight
wget -O ./data/weight/yolov2-tiny-voc.weights https://pjreddie.com/media/files/yolov2-tiny-voc.weights
