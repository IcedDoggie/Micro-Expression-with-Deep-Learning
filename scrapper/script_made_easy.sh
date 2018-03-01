#!/bin/bash

echo "Initializing Download Sequence..."

echo "Downloading Anaconda"
wget -P /home/$USER/Downloads/ "https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh"
echo "Printing Hash"
md5sum /home/$USER/Downloads/Anaconda3-4.4.0-Linux-x86_64.sh
echo "Installing Anaconda"
bash ./Downloads/Anaconda3-4.4.0-Linux-x86_64.sh
echo "Anaconda Installed"

echo "Downloading opencv2"
conda install -c menpo opencv3

echo "Downloading Pip3"
sudo apt-get -y install python3-pip
sudo apt-get update

echo "Downloading Theano"
conda install theano pygpu

echo "Downloading Tensorflow"
sudo pip3 install tensorflow-gpu
