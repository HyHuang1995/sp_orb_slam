#! /bin/bash

wget -c https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu92.zip 
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cu92.zip
mv libtorch orb_slam2/3rdparty

wstool init .. ./.spslam_https.install
wstool update

catkin build orb_ros
