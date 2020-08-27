# SP_ORB_SLAM

[![Build Status](https://travis-ci.org/HyHuang1995/sp_orb_slam.svg?branch=master)](https://travis-ci.org/github/HyHuang1995/sp_orb_slam)
[![LICENSE](https://img.shields.io/badge/license-GPL%20(%3E%3D%202)-informational)](https://github.com/HyHuang1995/sp_orb_slam/blob/master/LICENSE)

Using Learnt Features in Indirect Visual SLAM. [[project]](https://sites.google.com/view/rdvo/)

## Paper and Video

Related publication:
```latex
@inproceedings{hyhuang2020rdvo,
  title={Monocular Visual Odometry using Learned Repeatability and Description},
  author={Huaiyang Huang, Haoyang Ye, Yuxiang Sun and Ming Liu},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020},
  organization={IEEE}
}
```

Demo videos:

<a href="https://www.youtube.com/watch?v=TqhAu1dxl9Y" target="_blank"><img src="https://www.ram-lab.com/image/hyhuang_icra2019_cover.png" 
alt="gmmloc" width="640" height="320" border="10" /></a>

## Prerequisites

We have tested this library in Ubuntu 18.04 with CUDA 9.2 and CUDA 10.1. Prerequisites for installation:

1. [ROS](http://wiki.ros.org/melodic/Installation) (ros-base is enough)
```
apt-get install ros-melodic-ros-base
```

2. miscs for installation:
```
apt-get install python-wstool python-catkin-tools 
```

3. [OpenCV3](https://docs.opencv.org/3.4.11/d7/d9f/tutorial_linux_install.html)
```
apt-get install libopencv-dev
```

4. [Pangolin](https://github.com/stevenlovegrove/Pangolin) (optional, for visualization)

4. [evo](https://github.com/MichaelGrupp/evo) (optional, for evaluation)
```
pip install evo --upgrade --no-binary evo
```

## Installation
Initialize a workspace:

```
mkdir -p /EXAMPLE/CATKIN/WORK_SPACE
cd /EXAMPLE/CATKIN/WORK_SPACE

mkdir src
catkin init
catkin config --extend /opt/ros/melodic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel
```

Clone the code:
```
cd src
git clone git@github.com:hyhuang1995/sp_orb_slam.git
```

Use the installation script:
```
cd sp_orb_slam
./install.sh
```

## Running Examples
We provide examples on the New Tsukuba and the EuRoC MAV dataset. To run the demo on the New Tsukuba sequences:

1. Download the [dataset](https://home.cvlab.cs.tsukuba.ac.jp/dataset)

2. Replace the **/PATH/TO/TSUKUBA/DATASET** in [tsukuba.launch](https://github.com/HyHuang1995/sp_orb_slam/blob/master/orb_ros/launch/tsukuba.launch) with where the sequence is decompressed:
```
<param name="data_path" value="/PATH/TO/TSUKUBA/DATASET" />
```

3. Launch:
```
roslaunch tsukuba.launch seq:=lamps
```

To run the demo on the EuRoC MAV dataset:
1. Download the [sequences](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (ASL Format)

2. Replace the **/PATH/TO/EUROC/DATASET/** in [euroc_mono.launch](https://github.com/HyHuang1995/sp_orb_slam/blob/master/orb_ros/launch/euroc_mono.launch) with where the sequence is decompressed:
```
<param name="data_path" value="/PATH/TO/EUROC/DATASET/$(arg seq)/mav0/" />
```

3. Launch:
```
roslaunch euroc_mono.launch seq:=MH_05_difficult
```
The output trajectories would be saved to **orb_ros/expr**.

## Evaluation
If evo is installed, we provide scripts for evaluating the VO performances.

```
roscd orb_ros
./scripts/evaluate_tsukuba.sh
```
or 
```
./scripts/evaluate_euroc.sh
```
and the results would be saved to **orb_ros/expr**.

## Credits

Our implementation is built on top of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), see the license in the source files for more details. The authors would like to thank Raul et al. for their great work.
