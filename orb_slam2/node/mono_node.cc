#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>

#include "orb_slam/type/type.h"

#include <orb_slam/system.h>

#include "orb_slam/common.h"
#include "orb_slam/config.h"
#include "orb_slam/global.h"
#include "orb_slam/init_cfg.hpp"
#include "orb_slam/utils/timing.h"

using namespace std;
using namespace orbslam;

int main(int argc, char **argv) {
  ros::init(argc, argv, "orb_slam");
  // GOOGLE_PROTOBUF_VERIFY_VERSION;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);

  ros::NodeHandle nh("~");
  orbslam::initParameters(nh);

  orbslam::System SLAM(orbslam::common::sensor, common::visualize);
  SLAM.spin();

  SLAM.Shutdown();

  // Save camera trajectory

  return 0;
}
