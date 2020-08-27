#include "orb_slam/system.h"

#include <iomanip>  // std::setprecision
#include <iostream> // std::cout, std::fixed
#include <memory>
#include <thread>

#include <pangolin/pangolin.h>

#include "orb_slam/common.h"
#include "orb_slam/config.h"
#include "orb_slam/global.h"

#include "orb_slam/type/type.h"

#include "orb_slam/tracking/mono_tracker.h"
// #include "orb_slam/tracking/rgbd_tracker.h"
// #include "orb_slam/tracking/stereo_tracker.h"

#include "orb_slam/utils/converter.h"

// #include <fmt/core.h>
// #include <fmt/format.h>

bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}

namespace orbslam {

using namespace std;

System::~System() {
  // cout << "a" << endl;

  if (mptLocalMapping != nullptr) {
    mptLocalMapping->join();
  }
  if (mptLoopClosing != nullptr) {
    mptLoopClosing->join();
  }

  cout << "viewer.." << endl;
  if (mptViewer != nullptr) {
    mptViewer->join();
  }
}

System::System(const eSensor sensor, const bool bUseViewer)
    : loader_(nullptr), mSensor(sensor) {

  if (mSensor != MONOCULAR) {
    throw std::runtime_error("implementation for monocular only");
  }

  // if (common) {
  //   throw std::runtime_error("implementation for monocular only");
  // }

  // cout << endl << "Loading Vocabulary. This could take a while..." << endl;
  // LOG(WARNING) << "logging" << endl;

  // global::vocabulary = new ORBVocabulary();
  // bool bVocLoad = false; // chose loading method based on file extension
  //                        //   if (has_suffix(common::voc_path, ".txt"))
  // global::vocabulary->load(common::voc_path);
  // cout << "Vocabulary loaded!" << endl << endl;

  // Create KeyFrame Database
  // global::keyframe_db = new KeyFrameDatabase();

  // Create the Map
  global::map = new Map();

  // Create Drawers. These are used by the Viewer
  global::frame_drawer = new FrameDrawer();
  global::map_drawer = new MapDrawer();

  global::frame_drawer->setMap(global::map);
  global::map_drawer->setMap(global::map);

  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  if (mSensor == MONOCULAR) {
    global::tracker = new MonoTracker();
  } else {
    throw std::runtime_error("no implementation");
  }

  // Initialize the Local Mapping thread and launch
  global::mapper = new LocalMapping(mSensor == MONOCULAR);
  if (common::online) {
    mptLocalMapping = std::unique_ptr<std::thread>(
        new thread(&orbslam::LocalMapping::Run, global::mapper));
  }

  global::looper = new LoopClosingVLAD(mSensor != MONOCULAR);
  if (common::use_loop) {
    cout << "Starting Loop Closing" << endl;
    mptLoopClosing = std::unique_ptr<std::thread>(
        new thread(&orbslam::LoopClosingVLAD::Run, global::looper));
  } else {
    cout << "Loop Closing Disabled" << endl;
  }

  global::viewer = new Viewer();
  global::viewer->setFrameDrawer(global::frame_drawer);
  global::viewer->setMapDrawer(global::map_drawer);

  if (common::visualize) {
    mptViewer =
        std::unique_ptr<std::thread>(new thread(&Viewer::Run, global::viewer));
  }

  LOG(INFO) << "dataloader initialize..";
  // loader_ = Dataloader::Ptr(new DataloaderKITTIExport(
  //     orbslam::common::data_path, common::seq, DataType::GT | DataType::Mono
  //     | DataType::Depth));
  if (common::dataset == "euroc") {
    loader_ = Dataloader::Ptr(
        new DataloaderEuRoC(orbslam::common::data_path,
                            DataType::GT | DataType::Mono | DataType::Depth));
  } else if (common::dataset == "kitti") {
    loader_ = Dataloader::Ptr(new DataloaderKITTIExport(
        orbslam::common::data_path, common::seq,
        DataType::GT | DataType::Mono | DataType::Depth));
  } else if (common::dataset == "tsukuba") {
    loader_ = Dataloader::Ptr(new DataloaderTsukuba(
        orbslam::common::data_path, common::seq, DataType::Mono));
  }

  LOG(INFO) << "dataloader done. "
            << "total #of frames: " << loader_->getSize();
}

void System::spin() {
  cv::Mat imRGB, imD;

  ros::Rate rate(camera::fps);

  int ni = 0;

  // getchar();

  while (ros::ok()) {

    if (!global::b_pause || global::b_step) {

      auto data_frame = loader_->getNextFrame();

      // cv::imshow("win", data_frame->mono);
      // cv::waitKey(-1);

      if (!data_frame) {
        LOG(INFO) << "end of dataset, break..";
        break;
      }
      data_frame->mono =
          data_frame->mono(cv::Rect(0, 0, camera::width, camera::height));

      LOG(INFO) << "current frame idx: " << data_frame->idx
                << " stamp: " << (uint64_t)(data_frame->timestamp * 1e9);

      // auto info_str = fmt::format("frame size: {}x{}, desc size: {}x{}",
      //                             data_frame->mono.cols,
      //                             data_frame->mono.rows,
      //                             data_frame->global_desc.cols,
      //                             data_frame->global_desc.rows);
      // auto info_str = fmt::format("frame size: {}, desc size: {}", 1, 2);
      // LOG(INFO) << info_str;

      auto res = global::tracker->trackFrame(data_frame);

      if (!common::online) {
        global::mapper->spinOnce();
      }
    }
    if (global::b_step)
      global::b_step = false;

    if (global::b_system_reset) {
      // global::tracker->Reset();
      resetSystem();
      global::b_system_reset = false;
    }

    ros::spinOnce();
    if (common::online) {
      rate.sleep();
    }
  }

  // getchar();

  SaveTrajectoryTUM(common::output_path + "/traj.txt");
  SaveKeyFrameTrajectoryTUM(common::output_path + "/kf.txt");
  // SaveTrajectoryTUM(common::output_path + "/traj.txt");
}

cv::Mat System::run(const double &timestamp, const cv::Mat &im0,
                    const cv::Mat &im1) {
  if (global::b_local_on) {
    global::mapper->RequestStop();

    // Wait until Local Mapping has effectively stopped
    while (!global::mapper->isStopped()) {
      // usleep(1000);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    global::tracker->InformOnlyTracking(true);
    global::b_local_on = false;
  }
  if (global::b_local_off) {
    global::tracker->InformOnlyTracking(false);
    global::mapper->Release();
    global::b_local_off = false;
  }

  // check system reset option
  if (global::b_system_reset) {
    // global::tracker->Reset();
    resetSystem();
    global::b_system_reset = false;
  }

  auto res = global::tracker->trackFrame(timestamp, im0, im1);
  // global::frame_drawer->Update(global::tracker);

  return res;
}

} // namespace orbslam
