#pragma once

#include <string>
#include <thread>

#include <opencv2/core/core.hpp>

#include "io/data_loader.h"
#include "type/type.h"

namespace orbslam {

class System {

public:
  System(const eSensor sensor, const bool bUseViewer = true);

  ~System();

  cv::Mat run(const double &timestamp, const cv::Mat &im0,
              const cv::Mat &im1 = cv::Mat());

  void spin();

  void resetSystem();

  void Shutdown();

  void SaveTrajectoryTUM(const std::string &filename);

  void SaveKeyFrameTrajectoryTUM(const std::string &filename);

  void SaveTrajectoryKITTI(const std::string &filename);
  void SaveTrajectoryEuroc(const std::string &filename);

private:
  Dataloader::Ptr loader_;

  eSensor mSensor;

  std::unique_ptr<std::thread> mptLocalMapping = nullptr;
  std::unique_ptr<std::thread> mptLoopClosing = nullptr;
  std::unique_ptr<std::thread> mptViewer = nullptr;
};

} // namespace orbslam
