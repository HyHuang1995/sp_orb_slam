#pragma once

#include "tracker.h"

namespace orbslam {

class MonoTracker : public Tracking {

public:
  MonoTracker() : Tracking() {}


  // ADD(debug)
  cv::Mat mImInit;

protected:
  void setFrameData(const double &timestamp, const cv::Mat &im1,
                    const cv::Mat &im2 = cv::Mat()) override final;

  virtual void Initialization() override final;

  void CreateInitialMap();
};

} // namespace orbslam