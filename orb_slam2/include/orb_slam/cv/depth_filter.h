#pragma once

#include <memory>

#include "../type/frame.h"
#include "../type/keyframe.h"

namespace orbslam {

class Frame;

class KeyFrame;

/// A seed is a probabilistic depth estimate for a single pixel.
class Seed {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // static float computeTau(const cv::Mat &T_ref_cur, const );

  using Ptr = std::shared_ptr<Seed>;

  Seed(float depth_mean, float depth_min, const cv::Point2f &uv_);

  Seed(float depth_mean, float depth_min, const KeyFrame *pKF, const int idx);

  // TODO:svo 309
  void updateSeed(float x, float tau2);

  // static int batch_counter;
  // static int seed_counter;
  // int batch_id;                //!< Batch id is the id of the keyframe for
  // which the seed was created. int id;                      //!< Seed ID, only
  // used for visualization. Feature* ftr;                //!< Feature in the
  // keyframe for which the depth should be computed.
  float
      a; //!< a of Beta distribution: When high, probability of inlier is large.
  float b;  //!< b of Beta distribution: When high, probability of outlier is
            //!< large.
  float mu; //!< Mean of normal distribution.
  float z_range;             //!< Max range of the possible depth.
  float sigma2;              //!< Variance of normal distribution.
  Eigen::Matrix2d patch_cov; //!< Patch covariance in reference image.

  cv::Point2f uv;
  cv::Mat f; // image plane

  bool converged;
};

class DepthFilter {
public:
  static void searchEpipolar(KeyFrame *pKF1, Frame *pFrame);

  // public:
  // DepthFilter(/* args */);

  // ~DepthFilter();

  // void

  // void addFrame(Frame *pFrame);

  // void addKeyFrame(KeyFrame *KeyFrame);

  // void run();
};

// DepthFilter::DepthFilter(/* args */) {}

// DepthFilter::~DepthFilter() {}

} // namespace orbslam
