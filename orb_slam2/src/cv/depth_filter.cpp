
#include "orb_slam/cv/depth_filter.h"

#include <boost/math/distributions/normal.hpp>

namespace orbslam {

using namespace std;

Seed::Seed(float depth_mean, float depth_min, const cv::Point2f &uv_)
    : // batch_id(batch_counter), id(seed_counter++),
      a(10), b(10), mu(1.0 / depth_mean), z_range(20.0 / depth_min),
      sigma2(z_range * z_range / 36), patch_cov(Eigen::Matrix2d::Identity()),
      uv(uv_.x, uv_.y), converged(false) {
  // using namespace std;
  cout << "construction" << endl;

  // cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.x - cx) * invfx,
  //                (kp1.y - cy) * invfy, 1.0);
  throw std::runtime_error("not in use");
}

Seed::Seed(float depth_mean, float depth_min, const KeyFrame *pKF,
           const int idx)
    : // batch_id(batch_counter), id(seed_counter++),
      a(10), b(10), mu(1.0 / depth_mean), z_range(1.0 / depth_min),
      sigma2(z_range * z_range /36.0f ), patch_cov(Eigen::Matrix2d::Identity()),
      uv(pKF->mvKeysUn[idx].pt.x, pKF->mvKeysUn[idx].pt.y), converged(false) {
  // constrution

  f = (cv::Mat_<float>(3, 1) << (uv.x - pKF->cx) * pKF->invfx,
       (uv.y - pKF->cy) * pKF->invfy, 1.0);
}

void Seed::updateSeed(float x, float tau2) {
  float norm_scale = sqrt(sigma2 + tau2);
  if (std::isnan(norm_scale))
    return;
  boost::math::normal_distribution<float> nd(mu, norm_scale);
  float s2 = 1. / (1. / sigma2 + 1. / tau2);
  float m = s2 * (mu / sigma2 + x / tau2);
  float C1 = a / (a + b) * boost::math::pdf(nd, x);
  float C2 = b / (a + b) * 1. / z_range;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f = C1 * (a + 1.) / (a + b + 1.) + C2 * a / (a + b + 1.);
  float e = C1 * (a + 1.) * (a + 2.) / ((a + b + 1.) * (a + b + 2.)) +
            C2 * a * (a + 1.0f) / ((a + b + 1.0f) * (a + b + 2.0f));

  // update parameters
  float mu_new = C1 * m + C2 * mu;
  sigma2 = C1 * (s2 + m * m) + C2 * (sigma2 + mu * mu) - mu_new * mu_new;
  mu = mu_new;
  a = (e - f) / (f - e / f);
  b = a * (1.0f - f) / f;

  if (sqrt(sigma2) < z_range / 200.0)
    converged = true;
}
// void DepthFilter::searchEpipolar(KeyFrame *pKF1, KeyFrame *pKF2) {

// float d_kf_median = pKF1->scene_depth_median;
// float d_kf_min = pKF1->scene_depth_min;
// float d_kf_max = pKF1->scene_depth_max;

// range =

// for (size_t i = 0; i < pKF1->N; i++) {
//   if (pKF1->GetMapPoint(i))
//     continue;

//   pKF1
// }

// // Compute epipole in second image
// cv::Mat Cw = pKF1->GetCameraCenter(); // twc1
// cv::Mat R2w = pKF2->GetRotation();    // Rc2w
// cv::Mat t2w = pKF2->GetTranslation(); // tc2w
// const float invz = 1.0f / C2.at<float>(2);
// const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
// const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

// // Epipolar line in second image l = x1'F12 = [a b c]
// const float a = kp1.pt.x * F12.at<float>(0, 0) +
//                 kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
// const float b = kp1.pt.x * F12.at<float>(0, 1) +
//                 kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
// const float c = kp1.pt.x * F12.at<float>(0, 2) +
//                 kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);
// }

// vo

// void DepthFilter::addFrame(Frame *pFrame)
// {

// }

// void DepthFilter::addKeyFrame(KeyFrame *pKF)
// {
//   // pKF

// }

} // namespace orbslam
