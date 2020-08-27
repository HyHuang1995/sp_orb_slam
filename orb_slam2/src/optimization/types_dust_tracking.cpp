#include "orb_slam/optimization/types_dust_tracking.h"

#include <g2o/core/factory.h>

#include <iostream>

using namespace std;

namespace g2o {

G2O_REGISTER_TYPE_GROUP(dust);
G2O_REGISTER_TYPE(EDGE_SE3_PROJECT_DUSTONLYPOSE
                  : DUST, EdgeSE3ProjectDustOnlyPose);

// Vector2 project2d(const Vector3 &v) {
//   Vector2 res;
//   res(0) = v(0) / v(2);
//   res(1) = v(1) / v(2);
//   return res;
// }

// Vector3 unproject2d(const Vector2 &v) {
//   Vector3 res;
//   res(0) = v(0);
//   res(1) = v(1);
//   res(2) = 1;
//   return res;
// }

// inline Vector3 invert_depth(const Vector3 &x) {
//   return unproject2d(x.head<2>()) / x[2];
// }

// const bool EdgeSE3ProjectDust

const bool EdgeSE3ProjectDustOnlyPose::isInImage(const double &u,
                                                 const double &v,
                                                 double border) const {
  return (u >= border && u + border + 1 < w_ && v >= border &&
          v + border + 1 < h_);
}

float EdgeSE3ProjectDustOnlyPose::getPixelValue(float x, float y) {

  const int x_f = floor(x);
  const int y_f = floor(y);

  float xx = x - x_f;
  float yy = y - y_f;

  return float((1 - xx) * (1 - yy) * dust_->at<float>(y_f, x_f) +
               xx * (1 - yy) * dust_->at<float>(y_f, x_f + 1) +
               (1 - xx) * yy * dust_->at<float>(y_f + 1, x_f) +
               xx * yy * dust_->at<float>(y_f + 1, x_f + 1));
  return 0;
}

bool EdgeSE3ProjectDustOnlyPose::read(std::istream &is) {}

bool EdgeSE3ProjectDustOnlyPose::write(std::ostream &os) const {}

void EdgeSE3ProjectDustOnlyPose::computeError() {
  // count ++;
  const VertexSE3Expmap *vi =
      static_cast<const VertexSE3Expmap *>(_vertices[0]);

  // cout << "xw: " << Xw.transpose() << endl;
  // cout << vi->estimate().to_homogeneous_matrix() << endl;
  Eigen::Vector3d x_local = vi->estimate().map(Xw);
  // if (x)
  const auto z = x_local[2];
  if (z < 0.0) {
    _error(0, 0) = 0.0;
    this->setLevel(1);
    return;
  }

  // cout << "x local: " << x_local.transpose() << endl;
  double x = x_local[0] * fx / x_local[2] + cx;
  double y = x_local[1] * fy / x_local[2] + cy;

  // check x,y is in the image
  if (!isInImage(x, y)) {
    _error(0, 0) = 0.0;
    this->setLevel(1);
  } else {
    // _error(0, 0) = getPixelValue(x, y) - _measurement;
    // cout << "what " << x << ' ' << y << endl;
    _error(0, 0) = getPixelValue(x, y);

    u_ = x;
    v_ = y;
  }
}

void EdgeSE3ProjectDustOnlyPose::linearizeOplus() {
  if (level() == 1) {
    _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
    return;
  }

  const VertexSE3Expmap *vi =
      static_cast<const VertexSE3Expmap *>(_vertices[0]);
  Vector3 xyz_trans = vi->estimate().map(Xw);

  number_t x = xyz_trans[0];
  number_t y = xyz_trans[1];
  number_t invz = 1.0 / xyz_trans[2];
  number_t invz_2 = invz * invz;

  double u = x * fx * invz + cx;
  double v = y * fy * invz + cy;

  if (!isInImage(u, v)) {
    throw std::runtime_error(" should be omitted");
  }
  Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

  jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx;
  jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx;
  jacobian_uv_ksai(0, 2) = -y * invz * fx;
  jacobian_uv_ksai(0, 3) = invz * fx;
  jacobian_uv_ksai(0, 4) = 0;
  jacobian_uv_ksai(0, 5) = -x * invz_2 * fx;

  jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy;
  jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy;
  jacobian_uv_ksai(1, 2) = x * invz * fy;
  jacobian_uv_ksai(1, 3) = 0;
  jacobian_uv_ksai(1, 4) = invz * fy;
  jacobian_uv_ksai(1, 5) = -y * invz_2 * fy;

  Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

  jacobian_pixel_uv(0, 0) =
      (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2.0f;
  jacobian_pixel_uv(0, 1) =
      (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2.0f;

  _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
}

// Vector2
// EdgeSE3ProjectDustOnlyPose::cam_project(const Vector3 &trans_xyz) const {
//   Vector2 proj = project2d(trans_xyz);
//   Vector2 res;
//   res[0] = proj[0] * fx + cx;
//   res[1] = proj[1] * fy + cy;
//   return res;
// }

} // namespace g2o
