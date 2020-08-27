#pragma once

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/se3_ops.h"

#include "g2o/types/sba/types_six_dof_expmap.h"

#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include <iostream>

namespace g2o {
namespace types_dust_tracking {
void init();
}

// Edge to optimize only the camera pose
class EdgeSE3ProjectDustOnlyPose
    : public BaseUnaryEdge<1, double, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectDustOnlyPose() : BaseUnaryEdge<1, double, VertexSE3Expmap>() {}

  void setDustData(const cv::Mat *dust) {
    dust_ = dust;
    w_ = dust_->cols;
    h_ = dust_->rows;
  }

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError();

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 =
        static_cast<const VertexSE3Expmap *>(_vertices[0]);
    return (v1->estimate().map(Xw))(2) > 0;
  }

  virtual void linearizeOplus();

  // Vector2 cam_project(const Vector3 &trans_xyz) const;

  // get a gray scale value from reference image (bilinear interpolated)
  float getPixelValue(float x, float y);

  const bool isInImage(const double &u, const double &v,
                       double border = 1.0) const;

  Vector3 Xw;
  number_t fx, fy, cx, cy;
  float u_, v_;

  const cv::Mat *dust_ = nullptr;
  float w_, h_;
  int count = 0;
};

} // namespace g2o
