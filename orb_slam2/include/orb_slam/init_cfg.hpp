#pragma once

#include <ros/ros.h>

#include "config.h"

namespace orbslam {

void initParameters(ros::NodeHandle &nh) {

#define GPARAM(x, y)                                                           \
  do {                                                                         \
    if (!nh.getParam(x, y)) {                                                  \
      LOG(WARNING) << ("retrive pararm " #x " error!");                        \
    }                                                                          \
  } while (0)

  GPARAM("sequence", common::seq);
  GPARAM("dataset", common::dataset);
  GPARAM("output_path", common::output_path);
  GPARAM("data_path", common::data_path);
  GPARAM("model_path", common::model_path);
  GPARAM("use_loop", common::use_loop);
  GPARAM("verbose", common::verbose);
  GPARAM("visualize", common::visualize);
  GPARAM("online", common::online);

  // int device;
  // GPARAM("sp/device", device);
  // sp::device_type = static_cast<Device>(device);
  // GPARAM("sp/weight_path", sp::weight_path);

  int sensor;
  GPARAM("sensor_type", sensor);
  common::sensor = static_cast<eSensor>(sensor);

  GPARAM("camera/fx", camera::fx);
  GPARAM("camera/fy", camera::fy);
  GPARAM("camera/cx", camera::cx);
  GPARAM("camera/cy", camera::cy);
  GPARAM("camera/fps", camera::fps);
  GPARAM("camera/width", camera::width);
  GPARAM("camera/height", camera::height);
  GPARAM("camera/is_rgb", camera::is_rgb);

  int d_type;
  std::vector<float> distortion;
  GPARAM("camera/distortion_type", d_type);
  if (d_type == 1) {
    GPARAM("camera/distortion", distortion);
    if (distortion.size() != 4) {
      throw std::runtime_error("distortion size mismatch");
    }

    camera::k1 = distortion[0];
    camera::k2 = distortion[1];
    camera::p1 = distortion[2];
    camera::p2 = distortion[3];
  } else if (d_type == 2) {
    GPARAM("camera/distortion", distortion);
    if (distortion.size() != 5) {
      throw std::runtime_error("distortion size mismatch");
    }

    camera::k1 = distortion[0];
    camera::k2 = distortion[1];
    camera::p1 = distortion[2];
    camera::p2 = distortion[3];
    camera::k3 = distortion[4];
  }

  int extractor_;
  GPARAM("tracking/extractor_type", extractor_);
  tracking::extractor_type = tracking::Extractor(extractor_);
  GPARAM("tracking/num_features", tracking::num_features);
  GPARAM("tracking/scale_check", tracking::scale_check);

  GPARAM("tracking/create_kf_tracked_over_curr",
         tracking::create_kf_tracked_over_curr);
  GPARAM("tracking/create_kf_tracked_over_ref",
         tracking::create_kf_tracked_over_ref);
  GPARAM("tracking/create_kf_ref_ratio", tracking::create_kf_ref_ratio);
  GPARAM("tracking/create_kf_nmatch", tracking::create_kf_nmatch);

  GPARAM("tracking/dust/c2_thresh", tracking::dust::c2_thresh);
  GPARAM("tracking/dust/th_ratio", tracking::dust::th_ratio);
  GPARAM("tracking/dust/th_ninlier", tracking::dust::th_ninlier);
  GPARAM("tracking/dust/th_nmatch", tracking::dust::th_nmatch);

  GPARAM("tracking/motion/th_window_size", tracking::motion::th_window_size);
  GPARAM("tracking/motion/th_nmatch_proj", tracking::motion::th_nmatch_proj);
  GPARAM("tracking/motion/th_nmatch_opt", tracking::motion::th_nmatch_opt);
  GPARAM("tracking/motion/th_nn_ratio", tracking::motion::th_nn_ratio);

  GPARAM("tracking/map/th_view_cos", tracking::map::th_view_cos);
  GPARAM("tracking/map/th_window_size", tracking::map::th_window_size);
  GPARAM("tracking/map/th_ninlier_high", tracking::map::th_ninlier_high);
  GPARAM("tracking/map/th_ninlier_low", tracking::map::th_ninlier_low);
  GPARAM("tracking/map/th_nn_ratio", tracking::map::th_nn_ratio);
  GPARAM("tracking/map/match_adaptive", tracking::map::match_adaptive);
  // GPARAM("tracking/th_depth", tracking::th_depth);

  GPARAM("mapping/culling_kf", mapping::culling_kf);
  GPARAM("mapping/kf_culling_cov_ratio", mapping::kf_culling_cov_ratio);
  GPARAM("mapping/kf_culling_num_obs", mapping::kf_culling_num_obs);
  GPARAM("mapping/triangulation_nn_ratio", mapping::triangulation_nn_ratio);
  GPARAM("mapping/triangulation_num_kfs", mapping::triangulation_num_kfs);
  GPARAM("mapping/matching_flann", mapping::matching_flann);
  GPARAM("mapping/matching_method", mapping::matching_method);

  GPARAM("viewer/keyframe_size", viewer::kf_size);
  GPARAM("viewer/keyframe_line_width", viewer::kf_line_width);
  GPARAM("viewer/graph_line_width", viewer::graph_line_width);
  GPARAM("viewer/point_size", viewer::point_size);
  GPARAM("viewer/camera_size", viewer::camera_size);
  GPARAM("viewer/camera_line_width", viewer::camera_line_width);

  std::vector<float> viewpoint;
  GPARAM("viewer/viewpoint", viewpoint);
  if (viewpoint.size() != 4)
    throw std::runtime_error("view point size mismatch");
  viewer::viewpoint_x = viewpoint[0];
  viewer::viewpoint_y = viewpoint[1];
  viewer::viewpoint_z = viewpoint[2];
  viewer::viewpoint_f = viewpoint[3];

#undef GPARAM
}

void printParameters() {}

} // namespace orbslam
