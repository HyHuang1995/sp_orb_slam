#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "type/type.h"

namespace orbslam {

namespace common {

extern std::string seq;

extern std::string dataset;

extern std::string output_path;

extern std::string data_path;

extern std::string model_path;

extern eSensor sensor;

extern bool use_loop;

extern bool online;

extern bool verbose;

extern bool visualize;
} // namespace common

namespace camera {
extern float fx, fy, cx, cy;

extern float k1, k2, p1, p2, k3;

extern int width, height;

extern float fps;

extern bool is_rgb;

extern int d_type;
} // namespace camera

namespace matching {
extern int ntree;

extern int nchecks;
} // namespace matching

namespace tracking {
// extern
enum Extractor { ORB, SP };

extern Extractor extractor_type;

extern int num_features;

extern bool scale_check;

extern float create_kf_tracked_over_ref;

extern float create_kf_tracked_over_curr;

extern float create_kf_ref_ratio;

extern float create_kf_nmatch;

namespace dust {
extern float th_ratio;

extern int th_ninlier;

extern int th_nmatch;

extern float c2_thresh;
} // namespace dust

namespace motion {
extern int th_window_size;

extern int th_nmatch_proj;

extern int th_nmatch_opt;

extern float th_nn_ratio;
} // namespace motion

namespace map {
extern float th_view_cos;

extern int th_window_size;

extern int th_ninlier_high;

extern int th_ninlier_low;

extern float th_nn_ratio;

extern bool match_adaptive;
} // namespace map

} // namespace tracking

namespace mapping {
extern bool culling_kf;

extern float kf_culling_cov_ratio;

extern int kf_culling_num_obs;

extern float triangulation_nn_ratio;

extern int triangulation_num_kfs;

extern bool matching_flann;

// 0: bow   1: flann   2: epi
extern int matching_method;

} // namespace mapping

namespace viewer {
extern float kf_size;
extern float kf_line_width;

extern float graph_line_width;

extern float point_size;

extern float camera_size;
extern float camera_line_width;

extern float viewpoint_x;
extern float viewpoint_y;
extern float viewpoint_z;
extern float viewpoint_f;
} // namespace viewer

} // namespace orbslam
