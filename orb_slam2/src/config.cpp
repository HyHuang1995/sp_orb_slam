#include "orb_slam/config.h"

#include <limits>

namespace orbslam {

namespace common {

std::string seq;

std::string data_path;

std::string model_path;

std::string dataset;

std::string output_path;

eSensor sensor;

bool use_loop;

bool verbose = false;

bool visualize = true;

bool online = false;

} // namespace common

namespace camera {

float fx, fy, cx, cy;

float k1 = 0.0f, k2 = 0.0f, p1 = 0.0f, p2 = 0.0f, k3 = 0.0f;

int width = 0, height = 0;

float fps;

bool is_rgb;

} // namespace camera

namespace matching {
int ntree = 4;

int nchecks = 32;
} // namespace matching

namespace tracking {
Extractor extractor_type;

int num_features;

bool scale_check = true;

float create_kf_tracked_over_ref = 0.5f;

float create_kf_tracked_over_curr = 0.5f;

float create_kf_ref_ratio = 0.90f;

float create_kf_nmatch = 25;

namespace dust {

float th_ratio = 0.5f;

int th_ninlier;

int th_nmatch;

float c2_thresh = 81.0f;
} // namespace dust

namespace motion {
int th_window_size = 15;

int th_nmatch_proj = 20;

int th_nmatch_opt = 10;

float th_nn_ratio = 0.9;
} // namespace motion

namespace map {
float th_view_cos = 0.5f;

int th_window_size = 1;

int th_ninlier_high = 50;

int th_ninlier_low = 30;

float th_nn_ratio = 0.8;

bool match_adaptive = false;
} // namespace map

} // namespace tracking

namespace mapping {
bool culling_kf = true;

float kf_culling_cov_ratio = 0.9f;

int kf_culling_num_obs = 3;

float triangulation_nn_ratio = 0.6f;

int triangulation_num_kfs = 20;

bool matching_flann = false;

int matching_method = 1;
} // namespace mapping

namespace viewer {

float kf_size;
float kf_line_width;

float graph_line_width;

float point_size;

float camera_size;
float camera_line_width;

float viewpoint_x;
float viewpoint_y;
float viewpoint_z;
float viewpoint_f;

} // namespace viewer

} // namespace orbslam
