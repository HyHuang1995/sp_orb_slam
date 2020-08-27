#pragma once

#include <atomic>
#include <memory>

#include "common.h"

#include "viz/frame_drawer.h"
#include "viz/map_drawer.h"
#include "viz/viewer.h"

#include "type/map.h"

#include "loopclosing/loop_closer_vlad.h"
#include "mapping/local_mapper.h"
#include "tracking/tracker.h"

// #include "utils/summary_writer.h"

namespace orbslam {

// extern std::shared_ptr<SummaryWriter> logger;

namespace global {
struct Config {
  // Current tracking result from tracking
  cv::Mat current_pose;
};

extern MapDrawer *map_drawer;
extern FrameDrawer *frame_drawer;
extern Viewer *viewer;

extern Map *map;

extern LocalMapping *mapper;
extern LoopClosingVLAD *looper;
extern Tracking *tracker;

extern Config config;

extern std::atomic_bool b_local_on;
extern std::atomic_bool b_local_off;
extern std::atomic_bool b_system_reset;
extern std::atomic_bool b_pause;
extern std::atomic_bool b_step;

} // namespace global

} // namespace orbslam