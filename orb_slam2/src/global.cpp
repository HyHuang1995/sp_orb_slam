#include "orb_slam/global.h"

namespace orbslam {

namespace global {
FrameDrawer *frame_drawer;
MapDrawer *map_drawer;
Viewer *viewer = nullptr;

Map *map = nullptr;

Tracking *tracker = nullptr;
LocalMapping *mapper = nullptr;
LoopClosingVLAD *looper = nullptr;

std::atomic_bool b_local_on(false);
std::atomic_bool b_local_off(false);
std::atomic_bool b_system_reset(false);
std::atomic_bool b_pause(false);
std::atomic_bool b_step(false);

} // namespace global

} // namespace orbslam