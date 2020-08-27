#pragma once

namespace orbslam {

// Input sensor
enum eSensor {
  MONOCULAR = 0,
  STEREO = 1,
  RGBD = 2,

};

enum Device {
  CPU = 0,
  CUDA = 1
};

} // namespace orbslam
