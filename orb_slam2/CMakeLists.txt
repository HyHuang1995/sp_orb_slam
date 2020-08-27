cmake_minimum_required(VERSION 3.4)
project(orb_slam2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -pthread -O3 -march=native -Wno-deprecated-declarations -fext-numeric-literals")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV 3.0 REQUIRED)
find_package(Pangolin REQUIRED)
# find_package(fmt)

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/3rdparty/libtorch/")
find_package(Torch REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)

cs_add_library(${PROJECT_NAME} SHARED
include/orb_slam/init_cfg.hpp
src/utils/timing.cpp

src/config.cpp
src/global.cpp

src/system.cpp
src/system_helper.cpp

src/cv/depth_filter.cpp
src/cv/base_extractor.cpp
src/cv/sp_extractor.cpp
src/cv/sp_matcher.cpp
src/cv/sp_matcher_loop.cpp

src/optimization/types_dust_tracking.cpp

src/loopclosing/loop_closer_vlad.cpp

src/mapping/local_mapper.cpp
src/mapping/local_mapper_raw.cpp
src/mapping/optimizer.cpp
src/mapping/optimizer_dust.cpp
src/mapping/sim3_solver.cpp

src/tracking/tracker.cpp
src/tracking/tracker_dust.cpp
src/tracking/tracker_raw.cpp
src/tracking/initializer.cpp
src/tracking/mono_tracker.cpp

src/type/frame.cpp
src/type/keyframe.cpp
src/type/map.cpp
src/type/mappoint.cpp

src/viz/frame_drawer.cpp
src/viz/map_drawer.cpp
src/viz/viewer.cpp

src/utils/converter.cpp

src/io/data_loader.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${TORCH_LIBRARIES}
# fmt::fmt-header-only
${TORCH_LIBRARIES}
)

add_executable(mono_node node/mono_node.cc)
target_link_libraries(mono_node ${PROJECT_NAME})

cs_install()
cs_export()
