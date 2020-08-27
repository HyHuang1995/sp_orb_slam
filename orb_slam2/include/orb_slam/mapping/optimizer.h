/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "g2o/types/sim3/types_seven_dof_expmap.h"

#include "../type/frame.h"
#include "../type/keyframe.h"
#include "../type/map.h"
#include "../type/mappoint.h"

#include "../loopclosing/loop_closer_vlad.h"

namespace orbslam {

// class Optimizer
// {
//   public:
//     void static BundleAdjustment(const std::vector<KeyFrame *> &vpKF,
//                                  const std::vector<MapPoint *> &vpMP,
//                                  int nIterations = 5, bool *pbStopFlag =
//                                  NULL, const unsigned long nLoopKF = 0, const
//                                  bool bRobust = true);
//     void static GlobalBundleAdjustemnt(Map *pMap, int nIterations = 5,
//                                        bool *pbStopFlag = NULL,
//                                        const unsigned long nLoopKF = 0,
//                                        const bool bRobust = true);
//     void static LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map
//     *pMap); int static PoseOptimization(Frame *pFrame);

//     // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise
//     // (mono)
//     void static OptimizeEssentialGraph(
//         Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
//         const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
//         const LoopClosing::KeyFrameAndPose &CorrectedSim3,
//         const map<KeyFrame *, set<KeyFrame *>> &LoopConnections,
//         const bool &bFixScale);

//     // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise
//     (mono) static int OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2,
//                             std::vector<MapPoint *> &vpMatches1,
//                             g2o::Sim3 &g2oS12, const float th2,
//                             const bool bFixScale);
// };

namespace Optimizer {

extern int inlier_count;

extern std::vector<Eigen::Vector2d> uv;

void BundleAdjustment(const std::vector<KeyFrame *> &vpKF,
                      const std::vector<MapPoint *> &vpMP, int nIterations = 5,
                      bool *pbStopFlag = NULL, const unsigned long nLoopKF = 0,
                      const bool bRobust = true);

void GlobalBundleAdjustemnt(Map *pMap, int nIterations = 5,
                            bool *pbStopFlag = NULL,
                            const unsigned long nLoopKF = 0,
                            const bool bRobust = true);

void LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap);

int PoseOptimization(Frame *pFrame);

int PoseOptimizationDustPost(Frame *pFrame);

int PoseOptimizationDust(Frame *frame, Frame *last_frame);

int PoseOptimizationDust(Frame *frame, KeyFrame *last_frame);

int PoseOptimizationHeat(Frame *frame, Frame *last_frame);

int PoseOptimizationDust(Frame *frame, const std::vector<MapPoint *> &mps);

int PoseOptimizationDust(Frame *frame, const std::vector<MapPoint *> &mps,
                         std::vector<bool> &is_visible);

// if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise
// (mono)
void OptimizeEssentialGraph(
    Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
    const LoopClosingVLAD::KeyFrameAndPose &NonCorrectedSim3,
    const LoopClosingVLAD::KeyFrameAndPose &CorrectedSim3,
    const std::map<KeyFrame *, std::set<KeyFrame *>> &LoopConnections,
    const bool &bFixScale);

// if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
int OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2,
                 std::vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12,
                 const float th2, const bool bFixScale);

} // namespace Optimizer

} // namespace orbslam
