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

#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Dense>

#include "../type/map.h"
#include "../type/mappoint.h"

#include "../tracking/tracker.h"
#include "../cv/depth_filter.h"

namespace orbslam {

class Tracking;
class Viewer;

class FrameDrawer {
public:
  FrameDrawer();

  // Update info from the last processed frame.
  void Update(Tracking *pTracker);

  void updateCoarse(Tracking* pTracker);

  // Draw last processed frame.
  cv::Mat DrawFrame();

  void setMap(Map *pMap) { mpMap = pMap; }

  cv::Mat mVizCoarse;
  bool is_ok = false;



  cv::Mat img_sp, img_sp_viz;
public:
  void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

  Map *mpMap;

  // ADD(debug for depth filter)
  std::vector<std::shared_ptr<Seed>> seeds;

  // ADD(debug info for coarse tracking)

  // Info of the frame to be drawn
  cv::Mat mIm;
  int N;
  std::vector<cv::KeyPoint> mvCurrentKeys;
  std::vector<Eigen::Vector2f> mvCov2_inv;

  std::vector<bool> mvbMap, mvbVO;
  bool mbOnlyTracking;
  int mnTracked, mnTrackedVO;
  std::vector<cv::KeyPoint> mvIniKeys;
  std::vector<int> mvIniMatches;
  int mState;

  std::mutex mMutex;

  bool save_im_ = false;
};

} // namespace orbslam
