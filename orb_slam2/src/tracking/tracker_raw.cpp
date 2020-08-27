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

#include "orb_slam/tracking/tracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <cmath>
#include <iostream>
#include <mutex>

#include "orb_slam/common.h"
#include "orb_slam/config.h"
#include "orb_slam/global.h"

#include "orb_slam/type/map.h"
#include "orb_slam/type/type.h"

#include "orb_slam/cv/orb_matcher.h"
#include "orb_slam/cv/sp_extractor.h"
#include "orb_slam/cv/sp_matcher.h"
#include "orb_slam/viz/frame_drawer.h"

#include "orb_slam/utils/converter.h"
#include "orb_slam/utils/timing.h"

// #include "orb_slam/cv/pnp_solver.h"
#include "orb_slam/mapping/optimizer.h"

using namespace std;

namespace orbslam {

using namespace std;

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame *pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();

  mLastFrame.SetPose(Tlr * pRef->GetPose());
  // Tlr*Trw = Tlw 1:last r:reference w:world

  if (mnLastKeyFrameId == mLastFrame.mnId || common::sensor == MONOCULAR)
    return;

  vector<pair<float, int>> vDepthIdx;
  vDepthIdx.reserve(mLastFrame.N);

  for (int i = 0; i < mLastFrame.N; i++) {
    float z = mLastFrame.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(make_pair(z, i));
    }
  }

  if (vDepthIdx.empty())
    return;

  std::sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    MapPoint *pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP)
      bCreateNew = true;
    else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      cv::Mat x3D = mLastFrame.UnprojectStereo(i);
      MapPoint *pNewMP = new MapPoint(x3D, &mLastFrame, i);

      mLastFrame.mvpMapPoints[i] = pNewMP;

      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
      break;
  }
}

void Tracking::reset() {
  mState = NO_IMAGES_YET;

  if (mpInitializer) {
    delete mpInitializer;
    mpInitializer = static_cast<Initializer *>(NULL);
  }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();
}

void Tracking::InformOnlyTracking(const bool &flag) { mbOnlyTracking = flag; }

} // namespace orbslam
