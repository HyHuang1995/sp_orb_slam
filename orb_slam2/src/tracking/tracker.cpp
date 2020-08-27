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

namespace orbslam {

using namespace std;

cv::Mat Tracking::trackFrame(DataFrame::ConstPtr data_frame) {
  static timing::Timer timer_track("tracking/_total", true);
  static timing::Timer timer_frame("tracking/frame", true);
  static bool timing_track = false;
  if (mState == OK) {
    timing_track = true;
    timer_track.Start();
    timer_frame.Start();
  } else
    timing_track = false;

  setFrameData(data_frame->timestamp, data_frame->mono);
  mCurrentFrame.global_desc = data_frame->global_desc.clone();

  if (timing_track)
    timer_frame.Stop();

  if (common::verbose) {
    LOG(INFO) << "################## tracking #################";
    LOG(INFO) << " track frame idx: " << mCurrentFrame.mnId;
    LOG(INFO) << "number of features: " << mCurrentFrame.N;
  }

  track();
  if (timing_track) {
    timer_track.Stop();
  }

  LOG(INFO) << "done tracking. ";

  return mCurrentFrame.mTcw.clone();
}

Tracking::Tracking()
    : mState(NO_IMAGES_YET), mbOnlyTracking(false), mbVO(false),
      mpInitializer(nullptr), mnLastRelocFrameId(0) {
  float fx = camera::fx;
  float fy = camera::fy;
  float cx = camera::cx;
  float cy = camera::cy;

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  // [k1 k2 p1 p2 k3]
  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = camera::k1;
  DistCoef.at<float>(1) = camera::k2;
  DistCoef.at<float>(2) = camera::p1;
  DistCoef.at<float>(3) = camera::p2;
  const float k3 = camera::k3;
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  // mbf = camera::bf;

  float fps = camera::fps;
  if (fps == 0)
    fps = 30;

  mMinFrames = 0;
  mMaxFrames = fps / 2;

  int nRGB = camera::is_rgb;
  mbRGB = nRGB;

  if (tracking::extractor_type == tracking::ORB) {
    // mpORBextractorLeft = new ORBextractor(
    //     tracking::num_features, tracking::scale_factor, tracking::num_level,
    //     tracking::th_fast_ini, tracking::th_fast_min);
  } else if (tracking::extractor_type == tracking::SP) {
    cout << "extractor init to SuperPoint" << endl;
    mpORBextractorLeft = new SPExtractor(tracking::num_features);
  } else {
    throw runtime_error("extractor not implemented");
  }

  if (common::sensor == MONOCULAR) {
    if (tracking::extractor_type == tracking::ORB) {
      cout << "extractor init to ORB" << endl;
      // mpIniORBextractor = new ORBextractor(
      //     2 * tracking::num_features, tracking::scale_factor,
      //     tracking::num_level, tracking::th_fast_ini, tracking::th_fast_min);
    } else if (tracking::extractor_type == tracking::SP) {
      cout << "extractor init to SuperPoint" << endl;
      mpIniORBextractor = mpORBextractorLeft;
    } else {
      throw runtime_error("extractor not implemented");
    }
  }
}

cv::Mat Tracking::trackFrame(const double &timestamp, const cv::Mat &im1,
                             const cv::Mat &im2) {
  static timing::Timer timer_track("tracking/_total", true);
  static timing::Timer timer_frame("tracking/frame", true);
  static bool timing_track = false;
  if (mState == OK) {
    timing_track = true;
    timer_track.Start();
    timer_frame.Start();
  } else
    timing_track = false;

  setFrameData(timestamp, im1, im2);
  if (timing_track)
    timer_frame.Stop();

  if (common::verbose) {
    LOG(INFO) << "################## tracking #################";
    LOG(INFO) << " track frame idx: " << mCurrentFrame.mnId;
    LOG(INFO) << "number of features: " << mCurrentFrame.N;
  }
  // logger->addScalar("total_features", mCurrentFrame.mnId, mCurrentFrame.N);

  track();
  if (timing_track) {
    timer_track.Stop();
  }

  return mCurrentFrame.mTcw.clone();
}

void Tracking::track() {
  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  mLastProcessedState = mState;

  unique_lock<mutex> lock(global::map->mMutexMapUpdate);

  if (mState == NOT_INITIALIZED) {
    Initialization();

    global::frame_drawer->Update(this);

    if (mState != OK)
      return;

    mVelocity = cv::Mat::eye(4, 4, CV_32FC1);
  } else {
    bool bOK;

    {
      CheckReplacedInLastFrame();

      if (mState == OK) {
        timing::Timer timer_dust("tracking/dust");
        bOK = trackFrameDustKFLocal();
        timer_dust.Stop();
        if (!bOK) {
          n_fail_dust++;
        }

        if (!bOK) {
          bOK = TrackWithMotionModel();
        }

      } else {

        bOK = trackReferenceKeyFrameANN();
      }

      if (!bOK)
        bOK = trackReferenceKeyFrameANN();
    }

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    if (bOK) {
      timing::Timer timer_map("tracking/local_map");
      bOK = TrackLocalMap();
      timer_map.Stop();
    }

    if (bOK)
      mState = OK;
    else
      mState = LOST;

    // update the seeds in kf
    // timing::Timer timer_update("tracking/update_seeds");
    // mpLastKeyFrame->updateSeeds(&mCurrentFrame);
    // timer_update.Stop();
    std::vector<cv::KeyPoint> keys;
    mvpLocalMapPointsViz.clear();
    for (int i = 0; i < mCurrentFrame.N; i++) {
      auto &&mp = mCurrentFrame.mvpMapPoints[i];
      if ((!mp) || mCurrentFrame.mvbOutlier[i] || mp->isBad()) {
        continue;
      }
      const cv::Mat Pc =
          mCurrentFrame.mRcw * mp->GetWorldPos() + mCurrentFrame.mtcw;

      cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
      kp.angle = Pc.at<float>(2);
      keys.push_back(kp);

      mp->depth_frame = Pc.at<float>(2);
      mvpLocalMapPointsViz.push_back(mp);
    }
    sort(keys.begin(), keys.end(),
         [](const cv::KeyPoint &a, const cv::KeyPoint &b) -> bool {
           return a.angle > b.angle;
         });
    sort(mvpLocalMapPointsViz.begin(), mvpLocalMapPointsViz.end(),
         [](MapPoint *&a, MapPoint *&b) -> bool {
           return a->depth_frame > b->depth_frame;
         });

    mCurrentFrame.mvKeysViz = keys;

    // Update drawer
    if (common::visualize) {
      global::frame_drawer->Update(this);
    }

    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(
            LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc; // Tcl
      } else
        mVelocity = cv::Mat();

      global::map_drawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

      // Clean VO matches
      for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
          if (pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i] = false;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
          }
      }

      // Delete temporal MapPoints
      for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(),
                                      lend = mlpTemporalPoints.end();
           lit != lend; lit++) {
        MapPoint *pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrameOverride2()) {
        if (common::verbose) {
          LOG(INFO) << "insert new keyframe";
        }
        CreateNewKeyFrameOverride();
      }

      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
      }
    }

    if (!bOK) {
      cout << "lost at " << mCurrentFrame.mnId << endl;
    }

    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (global::map->KeyFramesInMap() <= 5) {
        cout << "Track lost soon after initialisation, reseting..." << endl;
        global::b_system_reset = true;
        return;
      }
    }

    if (!mCurrentFrame.mpReferenceKF)
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

    mLastFrame = Frame(mCurrentFrame);
  }

  if (!mCurrentFrame.mTcw.empty()) {
    cv::Mat Tcr =
        mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back(mState == LOST);
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back(mState == LOST);
  }
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint *pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint *pRep = pMP->GetReplaced();
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracking::trackReferenceKeyFrameANN() {
  if (common::verbose)
    LOG(INFO) << "Track with reference kf";

  SPMatcher matcher(0.9);
  vector<MapPoint *> mps;
  matcher.SearchByBruteForce(mpReferenceKF, mCurrentFrame, mps);
  mCurrentFrame.mvpMapPoints = mps;

  // cv::hconcat();
  int nmatches = 0;
  for (auto mp : mCurrentFrame.mvpMapPoints) {
    if (mp) {
      nmatches++;
    }
  }
  cout << "nmatches track kf: " << nmatches << endl;

  // mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);

  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  // global::logger->>add_scalar("tracking/keyframe/matches",
  // mCurrentFrame.mnId, nmatches);
  // global::logger->>add_scalar("tracking/keyframe/inliers",
  // mCurrentFrame.mnId, nmatchesMap);
  return nmatchesMap >= tracking::motion::th_nmatch_opt;
}

bool Tracking::TrackReferenceKeyFrame() {
  throw std::runtime_error("not implemented");
  // if (common::verbose)
  //   LOG(INFO) << "Track with reference kf";

  // // Compute Bag of Words vector
  // // mCurrentFrame.ComputeBoW();

  // // We perform first an ORB matching with the reference keyframe
  // // If enough matches are found we setup a PnP solver
  // SPMatcher matcher(0.7);
  // vector<MapPoint *> vpMapPointMatches;

  // int nmatches =
  //     matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  // cout << "nmatches: " << nmatches << endl;

  // if (nmatches < 15)
  //   return false;

  // mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  // mCurrentFrame.SetPose(mLastFrame.mTcw);

  // Optimizer::PoseOptimization(&mCurrentFrame);

  // // Discard outliers
  // int nmatchesMap = 0;
  // for (int i = 0; i < mCurrentFrame.N; i++) {
  //   if (mCurrentFrame.mvpMapPoints[i]) {
  //     if (mCurrentFrame.mvbOutlier[i]) {
  //       MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

  //       mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
  //       mCurrentFrame.mvbOutlier[i] = false;
  //       pMP->mbTrackInView = false;
  //       pMP->mnLastFrameSeen = mCurrentFrame.mnId;
  //       nmatches--;
  //     } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
  //       nmatchesMap++;
  //   }
  // }

  // // global::logger->>add_scalar("tracking/keyframe/matches",
  // // mCurrentFrame.mnId, nmatches);
  // // global::logger->>add_scalar("tracking/keyframe/inliers",
  // // mCurrentFrame.mnId, nmatchesMap);
  // return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrameOverride() {
  // Update pose according to reference keyframe
  KeyFrame *pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();

  // Tlr*Trw = Tlw 1:last r:reference w:world
  mLastFrame.SetPose(Tlr * pRef->GetPose());

  return;
}

bool Tracking::TrackWithMotionModel() {
  // ORBmatcher matcher(0.9, true);
  SPMatcher matcher(tracking::motion::th_nn_ratio);

  // Update last frame pose according to its reference keyframe
  UpdateLastFrameOverride();

  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint *>(NULL));

  // Project points seen in previous frame
  int th;
  if (common::sensor != STEREO)
    th = tracking::motion::th_window_size;
  else
    th = 7;

  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                            common::sensor == MONOCULAR);

  // If few matches, uses a wider window search
  if (nmatches < tracking::motion::th_nmatch_proj) {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint *>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th,
                                          common::sensor == MONOCULAR); // 2*th
  }

  // FIXME: change the projection params
  // if (nmatches < tracking::motion::th_nmatch_proj) {
  //   LOG(INFO) << "second round failed... nmatches: " << nmatches;
  //   return false;
  // }

  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0, nout = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        // nmatches--;
        nout++;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (common::verbose) {
    LOG(INFO) << "projection: #nextra: " << nmatches
              << " #inliers: " << nmatchesMap << " #outliers: " << nout;
  }

  // {
  //   inlier_coarse.push_back(nmatchesMap);
  //   inlier_coarse_ratio.push_back(nmatchesMap * 1.0f / mCurrentFrame.N);

  //   n_inlier_coarse += nmatchesMap;
  //   total_coarse.push_back(n_inlier_coarse);
  // }

  // logger->addScalar("proj_match", mCurrentFrame.mnId, nmatches);
  // logger->addScalar("proj_inlier", mCurrentFrame.mnId, nmatchesMap);
  // logger->addScalar("proj_outlier", mCurrentFrame.mnId, nout);
  // if (mbOnlyTracking) {
  //   mbVO = nmatchesMap < tracking::motion::th_nmatch_opt;
  //   return nmatches > tracking::motion::th_nmatch_proj;
  // }

  return nmatchesMap >= tracking::motion::th_nmatch_opt;
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the
  // frame. We retrieve the local map and try to find matches to points in the
  // local map.

  // Update Local KeyFrames and Local Points
  UpdateLocalMap();

  int nmatch = SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&mCurrentFrame);
  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      } else if (common::sensor == STEREO)
        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
    }
  }

  if (common::verbose) {
    LOG(INFO) << "#matches: " << nmatch
              << " #map points: " << mvpLocalMapPoints.size()
              << " #inliers: " << mnMatchesInliers;
  }

  inlier_fine.push_back(mnMatchesInliers);
  inlier_fine_ratio.push_back(mnMatchesInliers * 1.0f / mCurrentFrame.N);

  n_inlier_fine += mnMatchesInliers;
  total_fine.push_back(n_inlier_fine);

  // logger->addScalar("map_match", mCurrentFrame.mnId, nmatch);
  // logger->addScalar("map_inliers", mCurrentFrame.mnId, mnMatchesInliers);

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
      mnMatchesInliers < tracking::map::th_ninlier_high)
    return false;

  if (mnMatchesInliers < tracking::map::th_ninlier_low)
    return false;
  else
    return true;
}

bool Tracking::NeedNewKeyFrameOverride() {

  if (global::mapper->isStopped() || global::mapper->stopRequested())
    return false;

  const int nKFs = global::map->KeyFramesInMap();

  std::unordered_set<MapPoint *> curr_mps;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    auto &&mp = mCurrentFrame.mvpMapPoints[i];
    if (mp && !mp->isBad()) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        curr_mps.insert(mp);
      }
    }
  }
  int num_obs_in_common, total_obs;
  mpReferenceKF->getTrackedInCommon(curr_mps, num_obs_in_common, total_obs);
  float ratio_in_common = num_obs_in_common * 1.0f / total_obs;
  // cout << "ratio in common: " << ratio_in_common << ' ' << num_obs_in_common
  //      << ' ' << total_obs << endl;

  bool bLocalMappingIdle = global::mapper->AcceptKeyFrames();

  // fps
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + camera::fps;
  // cout << mCurrentFrame.mnId << ' ' << mnLastKeyFrameId << ' ' <<
  //     mMaxFrames << endl;

  float ratio_in_curr = mnMatchesInliers * 1.0f / mCurrentFrame.N;
  const bool c1b = ratio_in_common < tracking::create_kf_tracked_over_ref &&
                   ratio_in_curr < tracking::create_kf_tracked_over_curr;

  const bool c1c = false;

  const bool c2 = ratio_in_common < tracking::create_kf_ref_ratio;

  const bool c3 = mnMatchesInliers < tracking::create_kf_nmatch;

  auto tf = [](bool flag) { return flag ? "true" : "false"; };
  // cout << "flags: " << tf(c1a) << ' ' << tf(c1b) << ' ' << tf(c1c)
  //      << bLocalMappingIdle << ' ' << tf(c2) << ' ' << tf(c3) << endl;
  if (((c1a || c1b || c1c || bLocalMappingIdle) && c2) || c3) {
    if (bLocalMappingIdle) {
      return true;
    } else {
      global::mapper->InterruptBA();
      if (c3)
        return true;
      return false;
    }
  } else
    return false;
}

// bool Tracking::NeedNewKeyFrameOverride() {
//   if (mbOnlyTracking)
//     return false;

//   // If Local Mapping is freezed by a Loop Closure do not insert keyframes
//   if (global::mapper->isStopped() || global::mapper->stopRequested())
//     return false;

//   const int nKFs = global::map->KeyFramesInMap();

//   // Tracked MapPoints in the reference keyframe
//   int nMinObs = 3;
//   if (nKFs <= 2)
//     nMinObs = 2;
//   int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

//   // Local Mapping accept keyframes?
//   bool bLocalMappingIdle = global::mapper->AcceptKeyFrames();

//   // Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
//   // "total matches = matches to map + visual odometry matches"
//   // Visual odometry matches will become MapPoints if we insert a keyframe.
//   // This ratio measures how many MapPoints we could create if we insert a
//   // keyframe.
//   int nMap = 0;
//   int nTotal = 0;

//   // Condition 1a: More than "MaxFrames" have passed from last keyframe
//   const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames / 2;

//   const bool c1b =
//       mnMatchesInliers < nRefMatches * tracking::create_kf_tracked_over_ref;

//   // const bool c1c = mnMatchesInliers <
//   //                  mCurrentFrame.N * tracking::create_kf_tracked_over_curr;
//   const bool c1c = true;

//   // const bool c1d = mnMatchesInliers < 25;

//   const bool c2 =
//       mnMatchesInliers < nRefMatches * tracking::create_kf_ref_ratio;
//   // const bool c2 =
//   //     (nRefMatches - mnMatchesInliers > tracking::create_kf_ref_ratio);
//   // mnMatchesInliers < nRefMatches * tracking::create_kf_ref_ratio;

//   auto tf = [](bool flag) { return flag ? "true" : "false"; };
//   // cout << tf(c1a) << ' ' << tf(c1b) << ' ' << tf(c1c) << ' ' << tf(c2) <<
//   // endl;

//   // if (((c1a || c1b || c1c || bLocalMappingIdle) && c2) || c1d) {
//   if ((c1a || c1b || c1c || bLocalMappingIdle) && c2) {
//     // if ((((c1a || c1b || c1c || bLocalMappingIdle) && c2))) {
//     // If the mapping accepts keyframes, insert keyframe.
//     // Otherwise send a signal to interrupt BA
//     if (bLocalMappingIdle) {
//       return true;
//     } else {
//       global::mapper->InterruptBA();
//       // if (common::sensor != MONOCULAR) {
//       //   if (global::mapper->KeyframesInQueue() < 3)
//       //     return true;
//       //   else
//       //     return false;
//       // } else
//       // if (c1d)
//       //   return true;
//       return false;
//     }
//   } else
//     return false;
// }

void Tracking::CreateNewKeyFrameOverride() {
  if (!global::mapper->SetNotStop(true))
    return;

  KeyFrame *pKF = new KeyFrame(mCurrentFrame);
  pKF->mIm = mImGray.clone();

  // ADD(depth filter)
  // pKF->ComputeSceneMeanDepth(2);
  // pKF->initializeSeeds();
  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  global::mapper->InsertKeyFrame(pKF);

  global::mapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;

  std::unique_lock<std::mutex> lock(global::map->mutex_lastkf);
  global::map->pLastKF = pKF;
}

int Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  int count = 0; // DEBUG

  for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(),
                                    vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint *>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;

        count++;
      }
    }
  }

  int nToMatch = 0;

  // Project points in frame and check its visibility
  for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(),
                                    vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;

    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
      continue;
    if (pMP->isBad())
      continue;

    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, tracking::map::th_view_cos)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  int extra;
  if (nToMatch > 0) {
    // ORBmatcher matcher(0.8);
    SPMatcher matcher(tracking::map::th_nn_ratio);
    int th = tracking::map::th_window_size;
    if (common::sensor == RGBD)
      th = 3;

    // If the camera has been relocalised recently, perform a coarser search
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
      th = 5;
    extra = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);

    count += extra;
  }

  if (common::verbose) {
    LOG(INFO) << "local_search: "
              << "#count " << count << " #extra: " << extra << " #ToMatch "
              << nToMatch;
  }

  return count;
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  global::map->SetReferenceMapPoints(mvpLocalMapPointsViz);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                          itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    KeyFrame *pKF = *itKF;
    const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

    for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(),
                                            itEndMP = vpMPs.end();
         itMP != itEndMP; itMP++) {
      MapPoint *pMP = *itMP;
      if (!pMP)
        continue;
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
        continue;
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  map<KeyFrame *, int> keyframeCounter;
  // {
  //   for (int i = 0; i < mLastFrame.N; i++) {
  //     if (mLastFrame.mvpMapPoints[i]) {
  //       MapPoint *pMP = mLastFrame.mvpMapPoints[i];
  //       if (!pMP->isBad()) {
  //         const map<KeyFrame *, size_t> observations =
  //         pMP->GetObservations(); for (map<KeyFrame *,
  //         size_t>::const_iterator
  //                  it = observations.begin(),
  //                  itend = observations.end();
  //              it != itend; it++)
  //           keyframeCounter[it->first]++;
  //       } else {
  //         mLastFrame.mvpMapPoints[i] = NULL;
  //       }
  //     }
  //   }
  // } else {
  {
    for (int i = 0; i < mCurrentFrame.N; i++) {
      if (mCurrentFrame.mvpMapPoints[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP->isBad()) {
          const map<KeyFrame *, size_t> observations = pMP->GetObservations();
          for (map<KeyFrame *, size_t>::const_iterator
                   it = observations.begin(),
                   itend = observations.end();
               it != itend; it++)
            keyframeCounter[it->first]++;
        } else {
          mCurrentFrame.mvpMapPoints[i] = NULL;
        }
      }
    }
  }

  if (keyframeCounter.empty())
    return;

  int max = 0;
  KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(),
                                            itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame *pKF = it->first;

    if (pKF->isBad())
      continue;

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                          itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80)
      break;

    KeyFrame *pKF = *itKF;

    const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(20);
    // const auto &&vNeighs = pKF->GetVectorCovisibleKeyFrames();
    for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(),
                                            itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame *pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    const set<KeyFrame *> spChilds = pKF->GetChilds();
    for (set<KeyFrame *>::const_iterator sit = spChilds.begin(),
                                         send = spChilds.end();
         sit != send; sit++) {
      KeyFrame *pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    KeyFrame *pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracking::Relocalization() {
  // // Compute Bag of Words Vector
  // mCurrentFrame.ComputeBoW();

  // // Relocalization is performed when tracking is lost
  // // Track Lost: Query KeyFrame Database for keyframe candidates for
  // vector<KeyFrame *> vpCandidateKFs =
  //     global::keyframe_db->DetectRelocalizationCandidates(&mCurrentFrame);

  // if (vpCandidateKFs.empty())
  //   return false;

  // const int nKFs = vpCandidateKFs.size();

  // // We perform first an ORB matching with each candidate
  // // If enough matches are found we setup a PnP solver
  // // ORBmatcher matcher(0.75, true);
  // SPMatcher matcher(0.75);

  // vector<PnPsolver *> vpPnPsolvers;
  // vpPnPsolvers.resize(nKFs);

  // vector<vector<MapPoint *>> vvpMapPointMatches;
  // vvpMapPointMatches.resize(nKFs);

  // vector<bool> vbDiscarded;
  // vbDiscarded.resize(nKFs);

  // int nCandidates = 0;

  // for (int i = 0; i < nKFs; i++) {
  //   KeyFrame *pKF = vpCandidateKFs[i];
  //   if (pKF->isBad())
  //     vbDiscarded[i] = true;
  //   else {
  //     int nmatches =
  //         matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
  //     if (nmatches < 15) {
  //       vbDiscarded[i] = true;
  //       continue;
  //     } else {
  //       PnPsolver *pSolver =
  //           new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
  //       pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
  //       vpPnPsolvers[i] = pSolver;
  //       nCandidates++;
  //     }
  //   }
  // }

  // // Alternatively perform some iterations of P4P RANSAC
  // // Until we found a camera pose supported by enough inliers
  // bool bMatch = false;
  // // ORBmatcher matcher2(0.9, true);
  // SPMatcher matcher2(0.9);

  // while (nCandidates > 0 && !bMatch) {
  //   for (int i = 0; i < nKFs; i++) {
  //     if (vbDiscarded[i])
  //       continue;

  //     // Perform 5 Ransac Iterations
  //     vector<bool> vbInliers;
  //     int nInliers;
  //     bool bNoMore;

  //     PnPsolver *pSolver = vpPnPsolvers[i];
  //     cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

  //     // If Ransac reachs max. iterations discard keyframe
  //     if (bNoMore) {
  //       vbDiscarded[i] = true;
  //       nCandidates--;
  //     }

  //     // If a Camera Pose is computed, optimize
  //     if (!Tcw.empty()) {
  //       Tcw.copyTo(mCurrentFrame.mTcw);

  //       set<MapPoint *> sFound;

  //       const int np = vbInliers.size();

  //       for (int j = 0; j < np; j++) {
  //         if (vbInliers[j]) {
  //           mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
  //           sFound.insert(vvpMapPointMatches[i][j]);
  //         } else
  //           mCurrentFrame.mvpMapPoints[j] = NULL;
  //       }

  //       int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

  //       if (nGood < 10)
  //         continue;

  //       for (int io = 0; io < mCurrentFrame.N; io++)
  //         if (mCurrentFrame.mvbOutlier[io])
  //           mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

  //       // If few inliers, search by projection in a coarse window and
  //       optimize
  //       // again
  //       //
  //       if (nGood < 50) {
  //         int nadditional = matcher2.SearchByProjection(
  //             mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

  //         if (nadditional + nGood >= 50) {
  //           nGood = Optimizer::PoseOptimization(&mCurrentFrame);

  //           // If many inliers but still not enough, search by projection
  //           again
  //           // in a narrower window the camera has been already optimized
  //           with
  //           // many points
  //           if (nGood > 30 && nGood < 50) {
  //             sFound.clear();
  //             for (int ip = 0; ip < mCurrentFrame.N; ip++)
  //               if (mCurrentFrame.mvpMapPoints[ip])
  //                 sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
  //             nadditional = matcher2.SearchByProjection(
  //                 mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

  //             // Final optimization
  //             if (nGood + nadditional >= 50) {
  //               nGood = Optimizer::PoseOptimization(&mCurrentFrame);

  //               for (int io = 0; io < mCurrentFrame.N; io++)
  //                 if (mCurrentFrame.mvbOutlier[io])
  //                   mCurrentFrame.mvpMapPoints[io] = NULL;
  //             }
  //           }
  //         }
  //       }

  //       // If the pose is supported by enough inliers stop ransacs and
  //       continue if (nGood >= 50) {
  //         bMatch = true;
  //         break;
  //       }
  //     }
  //   }
  // }

  // if (!bMatch) {
  //   return false;
  // } else {
  //   mnLastRelocFrameId = mCurrentFrame.mnId;
  //   return true;
  // }
  return false;
}

template <typename T> void saveVector(string file_name, const vector<T> &vec) {
  ofstream fs(file_name.c_str());
  for (auto &&e : vec) {
    fs << e << endl;
  }

  fs.close();
}

void Tracking::report() {
  LOG(WARNING) << "############## tracking summary #################" << endl;

  int sum_coarse = 0;
  for (size_t i = 0; i < inlier_coarse.size(); i++) {
    sum_coarse += inlier_coarse[i];
  }
  LOG(WARNING) << "#averge inlier coarse: "
               << sum_coarse * 1.0 / inlier_coarse.size() << endl;

  float sum_coarse_ratio = 0;
  for (size_t i = 0; i < inlier_coarse_ratio.size(); i++) {
    sum_coarse_ratio += inlier_coarse_ratio[i];
  }
  LOG(WARNING) << "#averge inlier ratio coarse: "
               << sum_coarse_ratio / inlier_coarse_ratio.size() << endl;

  int sum_fine = 0;
  for (size_t i = 0; i < inlier_fine.size(); i++) {
    sum_fine += inlier_fine[i];
  }
  float avg = sum_fine * 1.0 / inlier_fine.size();
  LOG(WARNING) << "#averge inlier fine: " << avg << endl;
  float std = 0.0;
  for (size_t i = 0; i < inlier_fine.size(); i++) {
    std += (inlier_fine[i] - avg) * (inlier_fine[i] - avg);
  }
  std /= inlier_fine.size();
  std = sqrt(std);
  LOG(WARNING) << "mean: " << avg << " variance: " << std << endl;

  float sum_fine_ratio = 0;
  for (size_t i = 0; i < inlier_fine_ratio.size(); i++) {
    sum_fine_ratio += inlier_fine_ratio[i];
  }
  LOG(WARNING) << "#averge inlier ratio fine: "
               << sum_fine_ratio / inlier_fine_ratio.size() << endl;

  LOG(WARNING) << "#dust tracking failure: " << n_fail_dust;
}
} // namespace orbslam
