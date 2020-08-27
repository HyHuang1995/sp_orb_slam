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

#include "orb_slam/mapping/local_mapper.h"

#include <chrono>
#include <mutex>

#include "orb_slam/common.h"
#include "orb_slam/config.h"
#include "orb_slam/global.h"

#include "orb_slam/cv/orb_matcher.h"
#include "orb_slam/cv/sp_matcher.h"

#include "orb_slam/mapping/optimizer.h"

#include "orb_slam/utils/timing.h"

namespace orbslam {

using namespace std;

LocalMapping::LocalMapping(const float bMonocular)
    : mbMonocular(bMonocular), mbResetRequested(false),
      mbFinishRequested(false), mbFinished(true), mbAbortBA(false),
      mbStopped(false), mbStopRequested(false), mbNotStop(false),
      mbAcceptKeyFrames(true) {}

void LocalMapping::spinOnce() {
  LOG(WARNING) << "spin once";

  while (CheckNewKeyFrames()) {

    timing::Timer timer_mapping("mapping");
    // BoW conversion and insertion in Map
    // VI-A keyframe insertion
    timing::Timer timer_kf("mapping/insert_kf");
    ProcessNewKeyFrame();
    timer_kf.Stop();
    LOG(INFO) << "current kf frame idx: " << mpCurrentKeyFrame->mnFrameId
              << " kfid: " << mpCurrentKeyFrame->mnId;

    // Check recent MapPoints
    // VI-B recent map points culling
    MapPointCulling();

    // Triangulate new MapPoints
    // VI-C new map points creation
    timing::Timer timer_np("mapping/create_mps");
    CreateNewMapPointsOverride();
    timer_np.Stop();

    if (!CheckNewKeyFrames()) {
      // Find more matches in neighbor keyframes and fuse point duplications
      timing::Timer timer_fuse("mapping/fuse_mps");
      SearchInNeighbors();
      timer_fuse.Stop();
    }

    mbAbortBA = false;

    if (!CheckNewKeyFrames() && !stopRequested()) {
      // VI-D Local BA
      if (global::map->KeyFramesInMap() > 2) {
        timing::Timer timer_ba("mapping/local_ba");
        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA,
                                         global::map);
        timer_ba.Stop();
      }

      // Check redundant local Keyframes
      // VI-E local keyframes culling
      if (mapping::culling_kf) {
        timing::Timer timer_ckf("mapping/kf_culling");
        KeyFrameCullingOverride();
        timer_ckf.Stop();
      }
    }

    timer_mapping.Stop();
    if (common::verbose) {
      timing::Timing::Print(std::cout);
    }

    if (common::use_loop) {
      global::looper->InsertKeyFrame(mpCurrentKeyFrame);
    }
  }

  if (Stop()) {
    // Safe area to stop
    while (isStopped() && !CheckFinish()) {
      // usleep(3000);
      std::this_thread::yield();
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    // if (CheckFinish())
    //   break;
  }

  // LOG(INFO) << "LocalMapping done. ";

  ResetIfRequested();

  // Tracking will see that Local Mapping is not busy
  SetAcceptKeyFrames(true);

  // SetFinish();
}

void LocalMapping::Run() {

  vector<float> vTimesTrack;
  mbFinished = false;

  while (1) {
    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {

      timing::Timer timer_mapping("mapping");
      // BoW conversion and insertion in Map
      timing::Timer timer_kf("mapping/insert_kf");
      ProcessNewKeyFrame();
      timer_kf.Stop();
      LOG(INFO) << "current kf frame idx: " << mpCurrentKeyFrame->mnFrameId
                << " kfid: " << mpCurrentKeyFrame->mnId;

      // Check recent MapPoints
      // VI-B recent map points culling
      timing::Timer timer_mp("mapping/culling_mps");
      MapPointCulling();
      timer_mp.Stop();

      // Triangulate new MapPoints
      // VI-C new map points creation
      timing::Timer timer_np("mapping/create_mps");
      CreateNewMapPointsOverride();
      timer_np.Stop();

      if (!CheckNewKeyFrames()) {
        // Find more matches in neighbor keyframes and fuse point duplications
        timing::Timer timer_fuse("mapping/fuse_mps");
        SearchInNeighbors();
        timer_fuse.Stop();
      }

      mbAbortBA = false;

      if (!CheckNewKeyFrames() && !stopRequested()) {
        // VI-D Local BA
        if (global::map->KeyFramesInMap() > 2) {
          timing::Timer timer_ba("mapping/local_ba");
          Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA,
                                           global::map);
          timer_ba.Stop();
        }

        // Check redundant local Keyframes
        // VI-E local keyframes culling
        if (mapping::culling_kf) {
          timing::Timer timer_ckf("mapping/kf_culling");
          KeyFrameCullingOverride();
          timer_ckf.Stop();
        }
      }

      timer_mapping.Stop();
      if (common::verbose) {
        timing::Timing::Print(std::cout);
      }

      if (common::use_loop) {
        global::looper->InsertKeyFrame(mpCurrentKeyFrame);
      }
    } else if (Stop()) {
      while (isStopped() && !CheckFinish()) {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
      if (CheckFinish())
        break;
    }

    // LOG(INFO) << "LocalMapping done. ";

    ResetIfRequested();

    // Tracking will see that Local Mapping is not busy
    SetAcceptKeyFrames(true);

    if (CheckFinish())
      break;

    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }

  // sort(vTimesTrack.begin(), vTimesTrack.end());
  // float totaltime = 0;
  // for (int ni = 0; ni < vTimesTrack.size(); ni++) {
  //   totaltime += vTimesTrack[ni];
  // }
  // cout << "-------" << endl << endl;
  // cout << "median mapping time: " << vTimesTrack[vTimesTrack.size() / 2]
  //      << endl;
  // cout << "mean mapping time: " << totaltime / vTimesTrack.size() << endl;
  SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexNewKFs);
  mlNewKeyFrames.push_back(pKF);
  mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mMutexNewKFs);
  return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame() {
  LOG(INFO) << "processing new keyframe";
  {
    unique_lock<mutex> lock(mMutexNewKFs);
    mpCurrentKeyFrame = mlNewKeyFrames.front();
    mlNewKeyFrames.pop_front();
  }

  timing::Timer timer_indexing("mapping/index");
  mpCurrentKeyFrame->buildIndexes();
  timer_indexing.Stop();

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();

  for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
          pMP->AddObservation(mpCurrentKeyFrame, i);
          pMP->UpdateNormalAndDepth();
          pMP->ComputeDistinctiveDescriptors();
          pMP->updateDescTrack(mpCurrentKeyFrame, i);
        } else {
          mlpRecentAddedMapPoints.push_back(pMP);
        }
      }
    }
  }

  // Update links in the Covisibility Graph
  mpCurrentKeyFrame->UpdateConnections();

  // Insert Keyframe in Map
  global::map->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling() {
  LOG(INFO) << "culling map point";

  list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
  const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

  int nThObs;
  if (mbMonocular)
    nThObs = 2;
  else
    nThObs = 3;
  const int cnThObs = nThObs;

  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint *pMP = *lit;
    if (pMP->isBad()) {
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (pMP->GetFoundRatio() < 0.25f) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 &&
               pMP->Observations() <= cnThObs) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
      lit = mlpRecentAddedMapPoints.erase(lit);
    else
      lit++;
  }
}

void LocalMapping::CreateNewMapPointsEpipolar() {
  if (common::verbose) {
    LOG(INFO) << "creating new map points";
  }

  // Retrieve neighbor keyframes in covisibility graph
  int nn = mapping::triangulation_num_kfs;

  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

  // ORBmatcher matcher(0.6, false);
  SPMatcher matcher(mapping::triangulation_nn_ratio);

  cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
  cv::Mat Tcw1(3, 4, CV_32F);
  Rcw1.copyTo(Tcw1.colRange(0, 3));
  tcw1.copyTo(Tcw1.col(3));

  cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

  const float &fx1 = mpCurrentKeyFrame->fx;
  const float &fy1 = mpCurrentKeyFrame->fy;
  const float &cx1 = mpCurrentKeyFrame->cx;
  const float &cy1 = mpCurrentKeyFrame->cy;
  const float &invfx1 = mpCurrentKeyFrame->invfx;
  const float &invfy1 = mpCurrentKeyFrame->invfy;

  const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

  int nnew = 0;
  int n_for_tri = 0;
  // Search matches with epipolar restriction and triangulate
  for (size_t i = 0; i < vpNeighKFs.size(); i++) {
    if (i > 0 && CheckNewKeyFrames())
      return;

    KeyFrame *pKF2 = vpNeighKFs[i];

    // Check first that baseline is not too short
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2 - Ow1;
    const float baseline = cv::norm(vBaseline);

    if (!mbMonocular) {
      if (baseline < pKF2->mb)
        continue;
    } else {
      const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
      const float ratioBaselineDepth = baseline / medianDepthKF2;
      // FIXME: new 3D?
      if (ratioBaselineDepth < 0.01)
        continue;
    }

    // Compute Fundamental Matrix
    cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

    // TODO: validate mappoint generated from depth info

    // Search matches that fullfil epipolar constraint
    timing::Timer timer_tri("mapping/search_triangle");
    vector<pair<size_t, size_t>> vMatchedIndices;
    int num_remain = pKF2->mIndicesRemain.size();
    if (mapping::matching_flann) {
      matcher.SearchForTriByFlann(mpCurrentKeyFrame, pKF2, F12,
                                  vMatchedIndices);
    } else {
      matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12,
                                     vMatchedIndices, false);
    }

    timer_tri.Stop();

    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = pKF2->GetTranslation();
    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));

    const float &fx2 = pKF2->fx;
    const float &fy2 = pKF2->fy;
    const float &cx2 = pKF2->cx;
    const float &cy2 = pKF2->cy;
    const float &invfx2 = pKF2->invfx;
    const float &invfy2 = pKF2->invfy;

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    n_for_tri += nmatches;
    int num_success = 0;
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int &idx1 = vMatchedIndices[ikp].first;

      const int &idx2 = vMatchedIndices[ikp].second;

      const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
      // const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
      // bool bStereo1 = kp1_ur >= 0;

      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

      cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1,
                     (kp1.pt.y - cy1) * invfy1, 1.0);
      cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2,
                     (kp2.pt.y - cy2) * invfy2, 1.0);

      cv::Mat ray1 = Rwc1 * xn1;
      cv::Mat ray2 = Rwc2 * xn2;
      const float cosParallaxRays =
          ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

      float cosParallaxStereo = cosParallaxRays + 1;
      float cosParallaxStereo1 = cosParallaxStereo;
      float cosParallaxStereo2 = cosParallaxStereo;

      cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

      cv::Mat x3D;
      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          cosParallaxRays < 0.9998) {
        // Linear Triangulation Method
        cv::Mat A(4, 4, CV_32F);
        A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
        A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
        A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
        A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        x3D = vt.row(3).t();

        if (x3D.at<float>(3) == 0)
          continue;

        // Euclidean coordinates
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
      } else
        continue; // No stereo and very low parallax

      cv::Mat x3Dt = x3D.t();

      // Check triangulation in front of cameras
      float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
      if (z1 <= 0)
        continue;

      float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
      if (z2 <= 0)
        continue;

      // Check reprojection error in first keyframe
      {
        const auto sigma = mpCurrentKeyFrame->cov2_inv_[idx1];
        const float sigma2_inv_x = sigma.x();
        const float sigma2_inv_y = sigma.y();

        const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
        const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
        const float invz1 = 1.0 / z1;

        float u1 = fx1 * x1 * invz1 + cx1;
        float v1 = fy1 * y1 * invz1 + cy1;
        float errX1 = u1 - kp1.pt.x;
        float errY1 = v1 - kp1.pt.y;
        if ((errX1 * errX1 * sigma2_inv_x + errY1 * errY1 * sigma2_inv_y) >
            5.991)
          continue;
      }

      // Check reprojection error in second keyframe
      {
        const auto sigma = pKF2->cov2_inv_[idx2];
        const float sigma2_inv_x = sigma.x();
        const float sigma2_inv_y = sigma.y();

        const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
        const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
        const float invz2 = 1.0 / z2;
        float u2 = fx2 * x2 * invz2 + cx2;
        float v2 = fy2 * y2 * invz2 + cy2;
        float errX2 = u2 - kp2.pt.x;
        float errY2 = v2 - kp2.pt.y;
        if ((errX2 * errX2 * sigma2_inv_x + errY2 * errY2 * sigma2_inv_y) >
            5.991)
          continue;
      }

      // Check scale consistency

      cv::Mat normal1 = x3D - Ow1;
      float dist1 = cv::norm(normal1);

      cv::Mat normal2 = x3D - Ow2;
      float dist2 = cv::norm(normal2);

      if (dist1 == 0 || dist2 == 0)
        continue;

      // Triangulation is succesfull
      MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame);

      pMP->AddObservation(mpCurrentKeyFrame, idx1);
      pMP->AddObservation(pKF2, idx2);

      mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
      pKF2->AddMapPoint(pMP, idx2);

      pMP->ComputeDistinctiveDescriptors();

      pMP->updateDescTrack(mpCurrentKeyFrame, idx1);

      pMP->UpdateNormalAndDepth();

      global::map->AddMapPoint(pMP);

      mlpRecentAddedMapPoints.push_back(pMP);
      num_success++;

      nnew++;
    }

    if (mapping::matching_flann) {
      if (num_success > num_remain / 2 &&
          num_remain > mpCurrentKeyFrame->N / 4) {
        pKF2->buildIndexes();
      }
    }
  }
  if (mapping::matching_flann) {
    int num_remain_curr = mpCurrentKeyFrame->mIndicesRemain.size();
    if (nnew > num_remain_curr / 2 &&
        num_remain_curr > mpCurrentKeyFrame->N / 4) {
      mpCurrentKeyFrame->buildIndexes();
    }
  }

  if (common::verbose) {
    LOG(INFO) << "#matches: " << n_for_tri << " #nnew: " << nnew;
  }
}

void LocalMapping::CreateNewMapPointsOverride() {
  if (common::verbose) {
    LOG(INFO) << "creating new map points";
  }

  // Retrieve neighbor keyframes in covisibility graph
  int nn = mapping::triangulation_num_kfs;

  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

  SPMatcher matcher(mapping::triangulation_nn_ratio);

  cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
  cv::Mat Tcw1(3, 4, CV_32F);
  Rcw1.copyTo(Tcw1.colRange(0, 3));
  tcw1.copyTo(Tcw1.col(3));

  cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

  const float &fx1 = mpCurrentKeyFrame->fx;
  const float &fy1 = mpCurrentKeyFrame->fy;
  const float &cx1 = mpCurrentKeyFrame->cx;
  const float &cy1 = mpCurrentKeyFrame->cy;
  const float &invfx1 = mpCurrentKeyFrame->invfx;
  const float &invfy1 = mpCurrentKeyFrame->invfy;

  const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

  int nnew = 0;
  int n_for_tri = 0, n_rej_para = 0, n_rej_depth = 0, n_rej_repro = 0;

  for (size_t i = 0; i < vpNeighKFs.size(); i++) {
    if (i > 0 && CheckNewKeyFrames())
      return;

    KeyFrame *pKF2 = vpNeighKFs[i];

    // Check first that baseline is not too short
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2 - Ow1;
    const float baseline = cv::norm(vBaseline);

    if (!mbMonocular) {
      if (baseline < pKF2->mb)
        continue;
    } else {
      const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
      const float ratioBaselineDepth = baseline / medianDepthKF2;
      // FIXME: new 3D?
      if (ratioBaselineDepth < 0.01)
        continue;
    }

    // Compute Fundamental Matrix
    cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

    // TODO: validate mappoint generated from depth info

    // Search matches that fullfil epipolar constraint
    timing::Timer timer_tri("mapping/search_triangle");
    vector<pair<size_t, size_t>> vMatchedIndices;
    int num_remain = pKF2->mIndicesRemain.size();
    if (mapping::matching_method == 1) {
      matcher.SearchForTriByFlann(mpCurrentKeyFrame, pKF2, F12,
                                  vMatchedIndices);
    } else if (mapping::matching_method == 2) {
      matcher.SearchForTriByEpi(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices);
    } else {
      matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12,
                                     vMatchedIndices, false);
    }

    timer_tri.Stop();

    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = pKF2->GetTranslation();
    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));

    const float &fx2 = pKF2->fx;
    const float &fy2 = pKF2->fy;
    const float &cx2 = pKF2->cx;
    const float &cy2 = pKF2->cy;
    const float &invfx2 = pKF2->invfx;
    const float &invfy2 = pKF2->invfy;

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    n_for_tri += nmatches;
    int num_success = 0;
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int &idx1 = vMatchedIndices[ikp].first;

      const int &idx2 = vMatchedIndices[ikp].second;

      const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

      // Check parallax between rays
      cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1,
                     (kp1.pt.y - cy1) * invfy1, 1.0);
      cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2,
                     (kp2.pt.y - cy2) * invfy2, 1.0);

      cv::Mat ray1 = Rwc1 * xn1;
      cv::Mat ray2 = Rwc2 * xn2;
      const float cosParallaxRays =
          ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

      float cosParallaxStereo = cosParallaxRays + 1;
      float cosParallaxStereo1 = cosParallaxStereo;
      float cosParallaxStereo2 = cosParallaxStereo;

      cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

      cv::Mat x3D;

      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          cosParallaxRays < 0.9998) {
        // Linear Triangulation Method
        // 见Initializer.cpp的Triangulate函数
        cv::Mat A(4, 4, CV_32F);
        A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
        A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
        A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
        A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        x3D = vt.row(3).t();

        if (x3D.at<float>(3) == 0)
          continue;

        // Euclidean coordinates
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
      } else {
        n_rej_para++;
        continue; // No stereo and very low parallax
      }

      cv::Mat x3Dt = x3D.t();

      // Check triangulation in front of cameras
      float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
      if (z1 <= 0) {
        n_rej_depth++;
        continue;
      }

      float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
      if (z2 <= 0) {
        n_rej_depth++;
        continue;
      }

      // Check reprojection error in first keyframe
      {
        const auto sigma = mpCurrentKeyFrame->cov2_inv_[idx1];
        const float sigma2_inv_x = sigma.x();
        const float sigma2_inv_y = sigma.y();

        const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
        const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
        const float invz1 = 1.0 / z1;

        float u1 = fx1 * x1 * invz1 + cx1;
        float v1 = fy1 * y1 * invz1 + cy1;
        float errX1 = u1 - kp1.pt.x;
        float errY1 = v1 - kp1.pt.y;
        if ((errX1 * errX1 * sigma2_inv_x + errY1 * errY1 * sigma2_inv_y) >
            5.991) {
          n_rej_repro++;
          continue;
        }
      }

      {
        const auto sigma = pKF2->cov2_inv_[idx2];
        const float sigma2_inv_x = sigma.x();
        const float sigma2_inv_y = sigma.y();

        const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
        const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
        const float invz2 = 1.0 / z2;
        float u2 = fx2 * x2 * invz2 + cx2;
        float v2 = fy2 * y2 * invz2 + cy2;
        float errX2 = u2 - kp2.pt.x;
        float errY2 = v2 - kp2.pt.y;
        if ((errX2 * errX2 * sigma2_inv_x + errY2 * errY2 * sigma2_inv_y) >
            5.991) {
          n_rej_repro++;
          continue;
        }
      }

      // Check scale consistency
      cv::Mat normal1 = x3D - Ow1;
      float dist1 = cv::norm(normal1);

      cv::Mat normal2 = x3D - Ow2;
      float dist2 = cv::norm(normal2);

      if (dist1 == 0 || dist2 == 0)
        continue;

      // Triangulation is succesfull
      MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame);

      pMP->AddObservation(mpCurrentKeyFrame, idx1);
      pMP->AddObservation(pKF2, idx2);

      mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
      pKF2->AddMapPoint(pMP, idx2);

      pMP->ComputeDistinctiveDescriptors();

      pMP->updateDescTrack(mpCurrentKeyFrame, idx1);

      pMP->UpdateNormalAndDepth();

      global::map->AddMapPoint(pMP);

      mlpRecentAddedMapPoints.push_back(pMP);
      num_success++;

      nnew++;
    }

    if (mapping::matching_method == 1) {
      // if (num_success > num_remain / 2 &&
      //     num_remain > mpCurrentKeyFrame->N / 4) {
      pKF2->buildIndexes();
      // }
      mpCurrentKeyFrame->buildIndexes();
    }
  }
  // if (mapping::matching_flann) {
  //   int num_remain_curr = mpCurrentKeyFrame->mIndicesRemain.size();
  //   if (nnew > num_remain_curr / 2 &&
  //       num_remain_curr > mpCurrentKeyFrame->N / 4) {
  //     mpCurrentKeyFrame->buildIndexes();
  //   }
  // }

  if (common::verbose) {
    LOG(INFO) << "#matches: " << n_for_tri << " #nrej_depth: " << n_rej_depth
              << " #nrej_repro: " << n_rej_repro
              << " #nrej_para: " << n_rej_para << " #nnew: " << nnew;
  }
}

void LocalMapping::SearchInNeighbors() {
  // Retrieve neighbor keyframes
  int nn = 10;
  if (mbMonocular)
    nn = 20;
  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

  vector<KeyFrame *> vpTargetKFs;
  for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(),
                                          vend = vpNeighKFs.end();
       vit != vend; vit++) {
    KeyFrame *pKFi = *vit;
    if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
      continue;
    vpTargetKFs.push_back(pKFi);
    pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

    // Extend to some second neighbors
    const vector<KeyFrame *> vpSecondNeighKFs =
        pKFi->GetBestCovisibilityKeyFrames(5);
    for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(),
                                            vend2 = vpSecondNeighKFs.end();
         vit2 != vend2; vit2++) {
      KeyFrame *pKFi2 = *vit2;
      if (pKFi2->isBad() ||
          pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
          pKFi2->mnId == mpCurrentKeyFrame->mnId)
        continue;
      vpTargetKFs.push_back(pKFi2);
    }
  }

  // Search matches by projection from current KF in target KFs
  SPMatcher matcher;

  vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();
  for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(),
                                    vend = vpTargetKFs.end();
       vit != vend; vit++) {
    KeyFrame *pKFi = *vit;

    matcher.Fuse(pKFi, vpMapPointMatches);
  }

  // Search matches by projection from target KFs in current KF
  vector<MapPoint *> vpFuseCandidates;
  vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

  for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(),
                                    vendKF = vpTargetKFs.end();
       vitKF != vendKF; vitKF++) {
    KeyFrame *pKFi = *vitKF;

    vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

    for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(),
                                      vendMP = vpMapPointsKFi.end();
         vitMP != vendMP; vitMP++) {
      MapPoint *pMP = *vitMP;
      if (!pMP)
        continue;

      if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
        continue;

      pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
      vpFuseCandidates.push_back(pMP);
    }
  }

  matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

  // Update points
  vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        pMP->ComputeDistinctiveDescriptors();

        pMP->UpdateNormalAndDepth();
      }
    }
  }

  mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::KeyFrameCullingOverride2() {
  vector<KeyFrame *> vpLocalKeyFrames =
      mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

  list<KeyFrame *> kf_list;
  for (auto kf : vpLocalKeyFrames) {
    if (kf->mnId != 0)
      kf_list.push_back(kf);
  }

  // (vpLocalKeyFrames.begin(), vpLocalKeyFrames.end());

  bool not_break = true;
  while (!kf_list.empty()) {
    KeyFrame *max_vit = nullptr;
    float ratio_max = 0.0f;
    vector<KeyFrame *> candidates;
    for (list<KeyFrame *>::iterator vit = kf_list.begin(); vit != kf_list.end();
         vit++) {
      KeyFrame *pKF = *vit;
      // if (pKF->mnId == 0)
      //   continue;
      const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

      int nObs = mapping::kf_culling_num_obs;
      const int thObs = nObs;
      int nRedundantObservations = 0;
      int nMPs = 0;

      for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
        MapPoint *pMP = vpMapPoints[i];
        if (pMP && !pMP->isBad()) {
          nMPs++;
          if (pMP->Observations() >= thObs) {
            nRedundantObservations++;
          }
        }
      }

      float ratio_redundant = nRedundantObservations * 1.0f / nMPs;
      if (ratio_redundant < mapping::kf_culling_cov_ratio) {
        vit = kf_list.erase(vit);
        vit--;
      } else {
        candidates.push_back(pKF);
        // ratio_max = ratio_redundant;
        // max_vit = pKF;
      }
    }

    // if (max_vit) {
    //   max_vit->SetBadFlag();
    //   kf_list.remove(max_vit);
    // }
    float min_dist = numeric_limits<float>::max();
    KeyFrame *min_kf = nullptr;
    for (auto &&pkf : candidates) {
      cv::Mat t0 = pkf->GetCameraCenter();
      cv::Mat t1 = pkf->GetParent()->GetCameraCenter();
      float dist = cv::norm(t0 - t1);
      if (dist < min_dist) {
        min_dist = dist;
        min_kf = pkf;
      }
    }
    cout << "culling candidates: " << candidates.size() << endl;
    if (min_kf) {
      min_kf->SetBadFlag();
      kf_list.remove(min_kf);
    }
  }
}

void LocalMapping::KeyFrameCullingOverride() {
  vector<KeyFrame *> vpLocalKeyFrames =
      mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

  list<KeyFrame *> kf_list;
  for (auto kf : vpLocalKeyFrames) {
    if (kf->mnId != 0)
      kf_list.push_back(kf);
  }

  // (vpLocalKeyFrames.begin(), vpLocalKeyFrames.end());

  bool not_break = true;
  while (!kf_list.empty()) {
    KeyFrame *max_vit = nullptr;
    float ratio_max = 0.0f;
    for (list<KeyFrame *>::iterator vit = kf_list.begin(); vit != kf_list.end();
         vit++) {
      KeyFrame *pKF = *vit;
      // if (pKF->mnId == 0)
      //   continue;
      const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

      int nObs = mapping::kf_culling_num_obs;
      const int thObs = nObs;
      int nRedundantObservations = 0;
      int nMPs = 0;

      for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
        MapPoint *pMP = vpMapPoints[i];
        if (pMP && !pMP->isBad()) {
          nMPs++;
          if (pMP->Observations() >= thObs) {
            nRedundantObservations++;
          }
        }
      }

      float ratio_redundant = nRedundantObservations * 1.0f / nMPs;
      if (ratio_redundant < mapping::kf_culling_cov_ratio) {
        vit = kf_list.erase(vit);
        vit--;
      } else if (ratio_redundant > ratio_max) {
        ratio_max = ratio_redundant;
        max_vit = pKF;
      }
    }

    if (max_vit) {
      max_vit->SetBadFlag();
      kf_list.remove(max_vit);
    }
  }
}

void LocalMapping::KeyFrameCulling() {
  vector<KeyFrame *> vpLocalKeyFrames =
      mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

  for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(),
                                    vend = vpLocalKeyFrames.end();
       vit != vend; vit++) {
    KeyFrame *pKF = *vit;
    if (pKF->mnId == 0)
      continue;
    const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

    int nObs = mapping::kf_culling_num_obs;
    const int thObs = nObs;
    int nRedundantObservations = 0;
    int nMPs = 0;

    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad()) {
          if (!mbMonocular) {
            if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
              continue;
          }

          nMPs++;
          if (pMP->Observations() > thObs) {
            const int &scaleLevel = pKF->mvKeysUn[i].octave;
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            int nObs = 0;
            for (map<KeyFrame *, size_t>::const_iterator
                     mit = observations.begin(),
                     mend = observations.end();
                 mit != mend; mit++) {
              KeyFrame *pKFi = mit->first;
              if (pKFi == pKF)
                continue;
              const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

              // Scale Condition
              if (scaleLeveli <= scaleLevel + 1) {
                nObs++;
                if (nObs >= thObs)
                  break;
              }
            }
            if (nObs >= thObs) {
              nRedundantObservations++;
            }
          }
        }
      }
    }

    if (nRedundantObservations > mapping::kf_culling_cov_ratio * nMPs) {
      if (common::verbose) {
        LOG(INFO) << "culling keyframe: " << (int)pKF->mnId << endl;
      }
      pKF->SetBadFlag();
    }
  }
}

} // namespace orbslam
