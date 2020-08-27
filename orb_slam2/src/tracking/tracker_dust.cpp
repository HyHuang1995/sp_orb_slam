#include "orb_slam/tracking/tracker.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <mutex>

#include "orb_slam/common.h"
#include "orb_slam/config.h"
#include "orb_slam/global.h"
#include "orb_slam/utils/timing.h"

#include "orb_slam/cv/sp_matcher.h"

#include "orb_slam/mapping/optimizer.h"

using namespace std;

namespace orbslam {

bool Tracking::trackFrameDustKFLocal() {
  UpdateLastFrameOverride();
  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint *>(nullptr));

  auto pRefKF = mLastFrame.mpReferenceKF;
  global::map->pDustRef = pRefKF;
  mCurrentFrame.mpReferenceKF = pRefKF;
  mps_for_track.clear();

  for (auto &mp : mLastFrame.mvpMapPoints) {
    if (mp && !mp->isBad()) {
      mps_for_track.push_back(mp);
      mp->mnTrackReferenceDust = mCurrentFrame.mnId;
      mp->in_view = false;
      mp->dust_match = false;
    }
  }

  if (mps_for_track.size() < 150) {
    const auto &&mps_kf = pRefKF->GetMapPointMatches();

    for (auto &mp : mps_kf) {
      if (mp && !mp->isBad() &&
          mp->mnTrackReferenceDust != mCurrentFrame.mnId) {
        mps_for_track.push_back(mp);
        mp->mnTrackReferenceDust = mCurrentFrame.mnId;
        mp->in_view = false;
        mp->dust_match = false;
      }
    }
  }
  // TODO: local map determination
  int n_kfs = 1;

  if (mps_for_track.size() < 150) {
    vector<KeyFrame *> vRefKFs;
    const auto &&vNeighs = pRefKF->GetBestCovisibilityKeyFrames(5);
    for (auto &kf : vNeighs) {
      if (kf->isBad())
        continue;
      n_kfs++;

      const vector<MapPoint *> &&vpMPs = kf->GetMapPointMatches();
      for (auto &mp_tmp : vpMPs) {
        if (!mp_tmp)
          continue;
        if (mp_tmp->isBad() ||
            mp_tmp->mnTrackReferenceDust == mCurrentFrame.mnId)
          continue;

        mps_for_track.push_back(mp_tmp);
        mp_tmp->mnTrackReferenceDust = mCurrentFrame.mnId;
        mp_tmp->in_view = false;

        // if (mps_for_track.size() >= 200)
        //   break;
      }
      if (mps_for_track.size() >= 150)
        break;

      // break;
    }
  }

  is_visible = std::vector<bool>(mps_for_track.size(), false);
  // int n_inlier = Optimizer::PoseOptimizationDust(&mCurrentFrame, pRefKF);
  int n_inlier = Optimizer::PoseOptimizationDust(&mCurrentFrame, mps_for_track,
                                                 is_visible);

  // float ratio_inlier = n_inlier * 1.0f / mps.size();
  // if (n_inlier < tracking::dust::th_ninlier && ratio_inlier <
  // tracking::dust::th_ratio) {
  if (n_inlier < tracking::dust::th_ninlier) {
    if (common::verbose) {
      LOG(ERROR) << "track dust failed at coarse tracking";
    }
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    return false;
  }

  cv::Mat occ_grid = mCurrentFrame.occ_grid.clone();
  // const float fx = mCurrentFrame.fx / 8.0f;
  // const float fy = mCurrentFrame.fy / 8.0f;
  // const float cx = (mCurrentFrame.cx - 3.5) / 8.0f;
  // const float cy = (mCurrentFrame.cy - 3.5) / 8.0f;

  // for (auto &mp : mLastFrame.mvpMapPoints) {
  int n_matches = 0;
  for (size_t i = 0; i < mps_for_track.size(); i++) {
    MapPoint *mp = mps_for_track[i];
    if (!mp->in_view || mp->isBad())
      continue;

    const uint16_t u = floor(mp->dust_proj_u);
    const uint16_t v = floor(mp->dust_proj_v);

    int16_t best_idx = -1, best_u, best_v;
    float best_dist = 0.75f;
    // float best_dist = std::numeric_limits<float>::max();
    // const cv::Mat &d_mp = mp->GetDescriptor();
    const cv::Mat d_mp = mp->getDescTrack();
    for (int16_t du = 0; du < 2; du++) {
      for (int16_t dv = 0; dv < 2; dv++) {
        auto u_tmp = u + du, v_tmp = v + dv;

        auto idx = occ_grid.at<int16_t>(v_tmp, u_tmp);
        if (idx != -1) {
          const cv::Mat &d1 = mCurrentFrame.mDescriptors.row(idx);
          float dist = SPMatcher::DescriptorDistance(d_mp, d1);
          if (dist < best_dist) {
            best_dist = dist;
            best_idx = idx;
            best_u = u_tmp;
            best_v = v_tmp;
          }
        }
      }
    }
    if (best_idx != -1) {
      // if (best_dist > 0.7f) {
      //   // 3D in absolute coordinates
      //   cv::Mat P = mp->GetWorldPos();

      //   // 3D in camera coordinates
      //   const float &PcX = Pc.at<float>(0);
      //   const float &PcY = Pc.at<float>(1);
      //   const float &PcZ = Pc.at<float>(2);

      //   // Project in image and check it is not outside
      //   const float invz = 1.0f / PcZ;
      //   const float u_ = mCurrentFrame.fx * PcX * invz + mCurrentFrame.cx;
      //   const float v_ = mCurrentFrame.fy * PcY * invz + mCurrentFrame.cy;

      //   const float du = mCurrentFrame.mvKeysUn[best_idx].pt.x - u_;
      //   const float dv = mCurrentFrame.mvKeysUn[best_idx].pt.y - v_;
      //   const float duv = du * du + dv * dv;
      //   const float thresh_dist = tracking::dust::c2_thresh /
      //   (tracking::dust::c2_thresh + duv); if (best_dist > thresh_dist)
      //     continue;
      // }

      mCurrentFrame.mvpMapPoints[best_idx] = mp; // add mappoint
      occ_grid.at<int16_t>(best_v, best_u) = -1;
      n_matches++;

      mp->dust_match = true; // FIXME: viz
    }
  }

  if (n_matches < tracking::dust::th_nmatch) {
    if (common::verbose) {
      LOG(ERROR) << "track dust failed at patch-wise association";
    }
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    return false;
  }

  // FIXME: add pose optimization
  int nopt_inlier = Optimizer::PoseOptimizationDustPost(&mCurrentFrame);

  // for (int i = 0; i < mCurrentFrame.N; i++) {
  //   if (mCurrentFrame.mvpMapPoints[i]) {
  //     if (mCurrentFrame.mvbOutlier[i]) {
  //       MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

  //       mCurrentFrame.mvpMapPoints[i] = nullptr;
  //       mCurrentFrame.mvbOutlier[i] = false;
  //       pMP->mbTrackInView = false;
  //       pMP->mnLastFrameSeen = mCurrentFrame.mnId;
  //       // nmatches--;
  //       // nout++;
  //     }
  //     // else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
  //     //   nmatchesMap++;
  //   }
  // }

  if (common::verbose) {
    LOG(INFO) << " #n_kfs " << n_kfs << " #inliers: " << n_inlier
              << " #matches: " << n_matches << " #ba_inlier: " << nopt_inlier;
  }

  {
    inlier_coarse.push_back(nopt_inlier);
    inlier_coarse_ratio.push_back(nopt_inlier * 1.0 / mCurrentFrame.N);

    n_inlier_coarse += nopt_inlier;
    total_coarse.push_back(n_inlier_coarse);
  }
  // logger->addScalar("dust_inlier", mCurrentFrame.mnId, n_inlier);
  // logger->addScalar("dust_match", mCurrentFrame.mnId, n_matches);

  // return nopt_inlier * 1.0f / ;
  if (nopt_inlier * 1.0f / n_matches > tracking::dust::th_ratio) {
    return true;
  } else {
    if (common::verbose) {
      LOG(ERROR) << "track dust failed at pose opt";
    }

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    return false;
  }
}

bool Tracking::trackFrameDustKF() {
  // cout << "track frame dust" << endl;
  UpdateLastFrame();
  // mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint *>(nullptr));

  auto pRefKF = mLastFrame.mpReferenceKF;
  mCurrentFrame.mpReferenceKF = pRefKF;
  fill(pRefKF->is_mp_visible_.begin(), pRefKF->is_mp_visible_.end(), false);
  auto &&mps = pRefKF->GetMapPointMatches();

  int n_inlier = Optimizer::PoseOptimizationDust(&mCurrentFrame, pRefKF);

  if (n_inlier < 20) {
    cout << "track dust failed" << endl;
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    return false;
  }

  cv::Mat occ_grid = mCurrentFrame.occ_grid.clone();
  const float fx = mCurrentFrame.fx / 8.0f;
  const float fy = mCurrentFrame.fy / 8.0f;
  const float cx = (mCurrentFrame.cx - 3.5) / 8.0f;
  const float cy = (mCurrentFrame.cy - 3.5) / 8.0f;

  // for (auto &mp : mLastFrame.mvpMapPoints) {
  int n_matches = 0;
  for (size_t i = 0; i < mpReferenceKF->N; i++) {
    if (!mpReferenceKF->is_mp_visible_[i])
      continue;

    MapPoint *mp = mps[i];

    // 3D in absolute coordinates
    cv::Mat P = mp->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mCurrentFrame.mRcw * P + mCurrentFrame.mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f)
      continue;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u_ = fx * PcX * invz + cx;
    const float v_ = fy * PcY * invz + cy;
    const int u = floor(u_);
    const int v = floor(v_);

    int16_t best_idx = -1, best_u, best_v;
    float best_dist = 0.7f;
    const cv::Mat &d_mp = mp->getDescTrack();
    for (int16_t du = 0; du < 2; du++) {
      for (int16_t dv = 0; dv < 2; dv++) {
        auto u_tmp = u + du, v_tmp = v + dv;

        auto idx = occ_grid.at<int16_t>(v_tmp, u_tmp);
        if (idx != -1) {
          const cv::Mat &d1 = mCurrentFrame.mDescriptors.row(idx);
          float dist = SPMatcher::DescriptorDistance(d_mp, d1);
          if (dist < best_dist) {
            best_dist = dist;
            best_idx = idx;
            best_u = u_tmp;
            best_v = v_tmp;
          }
        }
      }
    }
    if (best_idx != -1) {
      mCurrentFrame.mvpMapPoints[best_idx] = mp; // add mappoint
      occ_grid.at<int16_t>(best_v, best_u) = -1;
      n_matches++;
    }
  }

  LOG(INFO) << "#inliers: " << n_inlier << " #matches: " << n_matches;

  if (n_matches >= 20)
    return true;
  else {
    LOG(ERROR) << "track dust failed at patch-wise association";
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    return false;
  }
}

bool Tracking::trackFrameDust() {
  UpdateLastFrame();
  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint *>(nullptr));

  int n_inlier = Optimizer::PoseOptimizationDust(&mCurrentFrame, &mLastFrame);

  if (n_inlier < 20) {
    cout << "track dust failed" << endl;
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    return false;
  }

  cv::Mat occ_grid = mCurrentFrame.occ_grid.clone();
  const float fx = mCurrentFrame.fx / 8.0f;
  const float fy = mCurrentFrame.fy / 8.0f;
  const float cx = (mCurrentFrame.cx - 3.5) / 8.0f;
  const float cy = (mCurrentFrame.cy - 3.5) / 8.0f;

  int n_matches = 0;
  for (size_t i = 0; i < mLastFrame.N; i++) {
    if (!mLastFrame.is_mp_visible_[i])
      continue;
    MapPoint *mp = mLastFrame.mvpMapPoints[i];

    // 3D in absolute coordinates
    cv::Mat P = mp->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mCurrentFrame.mRcw * P + mCurrentFrame.mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f)
      continue;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u_ = fx * PcX * invz + cx;
    const float v_ = fy * PcY * invz + cy;
    const int u = floor(u_);
    const int v = floor(v_);

    int16_t best_idx = -1, best_u, best_v;
    float best_dist = std::numeric_limits<float>::max();
    const cv::Mat &d_mp = mp->getDescTrack();
    for (int16_t du = 0; du < 2; du++) {
      for (int16_t dv = 0; dv < 2; dv++) {
        auto u_tmp = u + du, v_tmp = v + dv;

        auto idx = occ_grid.at<int16_t>(v_tmp, u_tmp);
        if (idx != -1) {
          const cv::Mat &d1 = mCurrentFrame.mDescriptors.row(idx);
          float dist = SPMatcher::DescriptorDistance(d_mp, d1);
          if (dist < best_dist) {
            best_dist = dist;
            best_idx = idx;
            best_u = u_tmp;
            best_v = v_tmp;
          }
        }
      }
    }
    if (best_idx != -1) {
      mCurrentFrame.mvpMapPoints[best_idx] = mp; // add mappoint
      occ_grid.at<int16_t>(best_v, best_u) = -1;
      n_matches++;
    }
  }

  LOG(INFO) << "#inliers: " << n_inlier << " #matches: " << n_matches;

  return true;
}

bool Tracking::trackFrameHeat() {
  cout << "track frame heat" << endl;

  std::vector<MapPoint *> mps;
  for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(),
                                    vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;

    if (pMP->isBad())
      continue;

    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, tracking::map::th_view_cos)) {
      mps.push_back(pMP);
    }
  }

  int n_inlier = Optimizer::PoseOptimizationHeat(&mCurrentFrame, &mLastFrame);
  // int n_inlier = Optimizer::PoseOptimizationDust(&mCurrentFrame, mps);

  // int nmatchesMap = 0;
  // for (int i = 0; i < mCurrentFrame.N; i++) {
  //   if (mLastFrame.mvpMapPoints[i]) {
  //     if (mCurrentFrame.mvbOutlier[i]) {
  //       MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

  //       mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
  //       mCurrentFrame.mvbOutlier[i] = false;
  //       pMP->mbTrackInView = false;
  //       pMP->mnLastFrameSeen = mCurrentFrame.mnId;
  //       nmatches--;
  //     } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
  //       nmatchesMap++;
  //       if (mLastFrame.mvpMapPoints[i]->Observations()> 0)
  //   }
  // }
  // if (n_inlier > 20) {
  //   return true;
  // } else {
  //   mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
  //   return false;
  // }
}
} // namespace orbslam
