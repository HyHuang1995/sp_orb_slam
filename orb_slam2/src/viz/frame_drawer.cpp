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

#include "orb_slam/viz/frame_drawer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>

namespace orbslam {

using namespace std;

inline cv::Vec3b makeJet3B(float id) {
  if (id <= 0)
    return cv::Vec3b(128, 0, 0);
  if (id >= 1)
    return cv::Vec3b(0, 0, 128);

  int icP = (id * 8);
  float ifP = (id * 8) - icP;

  if (icP == 0)
    return cv::Vec3b(255 * (0.5 + 0.5 * ifP), 0, 0);
  if (icP == 1)
    return cv::Vec3b(255, 255 * (0.5 * ifP), 0);
  if (icP == 2)
    return cv::Vec3b(255, 255 * (0.5 + 0.5 * ifP), 0);
  if (icP == 3)
    return cv::Vec3b(255 * (1 - 0.5 * ifP), 255, 255 * (0.5 * ifP));
  if (icP == 4)
    return cv::Vec3b(255 * (0.5 - 0.5 * ifP), 255, 255 * (0.5 + 0.5 * ifP));
  if (icP == 5)
    return cv::Vec3b(0, 255 * (1 - 0.5 * ifP), 255);
  if (icP == 6)
    return cv::Vec3b(0, 255 * (0.5 - 0.5 * ifP), 255);
  if (icP == 7)
    return cv::Vec3b(0, 0, 255 * (1 - 0.5 * ifP));
  return cv::Vec3b(255, 255, 255);
}

void FrameDrawer::updateCoarse(Tracking *tracker) {
  unique_lock<mutex> lock(mMutex);

  is_ok = (tracker->mState == Tracking::OK);

  cv::Mat coarse = tracker->mImGray.clone();
  cv::cvtColor(coarse, mVizCoarse, cv::COLOR_GRAY2BGR);
  const auto &mps = tracker->mps_for_track;
  const auto &visible = tracker->is_visible;
  const auto &rot = tracker->mCurrentFrame.mRcw;
  const auto &trans = tracker->mCurrentFrame.mtcw;

  const auto fx = tracker->mCurrentFrame.fx;
  const auto fy = tracker->mCurrentFrame.fy;
  const auto cx = tracker->mCurrentFrame.cx;
  const auto cy = tracker->mCurrentFrame.cy;

  const auto height = mVizCoarse.rows;
  const auto width = mVizCoarse.cols;

  // for (const auto &mp : mps) {
  cv::Scalar color;
  for (size_t i = 0; i < mps.size(); i++) {
    const auto mp = mps[i];
    if (!mp || mp->isBad())
      continue;

    if (!mp->in_view)
      color = cv::Scalar(0, 255, 255);
    else if (!mp->dust_match)
      color = cv::Scalar(0, 0, 255);
    else
      color = cv::Scalar(0, 255, 0);

    const cv::Mat Pc = rot * mp->GetWorldPos() + trans;
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    const float invz = 1.0f / PcZ;
    const float u = fx * PcX * invz + cx;
    const float v = fy * PcY * invz + cy;

    if (u < 5 || u > width - 5 || v < 5 || v > height - 5)
      continue;

    cv::Point2f pt1, pt2;
    pt1.x = u - 5;
    pt1.y = v - 5;
    pt2.x = u + 5;
    pt2.y = v + 5;

    // pt1.x = vCurrentKeys[i].pt.x - r;
    // pt1.y = vCurrentKeys[i].pt.y - r;
    // pt2.x = vCurrentKeys[i].pt.x + r;
    // pt2.y = vCurrentKeys[i].pt.y + r;
    // pt1.x = pt1.x < 0.0f ? 0.0f

    cv::rectangle(mVizCoarse, pt1, pt2, color);
    cv::circle(mVizCoarse, cv::Point(u, v), 2, color, -1);
  }

  // cv::imshow("coarse", mVizCoarse);
}

FrameDrawer::FrameDrawer() {
  mState = Tracking::SYSTEM_NOT_READY;
  // mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  mIm = cv::Mat(480, 752, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat FrameDrawer::DrawFrame() {
  cv::Mat im;
  vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
  vector<int>
      vMatches; // Initialization: correspondeces with reference keypoints
  vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
  // vector<float>();
  vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
  int state;                // Tracking state

  {
    unique_lock<mutex> lock(mMutex);
    state = mState;
    if (mState == Tracking::SYSTEM_NOT_READY)
      mState = Tracking::NO_IMAGES_YET;

    mIm.copyTo(im);

    if (mState == Tracking::NOT_INITIALIZED) {
      vCurrentKeys = mvCurrentKeys;
      vIniKeys = mvIniKeys;
      vMatches = mvIniMatches;
    } else if (mState == Tracking::OK) {
      vCurrentKeys = mvCurrentKeys;
      vbVO = mvbVO;
      vbMap = mvbMap;
    } else if (mState == Tracking::LOST) {
      vCurrentKeys = mvCurrentKeys;
    }

    img_sp_viz = img_sp.clone();
  }

  if (im.channels() < 3) // this should be always true
    cvtColor(im, im, CV_GRAY2BGR);

  int height = im.rows;
  int width = im.cols;

  // Draw
  if (state == Tracking::NOT_INITIALIZED) // INITIALIZING
  {
  } else if (state == Tracking::OK) // TRACKING
  {
    int size = mvCurrentKeys.size();
    for (int i = 0; i < size; i++) {
      auto &&kp = mvCurrentKeys[i];
      // cv::circle(im, kp.pt, 3.0, makeJet3B(i * 0.8f / size), -1);
      auto pt1 = kp.pt;
      auto pt2 = kp.pt;
      pt1.x -= 3;
      pt1.y -= 3;
      pt2.x += 3;
      pt2.y += 3;
      cv::rectangle(im, cv::Rect(pt1, pt2), makeJet3B(i * 0.85f / size), -1);
    }
  }

  // static int count = 0;

  return im;
}

void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText) {
  stringstream s;
  if (nState == Tracking::NO_IMAGES_YET)
    s << " WAITING FOR IMAGES";
  else if (nState == Tracking::NOT_INITIALIZED)
    s << " TRYING TO INITIALIZE ";
  else if (nState == Tracking::OK) {
    if (!mbOnlyTracking)
      s << "SLAM MODE |  ";
    else
      s << "LOCALIZATION | ";
    int nKFs = mpMap->KeyFramesInMap();
    int nMPs = mpMap->MapPointsInMap();
    s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
    if (mnTrackedVO > 0)
      s << ", + VO matches: " << mnTrackedVO;
  } else if (nState == Tracking::LOST) {
    s << " TRACK LOST. TRYING TO RELOCALIZE ";
  } else if (nState == Tracking::SYSTEM_NOT_READY) {
    // s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
  }

  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

  imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
  im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
  imText.rowRange(im.rows, imText.rows) =
      cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
  cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void FrameDrawer::Update(Tracking *pTracker) {
  static int iter = 0;
  unique_lock<mutex> lock(mMutex);
  pTracker->mImGray.copyTo(mIm);
  mvCurrentKeys = pTracker->mCurrentFrame.mvKeysViz;
  mvCov2_inv = pTracker->mCurrentFrame.cov2_inv_;
  seeds = pTracker->mCurrentFrame.seeds_;
  N = mvCurrentKeys.size();

  mvbVO = vector<bool>(N, false);
  mvbMap = vector<bool>(N, false);
  mbOnlyTracking = pTracker->mbOnlyTracking;

  if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED) {
    mvIniKeys = pTracker->mInitialFrame.mvKeys;
    mvIniMatches = pTracker->mvIniMatches;
  } else if (pTracker->mLastProcessedState == Tracking::OK) {
    for (int i = 0; i < N; i++) {
      MapPoint *pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
      if (pMP) {
        if (!pTracker->mCurrentFrame.mvbOutlier[i]) {
          if (pMP->Observations() > 0)
            mvbMap[i] = true;
          else
            mvbVO[i] = true;
        }
      }
    }
  }
  mState = static_cast<int>(pTracker->mLastProcessedState);

  iter++;
}

} // namespace orbslam
