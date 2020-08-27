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

#include <opencv2/core/core.hpp>
// #include <opencv2/features2d/features2d.hpp>

#include "../type/type.h"

#include "../cv/orb_extractor.h"
#include "../io/data_loader.h"
#include "../type/frame.h"
#include "initializer.h"

#include <mutex>

namespace orbslam {

class Tracking {

public:
  Tracking();

  virtual ~Tracking() = default;

  cv::Mat trackFrame(DataFrame::ConstPtr data_frame);

  cv::Mat trackFrame(const double &timestamp, const cv::Mat &im1,
                     const cv::Mat &im2 = cv::Mat());

  void reset();

  void InformOnlyTracking(const bool &flag);

public:
  // summary for tracking module
  int n_fail_dust = 0;
  int n_inlier_coarse = 0;
  int n_inlier_fine = 0;

  std::vector<int> total_coarse, total_fine;

  // Tracking states
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
  };

  eTrackingState mState;
  eTrackingState mLastProcessedState;

  // Input sensor:MONOCULAR, STEREO, RGBD
  // int mSensor;
  // ADD(dust_debug): viz for tracking
  std::vector<MapPoint *> mps_for_track;
  std::vector<MapPoint *> mvpLocalMapPointsViz;
  std::vector<bool> is_visible;

  // Current Frame
  Frame mCurrentFrame;
  cv::Mat mImGray;

  // Initialization Variables (Monocular)
  std::vector<int> mvIniLastMatches;
  std::vector<int> mvIniMatches;
  std::vector<cv::Point2f> mvbPrevMatched;
  std::vector<cv::Point3f> mvIniP3D;
  Frame mInitialFrame;

  std::list<cv::Mat> mlRelativeFramePoses;
  std::list<KeyFrame *> mlpReferences;
  std::list<double> mlFrameTimes;
  std::list<bool> mlbLost;

  bool mbOnlyTracking;

  void Reset();

public:
  // ADD(statistics)
  void report();

  std::vector<int> inlier_coarse, inlier_fine;
  std::vector<float> inlier_coarse_ratio, inlier_fine_ratio;

  // Main tracking function. It is independent of the input sensor.
  void track();

  virtual void setFrameData(const double &timestamp, const cv::Mat &im1,
                            const cv::Mat &im2 = cv::Mat()) = 0;

  virtual void Initialization() = 0;

  bool trackReferenceKeyFrameANN();

  // ADD(dust): dust tracking
  bool trackFrameDust();
  bool trackFrameDustKF();
  bool trackFrameDustKFLocal();

  bool trackFrameHeat();

  void CheckReplacedInLastFrame();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();
  void UpdateLastFrameOverride();
  bool TrackWithMotionModel();

  bool Relocalization();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();

  bool TrackLocalMap();
  int SearchLocalPoints();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();
  bool NeedNewKeyFrameOverride();
  void CreateNewKeyFrameOverride();

  bool mbVO;

  BaseExtractor *mpORBextractorLeft, *mpORBextractorRight;
  BaseExtractor *mpIniORBextractor;

  // Initalization (only for monocular)
  Initializer *mpInitializer;

  // Local Map
  KeyFrame *mpReferenceKF;
  std::vector<KeyFrame *> mvpLocalKeyFrames;
  std::vector<MapPoint *> mvpLocalMapPoints;

  // Calibration matrix
  cv::Mat mK;
  cv::Mat mDistCoef;
  float mbf;

  // New KeyFrame rules (according to fps)
  int mMinFrames;
  int mMaxFrames;

  float mThDepth;

  float mDepthMapFactor;

  int mnMatchesInliers;

  KeyFrame *mpLastKeyFrame;
  Frame mLastFrame;
  unsigned int mnLastKeyFrameId;
  unsigned int mnLastRelocFrameId;

  cv::Mat mVelocity;

  bool mbRGB;

  std::list<MapPoint *> mlpTemporalPoints;
};

} // namespace orbslam
