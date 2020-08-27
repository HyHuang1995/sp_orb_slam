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

#include <vector>

// #include "dbow3/BowVector.h"
// #include "dbow3/FeatureVector.h"

#include "keyframe.h"
#include "mappoint.h"

#include "../cv/orb_extractor.h"
// #include "../cv/orb_vocabulary.h"

#include "../cv/depth_filter.h"

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

namespace orbslam {
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame {
public:
  Frame();

  // Copy constructor.
  Frame(const Frame &frame);

  Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
        BaseExtractor *extractorLeft, BaseExtractor *extractorRight, cv::Mat &K,
        cv::Mat &distCoef, const float &bf, const float &thDepth);

  // Constructor for RGB-D cameras.
  Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
        BaseExtractor *extractor, cv::Mat &K, cv::Mat &distCoef,
        const float &bf, const float &thDepth);

  // Constructor for Monocular cameras.
  Frame(const cv::Mat &imGray, const double &timeStamp,
        BaseExtractor *extractor, cv::Mat &K, cv::Mat &distCoef,
        const float &bf, const float &thDepth);

  // Extract ORB on the image. 0 for left image and 1 for right image.
  void ExtractORB(int flag, const cv::Mat &im);

  // Compute Bag of Words representation.
  // void ComputeBoW();

  // Set the camera pose.
  void SetPose(cv::Mat Tcw);

  // Computes rotation, translation and camera center matrices from the camera
  // pose.
  void UpdatePoseMatrices();

  // Returns the camera center.
  inline cv::Mat GetCameraCenter() { return mOw.clone(); }

  inline cv::Mat GetRotationInverse() { return mRwc.clone(); }

  bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

  bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

  std::vector<size_t> GetFeaturesInArea(const float &x, const float &y,
                                        const float &r, const int minLevel = -1,
                                        const int maxLevel = -1) const;

  void ComputeStereoMatches();

  void ComputeStereoFromRGBD(const cv::Mat &imDepth);

  cv::Mat UnprojectStereo(const int &i);

  // ADD(flann)
  void buildIndex();

public:
  // ADD(loop)
  cv::Mat global_desc;

  // ADD(flann)
  cv::Ptr<cv::FlannBasedMatcher> flann;

  // ADD(depth filter)
  std::vector<std::shared_ptr<Seed>> seeds_;

  // Vocabulary used for relocalization.
  // ORBVocabulary *mpORBvocabulary;

  // Feature extractor. The right is used only in the stereo case.
  BaseExtractor *mpORBextractorLeft, *mpORBextractorRight;

  // Frame timestamp.
  double mTimeStamp;

  // Calibration matrix and OpenCV distortion parameters.
  cv::Mat mK;
  static float fx;
  static float fy;
  static float cx;
  static float cy;
  static float invfx;
  static float invfy;
  cv::Mat mDistCoef;

  // DBoW3::BowVector mBowVec;
  // DBoW3::FeatureVector mFeatVec;

  // Stereo baseline multiplied by fx.
  float mbf;

  // Stereo baseline in meters.
  float mb;

  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  float mThDepth;

  // Number of KeyPoints.
  int N; ///< KeyPoints数量

  std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
  std::vector<cv::KeyPoint> mvKeysViz;

  std::vector<cv::KeyPoint> mvKeysUn;

  std::vector<float> depth;

  // ADD: variance
  std::vector<Eigen::Vector2f> cov2_inv_;
  std::vector<float> score;

  // ADD: dustbin
  cv::Mat dust_, heat_;

  // ADD: occ grid
  cv::Mat occ_grid;
  int grid_cols, grid_rows;

  std::vector<float> mvuRight;
  std::vector<float> mvDepth;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat mDescriptors, mDescriptorsRight;

  // MapPoints associated to keypoints, NULL pointer if no association.
  std::vector<MapPoint *> mvpMapPoints;
  std::vector<bool> is_mp_visible_;
  std::vector<size_t> idx_tracked_last_;

  // Flag to identify outlier associations.
  std::vector<bool> mvbOutlier;

  // Keypoints are assigned to cells in a grid to reduce matching complexity
  // when projecting MapPoints.
  static float mfGridElementWidthInv;
  static float mfGridElementHeightInv;
  std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

  // Camera pose.
  cv::Mat mTcw;

  // Current and Next Frame id.
  static long unsigned int nNextId; ///< Next Frame id.
  long unsigned int mnId;           ///< Current Frame id.

  // Reference Keyframe.
  KeyFrame *mpReferenceKF;

  // Scale pyramid info.
  int mnScaleLevels;
  float mfScaleFactor;
  float mfLogScaleFactor; //
  std::vector<float> mvScaleFactors;
  std::vector<float> mvInvScaleFactors;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;

  // Undistorted Image Bounds (computed once).
  static float mnMinX;
  static float mnMaxX;
  static float mnMinY;
  static float mnMaxY;

  static bool mbInitialComputations;

public:
  void UndistortKeyPoints();

  void ComputeImageBounds(const cv::Mat &imLeft);

  void AssignFeaturesToGrid();

  // Rotation, translation and camera center
  cv::Mat mRcw;
  cv::Mat mtcw;
  cv::Mat mRwc;
  cv::Mat mOw;
};

} // namespace orbslam
