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
#include <set>

#include <Eigen/Dense>

// #include "dbow3/BowVector.h"
// #include "dbow3/FeatureVector.h"

#include "frame.h"
#include "mappoint.h"

#include "../cv/depth_filter.h"
#include "../cv/orb_extractor.h"
// #include "../cv/orb_vocabulary.h"

#include <unordered_set>

namespace orbslam {

// class Map;
class MapPoint;
class Frame;
class Seed;
// class KeyFrameDatabase;

class KeyFrame {
public:
  KeyFrame(Frame &F);

  void SetPose(const cv::Mat &Tcw);
  cv::Mat GetPose();
  cv::Mat GetPoseInverse();
  cv::Mat GetCameraCenter();
  cv::Mat GetStereoCenter();
  cv::Mat GetRotation();
  cv::Mat GetTranslation();

  // Bag of Words Representation
  // void ComputeBoW();

  // Covisibility graph functions
  void AddConnection(KeyFrame *pKF, const int &weight);
  void EraseConnection(KeyFrame *pKF);
  void UpdateConnections();
  void UpdateBestCovisibles();
  std::set<KeyFrame *> GetConnectedKeyFrames();
  std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();
  std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);
  std::vector<KeyFrame *> GetCovisiblesByWeight(const int &w);
  int GetWeight(KeyFrame *pKF);

  // Spanning tree functions
  void AddChild(KeyFrame *pKF);
  void EraseChild(KeyFrame *pKF);
  void ChangeParent(KeyFrame *pKF);
  std::set<KeyFrame *> GetChilds();
  KeyFrame *GetParent();
  bool hasChild(KeyFrame *pKF);

  // Loop Edges
  void AddLoopEdge(KeyFrame *pKF);
  std::set<KeyFrame *> GetLoopEdges();

  // MapPoint observation functions
  void AddMapPoint(MapPoint *pMP, const size_t &idx);
  void EraseMapPointMatch(const size_t &idx);
  void EraseMapPointMatch(MapPoint *pMP);
  void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);
  std::set<MapPoint *> GetMapPoints();
  std::vector<MapPoint *> GetMapPointMatches();
  int TrackedMapPoints(const int &minObs);
  void getTrackedInCommon(std::unordered_set<MapPoint *> curr_mps, int &num_obs,
                          int &total_obs);
  MapPoint *GetMapPoint(const size_t &idx);

  // KeyPoint functions
  std::vector<size_t> GetFeaturesInArea(const float &x, const float &y,
                                        const float &r) const;
  cv::Mat UnprojectStereo(int i);

  // Image
  bool IsInImage(const float &x, const float &y) const;

  // Enable/Disable bad flag changes
  void SetNotErase();
  void SetErase();

  // Set/check bad flag
  void SetBadFlag();
  bool isBad();

  // Compute Scene Depth (q=2 median). Used in monocular.
  float ComputeSceneMedianDepth(const int q);
  float ComputeSceneMeanDepth(const int q);

  static bool weightComp(int a, int b) { return a > b; }

  static bool lId(KeyFrame *pKF1, KeyFrame *pKF2) {
    return pKF1->mnId < pKF2->mnId;
  }

  // The following variables are accesed from only 1 thread or never change (no
  // mutex needed).
  // ADD(flann)
  void buildIndexes();
  void buildIndexesMps();

  std::vector<cv::DMatch> matchMps(Frame *frame);

public:
  // ADD(loop)
  cv::Mat global_desc;

  // ADD(occ_grid)
  cv::Mat occ_grid;
  int grid_cols, grid_rows;

  // ADD(statistics for depth)
  float scene_depth_mean, scene_depth_median, scene_depth_min, scene_depth_max;

  // ADD(flann)
  cv::Ptr<cv::FlannBasedMatcher> flann, flann_mps;
  cv::Mat mDescReamin, mDescMps;
  std::vector<size_t> mIndicesRemain, mIndicesMps;

  // ADD(depth filter)
  cv::Mat mK_grid;
  // bool computeDepthFromTriangulation(const cv::Point2f &kp1,
  //                                    const cv::Point2f &kp2);
  int initializeSeeds();
  int updateSeeds(Frame *frame);
  std::vector<std::shared_ptr<Seed>> seeds_;
  std::vector<cv::Point3f> rgb_;

  // ADD(for debug use)
  cv::Mat mIm;

  static long unsigned int nNextId;
  long unsigned int mnId;
  const long unsigned int mnFrameId;

  const double mTimeStamp;

  // Grid (to speed up feature matching)
  const int mnGridCols;
  const int mnGridRows;
  const float mfGridElementWidthInv;
  const float mfGridElementHeightInv;

  // Variables used by the tracking
  long unsigned int mnTrackReferenceForFrame;
  long unsigned int mnFuseTargetForKF;

  // Variables used by the local mapping
  long unsigned int mnBALocalForKF;
  long unsigned int mnBAFixedForKF;

  // Variables used by the keyframe database
  long unsigned int mnLoopQuery;
  int mnLoopWords;
  float mLoopScore;
  long unsigned int mnRelocQuery;
  int mnRelocWords;
  float mRelocScore;

  // Variables used by loop closing
  cv::Mat mTcwGBA;
  cv::Mat mTcwBefGBA;
  long unsigned int mnBAGlobalForKF;

  // Calibration parameters
  const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

  // Number of KeyPoints
  const int N;

  // KeyPoints, stereo coordinate and descriptors (all associated by an index)
  const std::vector<cv::KeyPoint> mvKeys;
  const std::vector<cv::KeyPoint> mvKeysUn;
  std::vector<Eigen::Vector2f> cov2_inv_;

  // ADD: visibility flag
  std::vector<bool> is_mp_visible_;

  const std::vector<float> mvuRight;
  const std::vector<float> mvDepth;
  const cv::Mat mDescriptors;

  // Pose relative to parent (this is computed when bad flag is activated)
  cv::Mat mTcp;

  // Scale
  const int mnScaleLevels;
  const float mfScaleFactor;
  const float mfLogScaleFactor;
  const std::vector<float> mvScaleFactors;
  const std::vector<float> mvLevelSigma2;
  const std::vector<float> mvInvLevelSigma2;

  // Image bounds and calibration
  const int mnMinX;
  const int mnMinY;
  const int mnMaxX;
  const int mnMaxY;
  const cv::Mat mK;

  // The following variables need to be accessed trough a mutex to be thread
  // safe.
protected:
  // SE3 Pose and camera center
  cv::Mat Tcw;
  cv::Mat Twc;
  cv::Mat Ow;

  cv::Mat Cw; // Stereo middel point. Only for visualization

  // MapPoints associated to keypoints
  std::vector<MapPoint *> mvpMapPoints;

  // BoW
  // ORBVocabulary *mpORBvocabulary;

  // Grid over the image to speed up feature matching
  std::vector<std::vector<std::vector<size_t>>> mGrid;

  // Covisibility Graph
  std::map<KeyFrame *, int> mConnectedKeyFrameWeights;
  std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;
  std::vector<int> mvOrderedWeights;

  bool mbFirstConnection;
  KeyFrame *mpParent;
  std::set<KeyFrame *> mspChildrens;
  std::set<KeyFrame *> mspLoopEdges;

  bool mbNotErase;
  bool mbToBeErased;
  bool mbBad;

  float mHalfBaseline;

  std::shared_mutex mMutexPose;
  std::mutex mMutexConnections;
  std::mutex mMutexFeatures;
};

} // namespace orbslam
