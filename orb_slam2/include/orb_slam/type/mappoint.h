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
#include <shared_mutex>
#include <map>

#include <opencv2/core/core.hpp>

#include "frame.h"
#include "keyframe.h"
#include "map.h"

namespace orbslam {

class KeyFrame;
class Map;
class Frame;

class MapPoint {
public:
  MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF);
  MapPoint(const cv::Mat &Pos, Frame *pFrame, const int &idxF);

  void SetWorldPos(const cv::Mat &Pos);
  cv::Mat GetWorldPos();

  cv::Mat GetNormal();
  KeyFrame *GetReferenceKeyFrame();

  std::map<KeyFrame *, size_t> GetObservations();
  int Observations();

  void AddObservation(KeyFrame *pKF, size_t idx);
  void EraseObservation(KeyFrame *pKF);

  int GetIndexInKeyFrame(KeyFrame *pKF);
  bool IsInKeyFrame(KeyFrame *pKF);

  void SetBadFlag();
  bool isBad();

  void Replace(MapPoint *pMP);
  MapPoint *GetReplaced();

  void IncreaseVisible(int n = 1);
  void IncreaseFound(int n = 1);
  float GetFoundRatio();
  inline int GetFound() { return mnFound; }

  void ComputeDistinctiveDescriptors();

  cv::Mat GetDescriptor();

  void UpdateNormalAndDepth();

  float GetMinDistanceInvariance();
  float GetMaxDistanceInvariance();
  int PredictScale(const float &currentDist, KeyFrame *pKF);
  int PredictScale(const float &currentDist, Frame *pF);

  // ADD(update desc)
  void updateDescTrack(KeyFrame *pKF, const int idx);

  cv::Mat getDescTrack();

public:
  long unsigned int mnId; ///< Global ID for MapPoint
  static long unsigned int nNextId;
  const long int mnFirstKFid;
  const long int mnFirstFrame;
  int nObs;

  // ADD(dust tracking):
  bool in_view, dust_match;
  float dust_proj_u, dust_proj_v;

  float mTrackProjX;
  float mTrackProjY;
  float mTrackProjXR;
  int mnTrackScaleLevel;
  float mTrackViewCos;
  float depth_frame;

  bool mbTrackInView;

  long unsigned int mnTrackReferenceForFrame;

  // ADD: avoid rebundance
  long unsigned int mnTrackReferenceDust;

  long unsigned int mnLastFrameSeen;

  // Variables used by local mapping
  long unsigned int mnBALocalForKF;
  long unsigned int mnFuseCandidateForKF;

  // Variables used by loop closing
  long unsigned int mnLoopPointForKF;
  long unsigned int mnCorrectedByKF;
  long unsigned int mnCorrectedReference;
  cv::Mat mPosGBA;
  long unsigned int mnBAGlobalForKF;

  static std::mutex mGlobalMutex;

protected:
  // Position in absolute coordinates
  cv::Mat mWorldPos;

  // Keyframes observing the point and associated index in keyframe
  std::map<KeyFrame *, size_t> mObservations;

  // Mean viewing direction
  cv::Mat mNormalVector;

  // Best descriptor to fast matching
  cv::Mat mDescriptor;

  cv::Mat mDescTrack;

  // Reference KeyFrame
  KeyFrame *mpRefKF;

  // Tracking counters
  int mnVisible;
  int mnFound;

  // Bad flag (we do not currently erase MapPoint from memory)
  bool mbBad;
  MapPoint *mpReplaced;

  // Scale invariance distances
  float mfMinDistance;
  float mfMaxDistance;

  std::shared_mutex mMutexPos;
  std::mutex mMutexFeatures;

  // ADD(update desc)
  std::mutex mutex_desc_track;
};

} // namespace orbslam
