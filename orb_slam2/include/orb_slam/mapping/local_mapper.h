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

#include <condition_variable>
#include <mutex>

#include "../type/keyframe.h"
#include "../type/mappoint.h"

namespace orbslam {

class LocalMapping {
public:
  LocalMapping(const float bMonocular);

  // Main function
  void Run();

  void spinOnce();

  void InsertKeyFrame(KeyFrame *pKF);

  // Thread Synch
  void RequestStop();
  void RequestReset();
  bool Stop();
  void Release();
  bool isStopped();
  bool stopRequested();
  bool AcceptKeyFrames();
  void SetAcceptKeyFrames(bool flag);
  bool SetNotStop(bool flag);

  void InterruptBA();

  void RequestFinish();
  bool isFinished();

  int KeyframesInQueue() {
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    return mlNewKeyFrames.size();
  }

public:
  void CreateNewMapPointsOverride();

  void CreateNewMapPointsEpipolar();

  bool CheckNewKeyFrames();
  void ProcessNewKeyFrame();
  void CreateNewMapPoints();

  void MapPointCulling();
  void SearchInNeighbors();

  void KeyFrameCulling();
  void KeyFrameCullingOverride();
  void KeyFrameCullingOverride2();

  cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);

  cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

  bool mbMonocular;

  bool CheckReset();
  void ResetIfRequested();
  bool mbResetRequested;
  std::mutex mMutexReset;

  bool CheckFinish();
  void SetFinish();
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  std::list<KeyFrame *> mlNewKeyFrames;

  KeyFrame *mpCurrentKeyFrame;

  std::list<MapPoint *> mlpRecentAddedMapPoints;

  // ADD(conditional variable)
  std::condition_variable cv_new_kfs_;
  std::mutex mMutexNewKFs;

  bool mbAbortBA;

  bool mbStopped;
  bool mbStopRequested;
  bool mbNotStop;
  std::mutex mMutexStop;

  bool mbAcceptKeyFrames;
  std::mutex mMutexAccept;
};

} // namespace orbslam
