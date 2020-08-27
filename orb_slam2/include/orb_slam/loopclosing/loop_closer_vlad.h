#pragma once

#include <mutex>
#include <thread>

#include "g2o/types/sim3/types_seven_dof_expmap.h"

#include "../type/keyframe.h"

namespace orbslam {

class LoopClosingVLAD {
public:
  typedef std::pair<std::set<KeyFrame *>, int> ConsistentGroup;
  typedef std::map<
      KeyFrame *, g2o::Sim3, std::less<KeyFrame *>,
      Eigen::aligned_allocator<std::pair<const KeyFrame *, g2o::Sim3>>>
      KeyFrameAndPose;

public:
  LoopClosingVLAD(const bool bFixScale);

  void spinOnce();

  // Main function
  void Run();

  void InsertKeyFrame(KeyFrame *pKF);

  void RequestReset();

  std::vector<KeyFrame *> detectLoopCandidates(float minScore);

  // This function will run in a separate thread
  void RunGlobalBundleAdjustment(unsigned long nLoopKF);

  bool isRunningGBA() {
    std::unique_lock<std::mutex> lock(mMutexGBA);
    return mbRunningGBA;
  }
  bool isFinishedGBA() {
    std::unique_lock<std::mutex> lock(mMutexGBA);
    return mbFinishedGBA;
  }

  void RequestFinish();

  bool isFinished();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
  bool CheckNewKeyFrames();

  bool DetectLoop();

  bool detectLoopVLAD();

  bool ComputeSim3();

  void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

  void CorrectLoop();

  void ResetIfRequested();
  bool mbResetRequested;
  std::mutex mMutexReset;

  bool CheckFinish();
  void SetFinish();
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  std::list<KeyFrame *> mlpLoopKeyFrameQueue;

  std::mutex mMutexLoopQueue;

  // Loop detector parameters
  float mnCovisibilityConsistencyTh;

  // FIXME: vlad
  std::vector<cv::Mat> vlad_vec;

  // Loop detector variables
  KeyFrame *mpCurrentKF;
  KeyFrame *mpMatchedKF;
  std::vector<ConsistentGroup> mvConsistentGroups;
  std::vector<KeyFrame *> mvpEnoughConsistentCandidates;
  std::vector<KeyFrame *> mvpCurrentConnectedKFs;
  std::vector<MapPoint *> mvpCurrentMatchedPoints;
  std::vector<MapPoint *> mvpLoopMapPoints;
  cv::Mat mScw;
  g2o::Sim3 mg2oScw;

  long unsigned int mLastLoopKFid;

  // Variables related to Global Bundle Adjustment
  bool mbRunningGBA;
  bool mbFinishedGBA;
  bool mbStopGBA;
  std::mutex mMutexGBA;
  std::thread *mpThreadGBA;

  // Fix scale in the stereo/RGB-D case
  bool mbFixScale;

  size_t mnFullBAIdx;

  std::vector<KeyFrame *> db_frames;
  // bool ;
};

} // namespace orbslam
