#include "orb_slam/global.h"

#include "orb_slam/common.h"
#include "orb_slam/config.h"
#include "orb_slam/system.h"
#include "orb_slam/utils/converter.h"

namespace orbslam {

using namespace std;

void System::resetSystem() {
  if (global::viewer && common::visualize) {
    global::viewer->RequestStop();
    while (!global::viewer->isStopped())
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  cout << "System Reseting..." << endl;

  // Reset Local Mapping
  cout << "Reseting Local Mapper...";
  if (common::online) {
    global::mapper->RequestReset();
  } else {
    global::mapper->ResetIfRequested();
  }
  cout << " done" << endl;

  // Reset Loop Closing
  // TODO:
  if (common::use_loop) {
    cout << "Reseting Loop Closing...";
    if (common::online) {
      global::looper->RequestReset();
    } else {
      global::looper->ResetIfRequested();
    }

    cout << " done" << endl;
  }

  // Clear Map (this erase MapPoints and KeyFrames)
  global::map->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;

  global::tracker->reset();

  if (global::viewer)
    global::viewer->Release();
}

// void System::Reset() {
//   unique_lock<mutex> lock(mMutexReset);
//   mbReset = true;
// }

void System::Shutdown() {
  cout << endl;
  global::tracker->report();
  cout << endl;

  cout << "mapper finish request... " << endl;
  global::mapper->RequestFinish();
  cout << "looper finish request... " << endl;
  global::looper->RequestFinish();
  cout << "viewer finish request... " << endl;
  if (global::viewer) {
    global::viewer->RequestFinish();
    while (!global::viewer->isFinished())
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Wait until all thread have effectively stopped
  cout << "viewer finished... " << endl;
  while (!global::mapper->isFinished()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  if (common::use_loop) {
    while (!global::looper->isFinished() || global::looper->isRunningGBA()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  cout << "looper && mapper finished... " << endl;
}

void System::SaveTrajectoryTUM(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

  vector<KeyFrame *> vpKFs = global::map->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // ofstream fcloud;
  // fcloud.open("/home/hyhuang/cloud.ply");
  // fcloud << fixed;
  // auto map = global::map->GetAllMapPoints();
  // for (auto &p : map) {
  //   if (p->isBad())
  //     continue;
  //   auto pos = p->GetWorldPos();
  //   fcloud << setprecision(9) << pos.at<float>(0) << ' ' << pos.at<float>(2)
  //          << ' ' << pos.at<float>(1) << endl;
  // }
  // fcloud.close();

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  list<orbslam::KeyFrame *>::iterator lRit =
      global::tracker->mlpReferences.begin();
  list<double>::iterator lT = global::tracker->mlFrameTimes.begin();
  list<bool>::iterator lbL = global::tracker->mlbLost.begin();
  for (list<cv::Mat>::iterator
           lit = global::tracker->mlRelativeFramePoses.begin(),
           lend = global::tracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++, lbL++) {
    if (*lbL)
      continue;

    KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    // If the reference keyframe was culled, traverse the spanning tree to get a
    // suitable keyframe.
    while (pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    vector<float> q = Converter::toQuaternion(Rwc);

    f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0)
      << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0]
      << " " << q[1] << " " << q[2] << " " << q[3] << endl;
  }
  f.close();
  cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
  cout << endl
       << "Saving keyframe trajectory to " << filename << " ..." << endl;

  vector<KeyFrame *> vpKFs = global::map->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  // cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame *pKF = vpKFs[i];

    // pKF->SetPose(pKF->GetPose()*Two);

    if (pKF->isBad())
      continue;

    cv::Mat R = pKF->GetRotation().t();
    vector<float> q = Converter::toQuaternion(R);
    cv::Mat t = pKF->GetCameraCenter();
    f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " "
      << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " "
      << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
  }

  f.close();
  cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryEuroc(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
  // if (mSensor == MONOCULAR) {
  //   cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." <<
  //   endl; return;
  // }

  vector<KeyFrame *> vpKFs = global::map->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  list<orbslam::KeyFrame *>::iterator lRit =
      global::tracker->mlpReferences.begin();
  list<double>::iterator lT = global::tracker->mlFrameTimes.begin();
  for (list<cv::Mat>::iterator
           lit = global::tracker->mlRelativeFramePoses.begin(),
           lend = global::tracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++) {
    orbslam::KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    while (pKF->isBad()) {
      cout << "bad parent" << endl;
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    /****FIXME ***/
    // cv::Mat Twc = Two_*Tcw.inv();
    /****FIXME ***/

    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    // cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
    // cv::Mat twc = Twc.rowRange(0, 3).col(3);
    // twc = Rwc * tsb + twc;
    // Rwc = Rwc * Rsb;

    f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1)
      << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " "
      << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " "
      << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " "
      << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " "
      << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
  }
  f.close();
  cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
  // if (mSensor == MONOCULAR) {
  //   cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." <<
  //   endl; return;
  // }

  vector<KeyFrame *> vpKFs = global::map->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  list<orbslam::KeyFrame *>::iterator lRit =
      global::tracker->mlpReferences.begin();
  list<double>::iterator lT = global::tracker->mlFrameTimes.begin();
  for (list<cv::Mat>::iterator
           lit = global::tracker->mlRelativeFramePoses.begin(),
           lend = global::tracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++) {
    orbslam::KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    while (pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1)
      << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " "
      << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " "
      << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " "
      << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " "
      << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
  }
  f.close();
  cout << endl << "trajectory saved!" << endl;
}

} // namespace orbslam
