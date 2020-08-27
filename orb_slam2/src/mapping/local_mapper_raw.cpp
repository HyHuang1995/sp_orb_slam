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

void LocalMapping::CreateNewMapPoints() {
  LOG(INFO) << "creating new map points";

  // Retrieve neighbor keyframes in covisibility graph
  int nn = 10;
  if (mbMonocular)
    nn = 20;
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

      if (ratioBaselineDepth < 0.01)
        continue;
    }

    // Compute Fundamental Matrix
    cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

    // TODO: validate mappoint generated from depth info

    // Search matches that fullfil epipolar constraint
    vector<pair<size_t, size_t>> vMatchedIndices;
    matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12,
                                   vMatchedIndices, false);

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
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int &idx1 = vMatchedIndices[ikp].first;

      const int &idx2 = vMatchedIndices[ikp].second;

      const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
      const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
      bool bStereo1 = kp1_ur >= 0;

      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
      const float kp2_ur = pKF2->mvuRight[idx2];
      bool bStereo2 = kp2_ur >= 0;

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

      if (bStereo1)
        cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2,
                                           mpCurrentKeyFrame->mvDepth[idx1]));
      else if (bStereo2)
        cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

      cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

      cv::Mat x3D;
      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
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
      } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
        x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
      } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
        x3D = pKF2->UnprojectStereo(idx2);
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

        if (!bStereo1) {
          float u1 = fx1 * x1 * invz1 + cx1;
          float v1 = fy1 * y1 * invz1 + cy1;
          float errX1 = u1 - kp1.pt.x;
          float errY1 = v1 - kp1.pt.y;
          if ((errX1 * errX1 * sigma2_inv_x + errY1 * errY1 * sigma2_inv_y) >
              5.991)
            continue;
        } else {
          throw std::runtime_error("stereo not implementd");
        }
      }

      // Check reprojection error in second keyframe
      {
        const auto sigma = pKF2->cov2_inv_[idx2];
        const float sigma2_inv_x = sigma.x();
        const float sigma2_inv_y = sigma.y();

        const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
        const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
        const float invz2 = 1.0 / z2;
        if (!bStereo2) {
          float u2 = fx2 * x2 * invz2 + cx2;
          float v2 = fy2 * y2 * invz2 + cy2;
          float errX2 = u2 - kp2.pt.x;
          float errY2 = v2 - kp2.pt.y;
          if ((errX2 * errX2 * sigma2_inv_x + errY2 * errY2 * sigma2_inv_y) >
              5.991)
            continue;
        } else {
          throw std::runtime_error("stereo not implemented");
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

      pMP->UpdateNormalAndDepth();

      global::map->AddMapPoint(pMP);

      mlpRecentAddedMapPoints.push_back(pMP);

      nnew++;
    }
  }

  LOG(INFO) << "#matches: " << n_for_tri << " #nnew: " << nnew;
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
  return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
          v.at<float>(2), 0, -v.at<float>(0), -v.at<float>(1), v.at<float>(0),
          0);
}

bool LocalMapping::CheckReset() {
  unique_lock<mutex> lock(mMutexReset);
  return mbResetRequested;
}

void LocalMapping::RequestReset() {
  {
    unique_lock<mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  {
    unique_lock<mutex> lock(mMutexNewKFs);
    cv_new_kfs_.notify_one();
  }

  while (1) {
    {
      unique_lock<mutex> lock2(mMutexReset);
      if (!mbResetRequested)
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
}

void LocalMapping::ResetIfRequested() {
  unique_lock<mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlNewKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();
    mbResetRequested = false;
  }
}

void LocalMapping::RequestFinish() {
  {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
  }

  unique_lock<mutex> lock(mMutexNewKFs);
  cv_new_kfs_.notify_one();
}

bool LocalMapping::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void LocalMapping::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
  unique_lock<mutex> lock2(mMutexStop);
  mbStopped = true;
}

bool LocalMapping::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
  // Fundamental Matrix: inv(K1)*E*inv(K2)

  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  cv::Mat R12 = R1w * R2w.t();
  cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

  cv::Mat t12x = SkewSymmetricMatrix(t12);

  const cv::Mat &K1 = pKF1->mK;
  const cv::Mat &K2 = pKF2->mK;

  return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::RequestStop() {
  unique_lock<mutex> lock(mMutexStop);
  mbStopRequested = true;
  unique_lock<mutex> lock2(mMutexNewKFs);
  mbAbortBA = true;
}

bool LocalMapping::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  if (mbStopRequested && !mbNotStop) {
    mbStopped = true;

    return true;
  }

  return false;
}

bool LocalMapping::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

bool LocalMapping::stopRequested() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopRequested;
}

void LocalMapping::Release() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);
  if (mbFinished)
    return;
  mbStopped = false;
  mbStopRequested = false;
  for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(),
                                  lend = mlNewKeyFrames.end();
       lit != lend; lit++)
    delete *lit;
  mlNewKeyFrames.clear();

  cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames() {
  unique_lock<mutex> lock(mMutexAccept);
  return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
  unique_lock<mutex> lock(mMutexAccept);
  mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag) {
  unique_lock<mutex> lock(mMutexStop);

  if (flag && mbStopped)
    return false;

  mbNotStop = flag;

  return true;
}

void LocalMapping::InterruptBA() { mbAbortBA = true; }
} // namespace orbslam
