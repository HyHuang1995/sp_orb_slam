#include "orb_slam/cv/sp_matcher.h"

namespace orbslam {

using namespace std;

int SPMatcher::SearchBySim3Override(KeyFrame *pKF1, KeyFrame *pKF2,
                                    vector<MapPoint *> &vpMatches12,
                                    const float &s12, const cv::Mat &R12,
                                    const cv::Mat &t12, const float th) {
  const float &fx = pKF1->fx;
  const float &fy = pKF1->fy;
  const float &cx = pKF1->cx;
  const float &cy = pKF1->cy;

  // Camera 1 from world
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();

  // Camera 2 from world
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  // Transformation between cameras
  cv::Mat sR12 = s12 * R12;
  cv::Mat sR21 = (1.0 / s12) * R12.t();
  cv::Mat t21 = -sR21 * t12;

  const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
  const int N1 = vpMapPoints1.size();

  const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
  const int N2 = vpMapPoints2.size();

  vector<bool> vbAlreadyMatched1(N1, false);
  vector<bool> vbAlreadyMatched2(N2, false);

  for (int i = 0; i < N1; i++) {
    MapPoint *pMP = vpMatches12[i];
    if (pMP) {
      vbAlreadyMatched1[i] = true;
      int idx2 = pMP->GetIndexInKeyFrame(pKF2);
      if (idx2 >= 0 && idx2 < N2)
        vbAlreadyMatched2[idx2] = true;
    }
  }

  vector<int> vnMatch1(N1, -1);
  vector<int> vnMatch2(N2, -1);

  // Transform from KF1 to KF2 and search
  for (int i1 = 0; i1 < N1; i1++) {
    MapPoint *pMP = vpMapPoints1[i1];

    if (!pMP || vbAlreadyMatched1[i1])
      continue;

    if (pMP->isBad())
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc1 = R1w * p3Dw + t1w;
    cv::Mat p3Dc2 = sR21 * p3Dc1 + t21;

    // Depth must be positive
    if (p3Dc2.at<float>(2) < 0.0)
      continue;

    const float invz = 1.0 / p3Dc2.at<float>(2);
    const float x = p3Dc2.at<float>(0) * invz;
    const float y = p3Dc2.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF2->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = cv::norm(p3Dc2);

    // Depth must be inside the scale invariance region
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

    // Search in a radius
    const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = std::numeric_limits<float>::max();
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    if (bestDist <= 0.7) {
      vnMatch1[i1] = bestIdx;
    }
  }

  // Transform from KF2 to KF1 and search
  for (int i2 = 0; i2 < N2; i2++) {
    MapPoint *pMP = vpMapPoints2[i2];

    if (!pMP || vbAlreadyMatched2[i2])
      continue;

    if (pMP->isBad())
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc2 = R2w * p3Dw + t2w;
    cv::Mat p3Dc1 = sR12 * p3Dc2 + t12;

    // Depth must be positive
    if (p3Dc1.at<float>(2) < 0.0)
      continue;

    const float invz = 1.0 / p3Dc1.at<float>(2);
    const float x = p3Dc1.at<float>(0) * invz;
    const float y = p3Dc1.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    if (!pKF1->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = cv::norm(p3Dc1);

    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

    // Search in a radius of 2.5*sigma(ScaleLevel)
    const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = numeric_limits<float>::max();
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    if (bestDist <= TH_HIGH) {
      vnMatch2[i2] = bestIdx;
    }
  }

  // Check agreement
  int nFound = 0;

  for (int i1 = 0; i1 < N1; i1++) {
    int idx2 = vnMatch1[i1];

    if (idx2 >= 0) {
      int idx1 = vnMatch2[idx2];
      if (idx1 == i1) {
        vpMatches12[i1] = vpMapPoints2[idx2];
        nFound++;
      }
    }
  }

  return nFound;
}

int SPMatcher::SearchByProjectionLoop(KeyFrame *pKF, cv::Mat Scw,
                                      const vector<MapPoint *> &vpPoints,
                                      vector<MapPoint *> &vpMatched, int th) {
  // Get Calibration Parameters for later projection
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;

  cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  cv::Mat Rcw = sRcw / scw;
  cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
  cv::Mat Ow = -Rcw.t() * tcw;

  // Set of MapPoints already found in the KeyFrame
  set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
  spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

  int nmatches = 0;

  // For each Candidate MapPoint Project and Match
  for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++) {
    MapPoint *pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP))
      continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0)
      continue;

    // Project into Image
    const float invz = 1 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    // Depth must be inside the scale invariance region of the point
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = 256;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;
      if (vpMatched[idx])
        continue;

      const int &kpLevel = pKF->mvKeysUn[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    if (bestDist <= TH_HIGH) {
      vpMatched[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;
}

int SPMatcher::SearchByBruteForce(KeyFrame *pKF1, KeyFrame *pKF2,
                                  std::vector<MapPoint *> &vpMatches12) {
  const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
  const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
  const cv::Mat &Descriptors1 = pKF1->mDescriptors;

  const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
  const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
  const cv::Mat &Descriptors2 = pKF2->mDescriptors;
  vpMatches12 = std::vector<MapPoint *>(vpMapPoints1.size(), nullptr);

  cv::Mat desc_train, desc_query;
  std::vector<size_t> indices_train, indices_query;

  for (size_t i = 0; i < vpMapPoints1.size(); i++) {
    auto mp = vpMapPoints1[i];
    if (mp) {
      desc_train.push_back(Descriptors1.row(i));
      indices_train.push_back(i);
    }
  }
  for (size_t i = 0; i < vpMapPoints2.size(); i++) {
    auto mp = vpMapPoints2[i];
    if (mp) {
      desc_query.push_back(Descriptors2.row(i));
      indices_query.push_back(i);
    }
  }

  std::vector<cv::DMatch> matches;

  auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);
  matcher->add(desc_train);
  matcher->train();
  matcher->match(desc_query, matches);

  for (auto &&m : matches) {
    vpMatches12[indices_train[m.trainIdx]] =
        vpMapPoints2[indices_query[m.queryIdx]];
  }

  return matches.size();
}

} // namespace orbslam
