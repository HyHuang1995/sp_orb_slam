#include "orb_slam/cv/sp_matcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <stdint.h>

#include "orb_slam/common.h"
#include "orb_slam/config.h"

using namespace std;

namespace orbslam {

// const int SPMathcer::TH_HIGH = 100;
const float SPMatcher::TH_HIGH = 0.7;
const float SPMatcher::TH_LOW = 0.3;
const int SPMatcher::HISTO_LENGTH = 30;

SPMatcher::SPMatcher(float nnratio) : mfNNratio(nnratio) {}

int SPMatcher::SearchForTriByEpi(
    KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
    std::vector<std::pair<size_t, size_t>> &vMatchedPairs) {
  auto hash_two_int = [](const int a, const int b) { return a + b * 1921; };
  // Compute epipole in second image
  cv::Mat R1w = pKF1->GetRotation();    // Rc2w
  cv::Mat Cw = pKF1->GetCameraCenter(); // twc1
  cv::Mat R2w = pKF2->GetRotation();    // Rc2w
  cv::Mat t2w = pKF2->GetTranslation(); // tc2w
  cv::Mat C2 = R2w * Cw + t2w;

  cv::Mat R21 = R2w * R1w.t();

  const float invz = 1.0f / C2.at<float>(2);
  const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
  const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;
  const float invfx = pKF1->invfx, invfy = pKF1->invfy;
  const float cx = pKF1->cx, cy = pKF1->cy;
  const float fx = pKF1->fx, fy = pKF1->fy;

  int n_total = 0, n_rej_f = 0, n_rej_dist = 0, n_rej_epi = 0;
  cv::Mat grid_for_search(camera::height, camera::width, CV_8UC1);
  vector<bool> vbMatched2(pKF2->N, false);
  vector<int> vMatches12(pKF1->N, -1);
  int nmatches = 0;
  for (size_t i = 0; i < pKF1->N; i++) {
    MapPoint *pMP1 = pKF1->GetMapPoint(i);
    if (pMP1)
      continue;

    const auto &kp1 = pKF1->mvKeysUn[i];

    cv::Mat pt_homo = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx) * invfx,
                       (kp1.pt.y - cy) * invfy, 1.0f);
    cv::Mat pt2 = R21 * pt_homo;
    float uvx = fx * (pt2.at<float>(0) / pt2.at<float>(2)) + cx;
    float uvy = fy * (pt2.at<float>(1) / pt2.at<float>(2)) + cy;

    uvx += (uvx - ex);
    uvy += (uvy - ey);

    std::map<int, cv::Point2i> cells_to_check;
    cv::LineIterator iter(grid_for_search, cv::Point2f(ex, ey),
                          cv::Point2f(uvx, uvy), 8);

    for (size_t i_iter = 0; i_iter < iter.count; i_iter++, iter++) {
      auto pos = cv::Point2f(iter.pos()) / 8.0f;
      cv::Point2i cell_idx = cv::Point2i(ceil(pos.x), ceil(pos.y));
      if (cell_idx.y < camera::height / 8 && cell_idx.x < camera::width / 8) {
        int cell_key = hash_two_int(cell_idx.x, cell_idx.y);
        if (!cells_to_check.count(cell_key)) {
          cells_to_check[cell_key] = cell_idx;
        }
      }
      cell_idx = cv::Point2i(ceil(pos.x), floor(pos.y));
      if (cell_idx.y < camera::height / 8 && cell_idx.x < camera::width / 8) {
        int cell_key = hash_two_int(cell_idx.x, cell_idx.y);
        if (!cells_to_check.count(cell_key)) {
          cells_to_check[cell_key] = cell_idx;
        }
      }
      cell_idx = cv::Point2i(floor(pos.x), ceil(pos.y));
      if (cell_idx.y < camera::height / 8 && cell_idx.x < camera::width / 8) {
        int cell_key = hash_two_int(cell_idx.x, cell_idx.y);
        if (!cells_to_check.count(cell_key)) {
          cells_to_check[cell_key] = cell_idx;
        }
      }
      cell_idx = cv::Point2i(floor(pos.x), floor(pos.y));
      if (cell_idx.y < camera::height / 8 && cell_idx.x < camera::width / 8) {
        int cell_key = hash_two_int(cell_idx.x, cell_idx.y);
        if (!cells_to_check.count(cell_key)) {
          cells_to_check[cell_key] = cell_idx;
        }
      }
    }

    const cv::Mat &d1 = pKF1->mDescriptors.row(i);
    float best_dist = 0.7f; // TODO: dist threshold??
    int16_t best_idx = -1;
    // TODO: exclude points with observation
    // TODO: create map point when converge???
    // TODO: refine epipolar line check ????
    // LOG(INFO) << "size of cells: " << cells_to_check.size();
    for (auto &&cell_idx : cells_to_check) {
      bool b1 = false, b2 = false, b3 = false;
      // LOG(INFO) << "check cell: " << cell_idx.second;
      int16_t idx = pKF2->occ_grid.at<int16_t>(cell_idx.second);
      if (idx != -1) {
        // cv::Scalar color;
        const MapPoint *mp2 = pKF2->GetMapPoint(idx);

        if (vbMatched2[idx] || mp2)
          continue;
        // if (mp2)
        //   continue;
        n_total++;

        const auto &kp2 = pKF2->mvKeysUn[idx];
        const float distex = ex - kp2.pt.x;
        const float distey = ey - kp2.pt.y;

        // TODO: sigma
        if (distex * distex + distey * distey <
            100 * pKF2->mvScaleFactors[kp2.octave]) {
          n_rej_epi++;
          continue;
          b1 = true;
        }

        if (!CheckDistEpipolarLine(kp1, kp2, F12, pKF2, idx)) {
          n_rej_f++;
          continue;
          b2 = true;
        }

        const cv::Mat &d2 = pKF2->mDescriptors.row(idx);
        float dist = SPMatcher::DescriptorDistance(d1, d2);

        // STEP.2 check epipolar constraints
        if (dist < best_dist) {
          best_idx = idx;
          best_dist = dist;
        }
      }
    }
    if (best_idx != -1) {

      nmatches++;
      vMatches12[i] = best_idx;
      vbMatched2[best_idx] = true;
    }
    // cv::Mat viz;
    // cv::hconcat(im1, im2, viz);
    // cv::imshow("epi", viz);
    // cv::waitKey(-1);

    // If we have already matched or there is a MapPoint skip
    // if (vbMatched2[idx2] || pMP2) continue;
  }
  vMatchedPairs.clear();
  vMatchedPairs.reserve(nmatches);

  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] < 0)
      continue;
    vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
  }

  // if (common::verbose) {
  //   LOG(INFO) << "nmatches: " << nmatches
  //             << " #match: " << vMatchedPairs.size();
  //   LOG(INFO) << "total: " << n_total << " rej_dist: " << n_rej_dist
  //             << "n_rej_epi: " << n_rej_epi << " n_rej_f " << n_rej_f;
  // }

  return nmatches;
}

int SPMatcher::SearchForTriByFlann(
    KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
    std::vector<std::pair<size_t, size_t>> &vMatchedPairs) {
  // Compute epipole in second image
  cv::Mat Cw = pKF1->GetCameraCenter(); // twc1
  cv::Mat R2w = pKF2->GetRotation();    // Rc2w
  cv::Mat t2w = pKF2->GetTranslation(); // tc2w
  cv::Mat C2 = R2w * Cw + t2w;
  const float invz = 1.0f / C2.at<float>(2);
  const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
  const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

  // match by flann
  int nmatches = 0;
  vector<bool> vbMatched2(pKF2->N, false);
  vector<int> vMatches12(pKF1->N, -1);
  vector<vector<cv::DMatch>> matches;
  pKF1->flann->knnMatch(pKF2->mDescReamin, matches, 2);

  // ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  int n_total = 0, n_rej_f = 0, n_rej_dist = 0, n_rej_epi = 0;
  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
      // good_matches.push_back(matches[i][0]);

      // find a good match
      size_t idx1 = pKF1->mIndicesRemain[matches[i][0].trainIdx];
      MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
      if (pMP1)
        continue;
      size_t idx2 = pKF2->mIndicesRemain[matches[i][0].queryIdx];
      MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

      // If we have already matched or there is a MapPoint skip
      if (vbMatched2[idx2] || pMP2)
        continue;

      const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

      n_total++;

      const float distex = ex - kp2.pt.x;
      const float distey = ey - kp2.pt.y;

      // TODO: sigma
      if (distex * distex + distey * distey <
          100 * pKF2->mvScaleFactors[kp2.octave]) {
        n_rej_epi++;
        continue;
      }

      if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2, idx2)) {
        vMatches12[idx1] = idx2;
        vbMatched2[idx2] = true;
        nmatches++;
      } else {
        n_rej_f++;
      }
    }
  }

  vMatchedPairs.clear();
  vMatchedPairs.reserve(nmatches);

  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] < 0)
      continue;
    vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
  }

  if (common::verbose) {
    LOG(INFO) << "total: " << n_total << " rej_dist: " << n_rej_dist
              << "n_rej_epi: " << n_rej_epi << " n_rej_f " << n_rej_f;
  }

  return nmatches;
}

int SPMatcher::SearchByFlann(
    KeyFrame *kf_db_ptr, KeyFrame *kf_qry_ptr,
    std::vector<std::pair<size_t, size_t>> &vMatchesPairs) {

  vector<vector<cv::DMatch>> matches;
  kf_db_ptr->flann->knnMatch(kf_qry_ptr->mDescReamin, matches, 2);

  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
      // good_matches.push_back(matches[i][0]);
      // vMatchesPairs.push_back();
    }
  }
}

int SearchByFlann(Frame &F, const std::vector<MapPoint *> &vpMapPoints,
                  const float th, const float th_dist) {
  // vector<vector<cv::DMatch>> matches;

  // t.start();
  // cv::Ptr<cv::flann::KDTreeIndexParams> index_param =
  //     new cv::flann::KDTreeIndexParams(2);
  // // auto search_param
  // cv::Ptr<cv::flann::SearchParams> search_param =
  //     new cv::flann::SearchParams(40);
  // cv::FlannBasedMatcher matcher(index_param, search_param);

  // Timer t_add;
  // t_add.start();
  // matcher.add(desc1);
  // cout << t_add.elapsedMilliseconds() << " ms for add" << endl;

  // Timer t_train;
  // t_train.start();
  // matcher.train();
  // cout << t_train.elapsedMilliseconds() << " ms for train" << endl;
  // // cv::BFMatcher matcher(cv::NORM_L2, true);

  // // auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);
  // // matches = matcher->knnMatch(desc0, desc1, 2);
  // Timer t_query;
  // t_query.start();
  // matcher.knnMatch(desc0, matches, 2);
  // cout << t_query.elapsedMilliseconds() << " ms for knn" << endl;

  // // matcher.match(desc0, desc1, matches);
  // cout << t.elapsedMilliseconds() << " ms for matching" << endl;
  // t.start();

  // // matcher->match(desc0, desc1, matches);

  // double max_dist = numeric_limits<double>::min();
  // double min_dist = numeric_limits<double>::max();

  // cout << matches.size() << endl;

  // // -- Quick calculation of max and min distances between keypoints
  // // for (int i = 0; i < desc0.rows; i++) {
  // //   double dist = matches[i].distance;
  // //   if (dist < min_dist)
  // //     min_dist = dist;
  // //   if (dist > max_dist)
  // //     max_dist = dist;
  // // }
  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  // //-- Filter matches using the Lowe's ratio test
  // const float ratio_thresh = 0.7f;
  // std::vector<cv::DMatch> good_matches;
  // for (size_t i = 0; i < matches.size(); i++) {
  //   if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
  //     good_matches.push_back(matches[i][0]);
  //   }
  // }
  // cout << "#good_matches " << good_matches.size() << endl;
}

int SPMatcher::SearchByProjection(Frame &F,
                                  const vector<MapPoint *> &vpMapPoints,
                                  const float th, const float th_dist) {
  int nmatches = 0;

  const bool bFactor = th != 1.0;

  for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    MapPoint *pMP = vpMapPoints[iMP];

    if (!pMP->mbTrackInView)
      continue;

    if (pMP->isBad())
      continue;

    const int &nPredictedLevel = pMP->mnTrackScaleLevel;

    // The size of the window will depend on the viewing direction
    float r = RadiusByViewingCos(pMP->mTrackViewCos);

    if (bFactor)
      r *= th;

    int lvl0 = -1, lvl1 = -1;
    if (tracking::scale_check) {
      lvl0 = nPredictedLevel - 1;
      lvl1 = nPredictedLevel;
    }
    const vector<size_t> vIndices =
        F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY,
                            r * F.mvScaleFactors[nPredictedLevel], lvl0, lvl1);

    if (vIndices.empty())
      continue;

    const cv::Mat MPdescriptor = pMP->getDescTrack();

    float bestDist = 256.0f;
    int bestLevel = -1;
    float bestDist2 = 256.0f;
    int bestLevel2 = -1;
    int bestIdx = -1;

    // Get best and second matches with near keypoints
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      if (F.mvpMapPoints[idx])
        if (F.mvpMapPoints[idx]->Observations() > 0)
          continue;

      const cv::Mat &d = F.mDescriptors.row(idx);

      const float dist = DescriptorDistance(MPdescriptor, d);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // Apply ratio to second match (only if best and second are in the same
    // scale level)
    if (bestDist <= th_dist) {

      F.mvpMapPoints[bestIdx] = pMP;
      nmatches++;
    } else {

      const float du = F.mvKeysUn[bestIdx].pt.x - pMP->mTrackProjX;
      const float dv = F.mvKeysUn[bestIdx].pt.y - pMP->mTrackProjY;
      const float duv = du * du + dv * dv;
      float thresh_dist = 0.7f;
      if (tracking::map::match_adaptive) {
        thresh_dist = 1.2f * tracking::dust::c2_thresh /
                      (tracking::dust::c2_thresh + duv);
      }
      if (bestDist < thresh_dist) {
        F.mvpMapPoints[bestIdx] = pMP;
        nmatches++;
      }
    }
  }

  return nmatches;
}

float SPMatcher::RadiusByViewingCos(const float &viewCos) {
  if (viewCos > 0.998)
    return 2.5;
  else
    return 4.0;
}

bool SPMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,
                                      const cv::KeyPoint &kp2,
                                      const cv::Mat &F12, const KeyFrame *pKF2,
                                      const int idx = -1) {
  const float a = kp1.pt.x * F12.at<float>(0, 0) +
                  kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
  const float b = kp1.pt.x * F12.at<float>(0, 1) +
                  kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
  const float c = kp1.pt.x * F12.at<float>(0, 2) +
                  kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

  float num, factor;

  const auto sigma = pKF2->cov2_inv_[idx];
  const float sigma2_inv_x = sigma.x();
  const float sigma2_inv_y = sigma.y();
  factor = 1.0f / min(sigma2_inv_x, sigma2_inv_y); // TODO: correctness??

  num = a * kp2.pt.x + b * kp2.pt.y + c;

  const float den = a * a + b * b;

  if (den == 0)
    return false;

  const float dsqr = num * num / den;

  return dsqr < 3.84 * factor;
}

// int SPMatcher::SearchByBoW(KeyFrame *pKF, Frame &F,
//                            vector<MapPoint *> &vpMapPointMatches) {
//   throw std::runtime_error("no implementation");
//   // const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

//   // vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint
//   *>(NULL));

//   // const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;

//   // int nmatches = 0;

//   // // We perform the matching over ORB that belong to the same vocabulary
//   node
//   // DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
//   // DBoW3::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
//   // DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
//   // DBoW3::FeatureVector::const_iterator Fend = F.mFeatVec.end();

//   // while (KFit != KFend && Fit != Fend) {
//   //   if (KFit->first == Fit->first) {
//   //     const vector<unsigned int> vIndicesKF = KFit->second;
//   //     const vector<unsigned int> vIndicesF = Fit->second;

//   //     for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
//   //       const unsigned int realIdxKF = vIndicesKF[iKF];

//   //       MapPoint *pMP = vpMapPointsKF[realIdxKF];

//   //       if (!pMP)
//   //         continue;

//   //       if (pMP->isBad())
//   //         continue;

//   //       const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

//   //       float bestDist1 = 256;
//   //       int bestIdxF = -1;
//   //       float bestDist2 = 256;

//   //       for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
//   //         const unsigned int realIdxF = vIndicesF[iF];

//   //         if (vpMapPointMatches[realIdxF])
//   //           continue;

//   //         const cv::Mat &dF = F.mDescriptors.row(realIdxF);

//   //         const float dist = DescriptorDistance(dKF, dF);

//   //         if (dist < bestDist1) {
//   //           bestDist2 = bestDist1;
//   //           bestDist1 = dist;
//   //           bestIdxF = realIdxF;
//   //         } else if (dist < bestDist2) {
//   //           bestDist2 = dist;
//   //         }
//   //       }

//   //       if (bestDist1 <= TH_LOW) {
//   //         if (static_cast<float>(bestDist1) <
//   //             mfNNratio * static_cast<float>(bestDist2)) {
//   //           vpMapPointMatches[bestIdxF] = pMP;

//   //           const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

//   //           nmatches++;
//   //         }
//   //       }
//   //     }

//   //     KFit++;
//   //     Fit++;
//   //   } else if (KFit->first < Fit->first) {
//   //     KFit = vFeatVecKF.lower_bound(Fit->first);
//   //   } else {
//   //     Fit = F.mFeatVec.lower_bound(KFit->first);
//   //   }
//   // }

//   // return nmatches;
// }

int SPMatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw,
                                  const vector<MapPoint *> &vpPoints,
                                  vector<MapPoint *> &vpMatched, int th) {
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

    if (bestDist <= TH_LOW) {
      vpMatched[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;
}

int SPMatcher::SearchForInitialization(Frame &F1, Frame &F2,
                                       vector<cv::Point2f> &vbPrevMatched,
                                       vector<int> &vnMatches12,
                                       int windowSize) {
  int nmatches = 0;
  vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

  vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
  vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

  for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++) {
    cv::KeyPoint kp1 = F1.mvKeysUn[i1];
    int level1 = kp1.octave;
    if (level1 > 0)
      continue;

    vector<size_t> vIndices2 = F2.GetFeaturesInArea(
        vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

    if (vIndices2.empty())
      continue;

    cv::Mat d1 = F1.mDescriptors.row(i1);

    float bestDist = std::numeric_limits<float>::max();
    float bestDist2 = std::numeric_limits<float>::max();
    int bestIdx2 = -1;

    for (vector<size_t>::iterator vit = vIndices2.begin();
         vit != vIndices2.end(); vit++) {
      size_t i2 = *vit;

      cv::Mat d2 = F2.mDescriptors.row(i2);

      float dist = DescriptorDistance(d1, d2);

      if (vMatchedDistance[i2] <= dist)
        continue;

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = i2;
      } else if (dist < bestDist2) {
        bestDist2 = dist;
      }
    }

    if (bestDist <= TH_LOW) {
      if (bestDist < (float)bestDist2 * mfNNratio) {
        if (vnMatches21[bestIdx2] >= 0) {
          vnMatches12[vnMatches21[bestIdx2]] = -1;
          nmatches--;
        }
        vnMatches12[i1] = bestIdx2;
        vnMatches21[bestIdx2] = i1;
        vMatchedDistance[bestIdx2] = bestDist;
        nmatches++;
      }
    }
  }

  // Update prev matched
  for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
    if (vnMatches12[i1] >= 0)
      vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

  return nmatches;
}

// int SPMatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2,
//                            vector<MapPoint *> &vpMatches12) {
//   const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
//   const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
//   const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
//   const cv::Mat &Descriptors1 = pKF1->mDescriptors;

//   const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
//   const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
//   const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
//   const cv::Mat &Descriptors2 = pKF2->mDescriptors;

//   vpMatches12 =
//       vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
//   vector<bool> vbMatched2(vpMapPoints2.size(), false);

//   vector<int> rotHist[HISTO_LENGTH];
//   for (int i = 0; i < HISTO_LENGTH; i++)
//     rotHist[i].reserve(500);

//   const float factor = HISTO_LENGTH / 360.0f;

//   int nmatches = 0;

//   DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
//   DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
//   DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
//   DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

//   while (f1it != f1end && f2it != f2end) {
//     if (f1it->first == f2it->first) {
//       for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
//         const size_t idx1 = f1it->second[i1];

//         MapPoint *pMP1 = vpMapPoints1[idx1];
//         if (!pMP1)
//           continue;
//         if (pMP1->isBad())
//           continue;

//         const cv::Mat &d1 = Descriptors1.row(idx1);

//         float bestDist1 = 256;
//         int bestIdx2 = -1;
//         float bestDist2 = 256;

//         for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
//           const size_t idx2 = f2it->second[i2];

//           MapPoint *pMP2 = vpMapPoints2[idx2];

//           if (vbMatched2[idx2] || !pMP2)
//             continue;

//           if (pMP2->isBad())
//             continue;

//           const cv::Mat &d2 = Descriptors2.row(idx2);

//           float dist = DescriptorDistance(d1, d2);

//           if (dist < bestDist1) {
//             bestDist2 = bestDist1;
//             bestDist1 = dist;
//             bestIdx2 = idx2;
//           } else if (dist < bestDist2) {
//             bestDist2 = dist;
//           }
//         }

//         if (bestDist1 < TH_LOW) {
//           if (static_cast<float>(bestDist1) <
//               mfNNratio * static_cast<float>(bestDist2)) {
//             vpMatches12[idx1] = vpMapPoints2[bestIdx2];
//             vbMatched2[bestIdx2] = true;

//             nmatches++;
//           }
//         }
//       }

//       f1it++;
//       f2it++;
//     } else if (f1it->first < f2it->first) {
//       f1it = vFeatVec1.lower_bound(f2it->first);
//     } else {
//       f2it = vFeatVec2.lower_bound(f1it->first);
//     }
//   }

//   return nmatches;
// }

int SPMatcher::SearchForTriangulation(
    KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
    vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo) {
  throw std::runtime_error("no implementation");
  // const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
  // const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

  // // Compute epipole in second image
  // cv::Mat Cw = pKF1->GetCameraCenter(); // twc1
  // cv::Mat R2w = pKF2->GetRotation();    // Rc2w
  // cv::Mat t2w = pKF2->GetTranslation(); // tc2w
  // cv::Mat C2 = R2w * Cw + t2w;
  // const float invz = 1.0f / C2.at<float>(2);
  // const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
  // const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

  // // Find matches between not tracked keypoints
  // // Matching speed-up by ORB Vocabulary
  // // Compare only ORB that share the same node

  // int nmatches = 0;
  // vector<bool> vbMatched2(pKF2->N, false);
  // vector<int> vMatches12(pKF1->N, -1);

  // DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
  // DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
  // DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
  // DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

  // int n_total = 0, n_rej_f = 0, n_rej_dist = 0, n_rej_epi = 0;
  // while (f1it != f1end && f2it != f2end) {
  //   if (f1it->first == f2it->first) {
  //     for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
  //       const size_t idx1 = f1it->second[i1];

  //       MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

  //       if (pMP1)
  //         continue;

  //       const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;

  //       if (bOnlyStereo)
  //         if (!bStereo1)
  //           continue;

  //       const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

  //       const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

  //       float bestDist = TH_HIGH; // FIXME: TH_LOW ===> HIGH
  //       int bestIdx2 = -1;

  //       for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
  //         size_t idx2 = f2it->second[i2];

  //         MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

  //         if (vbMatched2[idx2] || pMP2)
  //           continue;

  //         const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

  //         if (bOnlyStereo)
  //           if (!bStereo2)
  //             continue;

  //         n_total++;

  //         const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

  //         const float dist = DescriptorDistance(d1, d2);

  //         // FIXME:
  //         // if (dist > TH_LOW || dist > bestDist) {
  //         if (dist > bestDist) {
  //           n_rej_dist++;
  //           continue;
  //         }

  //         const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

  //         if (!bStereo1 && !bStereo2) {
  //           const float distex = ex - kp2.pt.x;
  //           const float distey = ey - kp2.pt.y;

  //           // TODO: sigma
  //           if (distex * distex + distey * distey <
  //               100 * pKF2->mvScaleFactors[kp2.octave]) {
  //             n_rej_epi++;
  //             continue;
  //           }
  //         }

  //         if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2, idx2)) {
  //           bestIdx2 = idx2;
  //           bestDist = dist;
  //         } else {
  //           n_rej_f++;
  //         }
  //       }

  //       if (bestIdx2 >= 0) {
  //         const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
  //         vMatches12[idx1] = bestIdx2;
  //         vbMatched2[bestIdx2] = true;
  //         nmatches++;
  //       }
  //     }

  //     f1it++;
  //     f2it++;
  //   } else if (f1it->first < f2it->first) {
  //     f1it = vFeatVec1.lower_bound(f2it->first);
  //   } else {
  //     f2it = vFeatVec2.lower_bound(f1it->first);
  //   }
  // }

  // vMatchedPairs.clear();
  // vMatchedPairs.reserve(nmatches);

  // for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
  //   if (vMatches12[i] < 0)
  //     continue;
  //   vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
  // }

  // if (common::verbose) {
  //   LOG(INFO) << "total: " << n_total << " rej_dist: " << n_rej_dist
  //             << "n_rej_epi: " << n_rej_epi << " n_rej_f " << n_rej_f;
  // }

  // return nmatches;
}

int SPMatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints,
                    const float th) {
  cv::Mat Rcw = pKF->GetRotation();
  cv::Mat tcw = pKF->GetTranslation();

  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;
  const float &bf = pKF->mbf;

  cv::Mat Ow = pKF->GetCameraCenter();

  int nFused = 0;

  const int nMPs = vpMapPoints.size();

  for (int i = 0; i < nMPs; i++) {
    MapPoint *pMP = vpMapPoints[i];

    if (!pMP)
      continue;

    if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    const float invz = 1 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    const float ur = u - bf * invz;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = 256;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

      const int &kpLevel = kp.octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      if (pKF->mvuRight[idx] >= 0) {
        // Check reprojection error in stereo
        const float &kpx = kp.pt.x;
        const float &kpy = kp.pt.y;
        const float &kpr = pKF->mvuRight[idx];
        const float ex = u - kpx;
        const float ey = v - kpy;
        const float er = ur - kpr;
        const float e2 = ex * ex + ey * ey + er * er;

        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
          continue;
      } else {
        const float &kpx = kp.pt.x;
        const float &kpy = kp.pt.y;
        const float ex = u - kpx;
        const float ey = v - kpy;
        const float e2 = ex * ex + ey * ey;

        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
          continue;
      }

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad()) {
          if (pMPinKF->Observations() > pMP->Observations())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int SPMatcher::Fuse(KeyFrame *pKF, cv::Mat Scw,
                    const vector<MapPoint *> &vpPoints, float th,
                    vector<MapPoint *> &vpReplacePoint) {
  // Get Calibration Parameters for later projection
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;

  // Decompose Scw
  cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  cv::Mat Rcw = sRcw / scw;
  cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
  cv::Mat Ow = -Rcw.t() * tcw;

  // Set of MapPoints already found in the KeyFrame
  const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

  int nFused = 0;

  const int nPoints = vpPoints.size();

  // For each candidate MapPoint project and match
  for (int iMP = 0; iMP < nPoints; iMP++) {
    MapPoint *pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP))
      continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    // Project into Image
    const float invz = 1.0 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    // Compute predicted scale level
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = std::numeric_limits<float>::max();
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin();
         vit != vIndices.end(); vit++) {
      const size_t idx = *vit;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_HIGH) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad())
          vpReplacePoint[iMP] = pMPinKF;
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int SPMatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2,
                            vector<MapPoint *> &vpMatches12, const float &s12,
                            const cv::Mat &R12, const cv::Mat &t12,
                            const float th) {
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

    if (bestDist <= TH_HIGH) {
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

    // Point must be inside the image
    if (!pKF1->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = cv::norm(p3Dc1);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Compute predicted octave
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

int SPMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                                  const float th, const bool bMono) {
  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

  const cv::Mat twc = -Rcw.t() * tcw;

  const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3); // tlw(l)

  // vector from LastFrame to CurrentFrame expressed in LastFrame
  const cv::Mat tlc = Rlw * twc + tlw;

  const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
  const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;

  for (int i = 0; i < LastFrame.N; i++) {
    MapPoint *pMP = LastFrame.mvpMapPoints[i];

    if (pMP) {
      if (!LastFrame.mvbOutlier[i]) {
        // Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        if (invzc < 0)
          continue;

        float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        int nLastOctave = LastFrame.mvKeys[i].octave;

        // Search in a window. Size depends on scale
        float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

        vector<size_t> vIndices2;

        if (tracking::scale_check) {
          if (bForward)
            vIndices2 =
                CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
          else if (bBackward)
            vIndices2 =
                CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
          else
            vIndices2 = CurrentFrame.GetFeaturesInArea(
                u, v, radius, nLastOctave - 1, nLastOctave + 1);
        } else {
          vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius);
        }

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->getDescTrack();

        float bestDist = std::numeric_limits<float>::max();
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                            vend = vIndices2.end();
             vit != vend; vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
              continue;

          if (CurrentFrame.mvuRight[i2] > 0) {
            const float ur = u - CurrentFrame.mbf * invzc;
            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
            if (er > radius)
              continue;
          }

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const float dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= TH_HIGH) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;
        }
      }
    }
  }
  return nmatches;
}

int SPMatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF,
                                  const set<MapPoint *> &sAlreadyFound,
                                  const float th, const int ORBdist) {
  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);
  const cv::Mat Ow = -Rcw.t() * tcw;

  const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint *pMP = vpMPs[i];

    if (pMP) {
      if (!pMP->isBad() && !sAlreadyFound.count(pMP)) {
        // Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        // Compute predicted scale level
        cv::Mat PO = x3Dw - Ow;
        float dist3D = cv::norm(PO);

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
          continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        // Search in a window
        const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

        int lvl0 = -1, lvl1 = -1;
        if (tracking::scale_check) {
          lvl0 = nPredictedLevel - 1;
          lvl1 = nPredictedLevel + 1;
        }
        const vector<size_t> vIndices2 =
            CurrentFrame.GetFeaturesInArea(u, v, radius, lvl0, lvl1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->getDescTrack();

        float bestDist = 256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin();
             vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const float dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= ORBdist) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;
        }
      }
    }
  }

  return nmatches;
}

float SPMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  float dist = (float)cv::norm(a, b, cv::NORM_L2);

  return dist;
}

int SPMatcher::SearchByBruteForce(KeyFrame *pKF1, Frame &pKF2,
                                  std::vector<MapPoint *> &vpMatches12) {
  const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
  const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
  const cv::Mat &Descriptors1 = pKF1->mDescriptors;

  const vector<cv::KeyPoint> &vKeysUn2 = pKF2.mvKeysUn;
  vpMatches12 = std::vector<MapPoint *>(pKF2.N, nullptr);

  cv::Mat desc_train, desc_query;
  std::vector<size_t> indices_train, indices_query;

  for (size_t i = 0; i < vpMapPoints1.size(); i++) {
    auto mp = vpMapPoints1[i];
    if (mp && !mp->isBad()) {
      desc_train.push_back(Descriptors1.row(i));
      indices_train.push_back(i);
    }
  }

  desc_query = pKF2.mDescriptors;

  std::vector<cv::DMatch> matches;

  auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);
  matcher->add(desc_train);
  matcher->train();
  matcher->match(desc_query, matches);

  for (auto &&m : matches) {
    vpMatches12[m.queryIdx] = vpMapPoints1[indices_train[m.trainIdx]];
  }
}

} // namespace orbslam
