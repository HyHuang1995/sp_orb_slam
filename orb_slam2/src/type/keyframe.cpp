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

#include "orb_slam/type/keyframe.h"

#include <mutex>
#include <random>

#include "orb_slam/cv/orb_matcher.h"
#include "orb_slam/cv/sp_matcher.h"

#include "orb_slam/config.h"
#include "orb_slam/global.h"

#include "orb_slam/utils/converter.h"

#include <boost/bind.hpp>

namespace orbslam {

using namespace std;

long unsigned int KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame &F)
    : global_desc(F.global_desc.clone()), occ_grid(F.occ_grid.clone()),
      flann(static_cast<cv::FlannBasedMatcher *>(nullptr)), mnFrameId(F.mnId),
      mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS),
      mnGridRows(FRAME_GRID_ROWS),
      mfGridElementWidthInv(F.mfGridElementWidthInv),
      mfGridElementHeightInv(F.mfGridElementHeightInv),
      mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0),
      mnBAFixedForKF(0), mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0),
      mnRelocWords(0), mnBAGlobalForKF(0), fx(F.fx), fy(F.fy), cx(F.cx),
      cy(F.cy), invfx(F.invfx), invfy(F.invfy), mbf(F.mbf), mb(F.mb),
      mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
      cov2_inv_(F.cov2_inv_), mvuRight(F.mvuRight), mvDepth(F.mvDepth),
      mDescriptors(F.mDescriptors.clone()), mnScaleLevels(F.mnScaleLevels),
      mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor),
      mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY),
      mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK(F.mK),
      mvpMapPoints(F.mvpMapPoints),
      // mpORBvocabulary(F.mpORBvocabulary),
      mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
      mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb / 2) {
  mnId = nNextId++;

  grid_cols = occ_grid.cols;
  grid_rows = occ_grid.rows;
  mK_grid = cv::Mat::eye(cv::Size(3, 3), CV_32FC1);
  mK_grid.at<float>(0, 0) = fx / 8.0f;
  mK_grid.at<float>(1, 1) = fy / 8.0f;
  mK_grid.at<float>(0, 2) = (cx - 3.5f) / 8.0f;
  mK_grid.at<float>(1, 2) = (cy - 3.5f) / 8.0f;
  // mGrid.resize(mnGridCols);
  // for (int i = 0; i < mnGridCols; i++) {
  //   mGrid[i].resize(mnGridRows);
  //   for (int j = 0; j < mnGridRows; j++)
  //     mGrid[i][j] = F.mGrid[i][j];
  // }

  seeds_ = vector<Seed::Ptr>(N, nullptr);

  static std::random_device rd;
  static std::mt19937 rng(rd()); // TODO:

  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  rgb_ = vector<cv::Point3f>(N);
  for (size_t i = 0; i < N; i++) {
    float r = uni(rng), g = uni(rng), b = uni(rng);
    rgb_[i].x = r;
    rgb_[i].y = g;
    rgb_[i].z = b;
  }

  is_mp_visible_ = vector<bool>(N, false);

  SetPose(F.mTcw);

  ComputeSceneMeanDepth(2);
  LOG(INFO) << "kf scene depth: median: " << scene_depth_median
            << " min: " << scene_depth_min << " max: " << scene_depth_max;
}

// ADD(depth filter)
int KeyFrame::initializeSeeds() {
  // TODO: maybe no need for lock ??
  unique_lock<mutex> lock(mMutexFeatures);
  for (size_t i = 0; i < N; i++) {
    const auto &mp = mvpMapPoints[i];
    if (!mp) {
      seeds_[i] =
          std::make_shared<Seed>(scene_depth_mean, scene_depth_min, this, i);
    }
  }
}

bool computeEpiDist(const cv::Point2f &kp1, const cv::Point2f &kp2,
                    const cv::Mat &F12, float &dist)
//  , const Frame *pKF, const int idx) {
{
  const float a = kp2.x * F12.at<float>(0, 0) + kp2.y * F12.at<float>(0, 1) +
                  F12.at<float>(0, 2);
  const float b = kp2.x * F12.at<float>(1, 0) + kp2.y * F12.at<float>(1, 1) +
                  F12.at<float>(1, 2);
  const float c = kp2.x * F12.at<float>(2, 0) + kp2.y * F12.at<float>(2, 1) +
                  F12.at<float>(2, 2);

  float num, factor;
  num = a * kp1.x + b * kp1.y + c;

  const float den = a * a + b * b;

  if (den == 0)
    return false;

  dist = num * num / den;

  return true;
}

bool computeDepthFromTriangulation(const cv::Mat R_th, const cv::Mat t_th,
                                   const Seed::Ptr seed, const cv::Mat &f_cur,
                                   float &depth) {
  cv::Mat A(3, 2, CV_32F);
  // Matrix<double, 3, 2> A;
  cv::Mat Rf = (R_th * seed->f);
  Rf.copyTo(A.col(0));
  f_cur.copyTo(A.col(1));
  // A.col(0).setTo();
  // A.col(1).setTo(f_cur);

  // cout <<  R_th * seed->f << endl;
  // cout << f_cur << endl;
  // cout << A << endl;

  cv::Mat AtA = A.t() * A;
  // AtA.det
  if (cv::determinant(AtA) < 0.000001f)
    return false;

  cv::Mat depth2 = -AtA.inv() * A.t() * t_th;
  depth = fabs(depth2.at<float>(0));

  return true;
}

float computeTau(const cv::Mat &t, const cv::Mat &f, const float z,
                 const float px_error_angle) {
  cv::Mat a = f * z - t;
  float t_norm = cv::norm(t);
  float a_norm = cv::norm(a);
  float alpha = acos(f.dot(t) / t_norm);            // dot product
  float beta = acos(a.dot(-t) / (t_norm * a_norm)); // dot product
  float beta_plus = beta + px_error_angle;
  float gamma_plus = M_PI - alpha - beta_plus; // triangle angles sum to PI
  float z_plus = t_norm * sin(beta_plus) / sin(gamma_plus); // law of sines
  return (z_plus - z);                                      // tau
}

int KeyFrame::updateSeeds(Frame *frame) {

  float px_noise = 1.0f;
  float px_error_angle =
      atan(px_noise / (2.0f * fx)) * 2.0f; // law of chord (sehnensatz)

  cv::Mat R1w = frame->mRcw;
  cv::Mat t1w = frame->mtcw;
  cv::Mat R2w = this->GetRotation();
  cv::Mat t2w = this->GetTranslation();

  cv::Mat R12 = R1w * R2w.t();
  cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

  cv::Mat t21 = -R2w * R1w.t() * t1w + t2w;

  auto skew = [](const cv::Mat &v) -> cv::Mat {
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0), -v.at<float>(1), v.at<float>(0),
            0);
  };

  cv::Mat t12x = skew(t12);
  const cv::Mat &K = this->mK;
  const cv::Mat &K2 = frame->mK;

  // x1^T F12 x2
  const cv::Mat F12 = K.t().inv() * t12x * R12 * K2.inv();

  cv::Mat grid_for_search(camera::height, camera::width, CV_8UC1);

  auto hash_two_int = [](const int a, const int b) { return a + b * 752; };

  // return K1.t().inv() * t12x * R12 * K2.inv();
  int n_seeds = 0, n_update = 0, n_found;
  for (size_t i = 0; i < N; i++) {
    auto &it = seeds_[i];
    if (it == nullptr)
      continue;

    n_seeds++;

    // set this value true when seeds updating should be interrupted
    // if(seeds_updating_halt_)
    //   return;

    // check if seed is not already too old
    // if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs) {
    //   it = seeds_.erase(it);
    //   continue;
    // }

    // TODO: check if point is visible in the current image
    // SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
    // const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f) );
    // if(xyz_f.z() < 0.0)  {
    //   ++it; // behind the camera
    //   continue;
    // }
    // if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
    //   ++it; // point does not project in image
    //   continue;
    // }

    // we are using inverse depth coordinates
    float z_inv_min = it->mu + 3 * sqrt(it->sigma2);
    float z_inv_max = max(it->mu - 3 * sqrt(it->sigma2), 0.00000001f);
    float z_min = 1.0f / z_inv_min, z_max = 1.0f / z_inv_max;
    // double z;

    auto &&uv = mvKeysUn[i].pt;
    auto pt = cv::Mat(cv::Point3f((uv.x - cx) / fx, (uv.y - cy) / fy, 1.0f));
    auto pt_min_kf = pt * z_min;
    auto pt_max_kf = pt * z_max;

    cv::Mat pt_min = R12 * pt_min_kf + t12;
    cv::Mat pt_max = R12 * pt_max_kf + t12;

    cv::Mat A = mK * (pt_min / pt_min.at<float>(2));
    cv::Mat B = mK * (pt_max / pt_max.at<float>(2));

    // cv::Mat A = mK_grid * (pt_min / pt_min.at<float>(2));
    // cv::Mat B = mK_grid * (pt_max / pt_max.at<float>(2));
    float xa = A.at<float>(0), xb = B.at<float>(0);
    float ya = A.at<float>(1), yb = B.at<float>(1);

    // Line iter: x, y
    std::map<int, cv::Point2i> cells_to_check;
    cv::LineIterator iter(grid_for_search, cv::Point2f(xa, ya),
                          cv::Point2f(xb, yb), 8);

    for (size_t i_iter = 0; i_iter < iter.count; i_iter++, iter++) {
      auto pos = cv::Point2f(iter.pos()) / 8.0f;
      // FIXME: cell check???
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

    float best_dist = 1.0f; // TODO: dist threshold??
    int16_t best_idx = -1;
    // TODO: exclude points with observation
    // TODO: create map point when converge???
    // TODO: refine epipolar line check ????
    for (auto &&cell_idx : cells_to_check) {
      // LOG(INFO) << "check cell: " << cell_idx.second;
      int16_t idx = frame->occ_grid.at<int16_t>(cell_idx.second);
      if (idx != -1) {

        auto &pt_frame = frame->mvKeysUn[idx].pt;
        // cv::Mat f_curr =
        //     (cv::Mat_<float>(3, 1) << (pt_frame.x - this->cx) * this->invfx,
        //      (pt_frame.y - this->cy) * this->invfy, 1.0);

        // STEP.1 check epipolar constraints
        // if (dist < best_dist)
        // {
        //   best_idx = idx;
        //   best_dist = dist;
        // }

        float epi_dist;
        if (!computeEpiDist(pt_frame, it->uv, F12, epi_dist)) {
          LOG(ERROR) << "compute epipolar dist error";
        } else {
          if (epi_dist > 4.0f)
            continue;
        }

        float dist = SPMatcher::DescriptorDistance(frame->mDescriptors.row(idx),
                                                   mDescriptors.row(i));
        // STEP.2 check epipolar constraints
        if (dist < best_dist) {
          best_idx = idx;
          best_dist = dist;
        }
      }
    }
    if (best_idx != -1) {
      n_found++;

      float depth_;
      auto &pt_frame = frame->mvKeysUn[best_idx].pt;
      cv::Mat f_curr =
          (cv::Mat_<float>(3, 1) << (pt_frame.x - this->cx) * this->invfx,
           (pt_frame.y - this->cy) * this->invfy, 1.0);
      if (!computeDepthFromTriangulation(R12, t12, it, f_curr, depth_)) {
        LOG(ERROR) << "compute depth error" << endl;
        continue;
      }

      float tau = computeTau(t21, it->f, depth_, px_error_angle);
      float tau_inv =
          0.5f * (1.0f / max(0.0000001f, depth_ - tau) - 1.0f / (depth_ + tau));

      it->updateSeed(1.0f / depth_, tau_inv * tau_inv);
      frame->seeds_[best_idx] = it;
      n_update++;
    }

    // if ((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) > 0.25)
    // {
    //   float k_epi = (yb - ya) / (xb - xa);
    //   float x0 = xa, y0 = ya;
    //   if (xa > xb) {
    //     std::swap(xa, xb);
    //     std::swap(ya, yb);
    //   }
    //   xa = floor(xa);
    //   xb = ceil(xb);
    //   ya = k_epi * (xa - x0) + y0 + 0.5;
    //   yb = k_epi * (xb - x0) + y0 + 0.5;
    // }
    // else
    // {
    //   xa = round(xa);
    //   xb = round(xb);
    //   ya = round(xa);
    //   yb = round(xb);
    // }

    // if (abs(xa - xb) > abs(ya - yb))
    // {

    // }

    // cv::LineIterator iter(frame->occ_grid, cv::Point2f(xa, ya),
    //                       cv::Point2f(xb, yb), 8);

    // for (size_t i_iter = 0; i_iter < iter.count; i_iter++, iter++) {
    //   int16_t idx_px = *(int16_t *)*iter;
    //   if (idx_px != -1) {
    //     n_found++;
    //   }
    // }

    // for (int start = 0; start != ;start++)
    // {

    // }
    // float xa = (pt_min.at<float>(0) / pt_min.at<float>(2) - cx) / fx,
    //       ya = (pt_min.at<float>(1) / pt_min.at<float>(2) - cy) / fy;
    // float xb = (pt_max.at<float>(0) / pt_max.at<float>(2) - cx) / fx,
    //       yb = (pt_max.at<float>(1) / pt_max.at<float>(2) - cy) / fy;

    // LOG(INFO) << "iter_count: " << cells_to_check.size();
    // LOG(INFO) << "n_found: " << n_found;
    // LOG(INFO) << "z_min: " << z_min << " z_max: " << z_max;
    // LOG(INFO) << "A: " << A.t() << " B: " << B.t();
    // LOG(INFO) << "xa: " << xa << " ya: " << ya << " xb: " << xb
    //           << " yb: " << yb;
  }

  LOG(INFO) << "updating: " << n_update << "/" << n_seeds
            << " n_found: " << n_found;
}

void KeyFrame::buildIndexesMps() {
  // flann instance
  cv::Ptr<cv::flann::KDTreeIndexParams> index_param =
      new cv::flann::KDTreeIndexParams(matching::ntree);
  cv::Ptr<cv::flann::SearchParams> search_param =
      new cv::flann::SearchParams(matching::nchecks);
  flann_mps = cv::Ptr<cv::FlannBasedMatcher>(
      new cv::FlannBasedMatcher(index_param, search_param));

  // gather unmatched features
  // cv::Mat desc_un;
  mDescMps = cv::Mat();
  mIndicesMps.clear();
  for (size_t i = 0; i < N; i++)
    if (mvpMapPoints[i]) {
      mDescMps.push_back(mDescriptors.row(i));
      mIndicesMps.push_back(i);
    }

  if (mDescriptors.rows > 0) {
    flann_mps->add(mDescMps);
    flann_mps->train();
  }
}

std::vector<cv::DMatch> KeyFrame::matchMps(Frame *frame) {
  // match by flann
  int nmatches = 0;
  // vector<bool> vbMatched2(pKF2->N, false);
  // vector<int> vMatches12(pKF1->N, -1);
  // vector<cv::DMatch> matches;
  // flann_mps->match(frame->mDescriptors, matches);

  // for (auto m : matches) {
  //   auto mp = mvpMapPoints[mIndicesMps[m.trainIdx]];
  //   if (mp) {
  //     frame->mvpMapPoints[m.queryIdx] = mp;
  //   }
  // }

  vector<vector<cv::DMatch>> matches;
  flann_mps->knnMatch(frame->mDescriptors, matches, 2);

  // ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  // int n_total = 0, n_rej_f = 0, n_rej_dist = 0, n_rej_epi = 0;

  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
      auto m = matches[i][0];
      auto mtmp = m;
      good_matches.push_back(mtmp);

      mtmp.trainIdx = mIndicesMps[m.trainIdx];

      auto mp = mvpMapPoints[mIndicesMps[m.trainIdx]];
      if (mp) {
        frame->mvpMapPoints[m.queryIdx] = mp;
      }
    }
  }

  return good_matches;
}

void KeyFrame::buildIndexes() {
  // flann instance
  cv::Ptr<cv::flann::KDTreeIndexParams> index_param =
      new cv::flann::KDTreeIndexParams(matching::ntree);
  cv::Ptr<cv::flann::SearchParams> search_param =
      new cv::flann::SearchParams(matching::nchecks);
  flann = cv::Ptr<cv::FlannBasedMatcher>(
      new cv::FlannBasedMatcher(index_param, search_param));

  // gather unmatched features
  // cv::Mat desc_un;
  mDescReamin = cv::Mat();
  mIndicesRemain.clear();
  for (size_t i = 0; i < N; i++)
    if (!mvpMapPoints[i]) {
      mDescReamin.push_back(mDescriptors.row(i));
      mIndicesRemain.push_back(i);
    }

  if (mDescriptors.rows > 0) {
    flann->add(mDescReamin);
    flann->train();
  }

  // cv::FlannBasedMatcher::create(index_param, search_param);
  // return;
}

// void KeyFrame::ComputeBoW()
// {
//   // if (mBowVec.empty() || mFeatVec.empty()) {
//   //   vector<cv::Mat> vCurrentDesc =
//   Converter::toDescriptorVector(mDescriptors);
//   //   // Feature vector associate features with nodes in the 4th level (from
//   //   // leaves up) We assume the vocabulary tree has 6 levels, change the 4
//   //   // otherwise
//   //   int levelsup = 4;
//   //   if (tracking::extractor_type == tracking::SP)
//   //     levelsup = 5;
//   //   mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, levelsup);
//   // }
// }

void KeyFrame::SetPose(const cv::Mat &Tcw_) {
  unique_lock<shared_mutex> lock(mMutexPose);
  Tcw_.copyTo(Tcw);
  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
  cv::Mat Rwc = Rcw.t();
  Ow = -Rwc * tcw;

  Twc = cv::Mat::eye(4, 4, Tcw.type());
  Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
  Ow.copyTo(Twc.rowRange(0, 3).col(3));

  cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);

  Cw = Twc * center;
}

cv::Mat KeyFrame::GetPose() {
  shared_lock<shared_mutex> lock(mMutexPose);
  return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse() {
  shared_lock<shared_mutex> lock(mMutexPose);
  return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter() {
  shared_lock<shared_mutex> lock(mMutexPose);
  return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter() {
  shared_lock<shared_mutex> lock(mMutexPose);
  return Cw.clone();
}

cv::Mat KeyFrame::GetRotation() {
  shared_lock<shared_mutex> lock(mMutexPose);
  return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}

cv::Mat KeyFrame::GetTranslation() {
  shared_lock<shared_mutex> lock(mMutexPose);
  return Tcw.rowRange(0, 3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight) {
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (!mConnectedKeyFrameWeights.count(pKF))
      mConnectedKeyFrameWeights[pKF] = weight;
    else if (mConnectedKeyFrameWeights[pKF] != weight)
      mConnectedKeyFrameWeights[pKF] = weight;
    else
      return;
  }

  UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles() {
  unique_lock<mutex> lock(mMutexConnections);
  // http://stackoverflow.com/questions/3389648/difference-between-stdliststdpair-and-stdmap-in-c-stl
  vector<pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(mConnectedKeyFrameWeights.size());

  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                      mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    vPairs.push_back(make_pair(mit->second, mit->first));

  sort(vPairs.begin(), vPairs.end());
  list<KeyFrame *> lKFs; // keyframe
  list<int> lWs;         // weight
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
  mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame *> KeyFrame::GetConnectedKeyFrames() {
  unique_lock<mutex> lock(mMutexConnections);
  set<KeyFrame *> s;
  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin();
       mit != mConnectedKeyFrameWeights.end(); mit++)
    s.insert(mit->first);
  return s;
}

vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames() {
  unique_lock<mutex> lock(mMutexConnections);
  return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) {
  unique_lock<mutex> lock(mMutexConnections);
  if ((int)mvpOrderedConnectedKeyFrames.size() < N)
    return mvpOrderedConnectedKeyFrames;
  else
    return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(),
                              mvpOrderedConnectedKeyFrames.begin() + N);
}

vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w) {
  unique_lock<mutex> lock(mMutexConnections);

  if (mvpOrderedConnectedKeyFrames.empty())
    return vector<KeyFrame *>();

  // http://www.cplusplus.com/reference/algorithm/upper_bound/
  vector<int>::iterator it =
      upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                  KeyFrame::weightComp);
  if (it == mvOrderedWeights.end() && *mvOrderedWeights.rbegin() < w)
    return vector<KeyFrame *>();
  else {
    int n = it - mvOrderedWeights.begin();
    return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(),
                              mvpOrderedConnectedKeyFrames.begin() + n);
  }
}

int KeyFrame::GetWeight(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF))
    return mConnectedKeyFrameWeights[pKF];
  else
    return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint *pMP) {
  int idx = pMP->GetIndexInKeyFrame(this);
  if (idx >= 0)
    mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP) {
  mvpMapPoints[idx] = pMP;
}

set<MapPoint *> KeyFrame::GetMapPoints() {
  unique_lock<mutex> lock(mMutexFeatures);
  set<MapPoint *> s;
  for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++) {
    if (!mvpMapPoints[i])
      continue;
    MapPoint *pMP = mvpMapPoints[i];
    if (!pMP->isBad())
      s.insert(pMP);
  }
  return s;
}

void KeyFrame::getTrackedInCommon(std::unordered_set<MapPoint *> curr_mps,
                                  int &num_obs, int &total_obs) {

  num_obs = 0;
  total_obs = 0;
  unique_lock<mutex> lock(mMutexFeatures);

  // const bool bCheckObs = minObs > 0;
  for (int i = 0; i < N; i++) {
    MapPoint *pMP = mvpMapPoints[i];
    if (pMP && !pMP->isBad()) {
      total_obs++;
      if (curr_mps.count(pMP)) {
        num_obs++;
      }

      // if (!pMP->isBad()) {
      //   if (bCheckObs) {
      //     if (mvpMapPoints[i]->Observations() >= minObs)
      //       nPoints++;
      //   } else
      //     nPoints++;
      // }
    }
  }

  // return nPoints;
}

int KeyFrame::TrackedMapPoints(const int &minObs) {
  unique_lock<mutex> lock(mMutexFeatures);

  int nPoints = 0;
  const bool bCheckObs = minObs > 0;
  for (int i = 0; i < N; i++) {
    MapPoint *pMP = mvpMapPoints[i];
    if (pMP) {
      if (!pMP->isBad()) {
        if (bCheckObs) {
          if (mvpMapPoints[i]->Observations() >= minObs)
            nPoints++;
        } else
          nPoints++;
      }
    }
  }

  return nPoints;
}

vector<MapPoint *> KeyFrame::GetMapPointMatches() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

MapPoint *KeyFrame::GetMapPoint(const size_t &idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections() {

  //===============1==================================
  map<KeyFrame *, int> KFcounter;

  vector<MapPoint *> vpMP;

  {
    unique_lock<mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  // For all map points in keyframe check in which other keyframes are they
  // seen Increase counter for those keyframes
  int n_total = 0, n_above_th = 0;
  for (vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;

    if (!pMP)
      continue;

    if (pMP->isBad())
      continue;

    n_total++;

    map<KeyFrame *, size_t> observations = pMP->GetObservations();

    for (map<KeyFrame *, size_t>::iterator mit = observations.begin(),
                                           mend = observations.end();
         mit != mend; mit++) {
      if (mit->first->mnId == mnId)
        continue;
      KFcounter[mit->first]++;
    }
  }

  // This should not happen
  if (KFcounter.empty())
    return;

  //===============2==================================
  // If the counter is greater than threshold add connection
  // In case no keyframe counter is over threshold add the one with maximum
  // counter
  int nmax = 0;
  KeyFrame *pKFmax = NULL;
  int th = 15; // FIXME:

  vector<pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (map<KeyFrame *, int>::iterator mit = KFcounter.begin(),
                                      mend = KFcounter.end();
       mit != mend; mit++) {
    if (mit->second > nmax) {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    if (mit->second >= th) {
      vPairs.push_back(make_pair(mit->second, mit->first));
      (mit->first)->AddConnection(this, mit->second);
    }
  }

  if (vPairs.empty()) {
    vPairs.push_back(make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
  }

  sort(vPairs.begin(), vPairs.end());
  list<KeyFrame *> lKFs;
  list<int> lWs;
  for (size_t i = 0; i < vPairs.size(); i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  //===============3==================================
  {
    unique_lock<mutex> lockCon(mMutexConnections);

    mConnectedKeyFrameWeights = KFcounter;
    mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

    if (mbFirstConnection && mnId != 0) {
      mpParent = mvpOrderedConnectedKeyFrames.front();
      mpParent->AddChild(this);
      mbFirstConnection = false;
    }
  }
}

void KeyFrame::AddChild(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mpParent = pKF;
  pKF->AddChild(this);
}

set<KeyFrame *> KeyFrame::GetChilds() {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mspChildrens;
}

KeyFrame *KeyFrame::GetParent() {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mbNotErase = true;
  mspLoopEdges.insert(pKF);
}

set<KeyFrame *> KeyFrame::GetLoopEdges() {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mspLoopEdges;
}

void KeyFrame::SetNotErase() {
  unique_lock<mutex> lock(mMutexConnections);
  mbNotErase = true;
}

void KeyFrame::SetErase() {
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mspLoopEdges.empty()) {
      mbNotErase = false;
    }
  }

  if (mbToBeErased) {
    SetBadFlag();
  }
}

void KeyFrame::SetBadFlag() {
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mnId == 0)
      return;
    else if (mbNotErase) {
      mbToBeErased = true;
      return;
    }
  }

  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                      mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    mit->first->EraseConnection(this);

  for (size_t i = 0; i < mvpMapPoints.size(); i++)
    if (mvpMapPoints[i])
      mvpMapPoints[i]->EraseObservation(this);

  {
    unique_lock<mutex> lock(mMutexConnections);
    unique_lock<mutex> lock1(mMutexFeatures);

    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    set<KeyFrame *> sParentCandidates;
    sParentCandidates.insert(mpParent);

    while (!mspChildrens.empty()) {
      bool bContinue = false;

      int max = -1;
      KeyFrame *pC;
      KeyFrame *pP;

      for (set<KeyFrame *>::iterator sit = mspChildrens.begin(),
                                     send = mspChildrens.end();
           sit != send; sit++) {
        KeyFrame *pKF = *sit;
        if (pKF->isBad())
          continue;

        // Check if a parent candidate is connected to the keyframe
        vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames();
        for (size_t i = 0, iend = vpConnected.size(); i < iend; i++) {
          for (set<KeyFrame *>::iterator spcit = sParentCandidates.begin(),
                                         spcend = sParentCandidates.end();
               spcit != spcend; spcit++) {
            if (vpConnected[i]->mnId == (*spcit)->mnId) {
              int w = pKF->GetWeight(vpConnected[i]);
              if (w > max) {
                pC = pKF;
                pP = vpConnected[i];
                max = w;
                bContinue = true;
              }
            }
          }
        }
      }

      if (bContinue) {
        pC->ChangeParent(pP);
        sParentCandidates.insert(pC);
        mspChildrens.erase(pC);
      } else
        break;
    }

    // If a children has no covisibility links with any parent candidate,
    // assign to the original parent of this KF
    if (!mspChildrens.empty())
      for (set<KeyFrame *>::iterator sit = mspChildrens.begin();
           sit != mspChildrens.end(); sit++) {
        (*sit)->ChangeParent(mpParent);
      }

    mpParent->EraseChild(this);
    mTcp = Tcw * mpParent->GetPoseInverse();
    mbBad = true;
  }

  global::map->EraseKeyFrame(this);
  // global::keyframe_db->erase(this);
}

bool KeyFrame::isBad() {
  unique_lock<mutex> lock(mMutexConnections);
  return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame *pKF) {
  bool bUpdate = false;
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF)) {
      mConnectedKeyFrameWeights.erase(pKF);
      bUpdate = true;
    }
  }

  if (bUpdate)
    UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y,
                                           const float &r) const {
  vector<size_t> vIndices;

  const int nMinCellX = max(0, (int)floor((x - mnMinX - r) / 8.0f));
  if (nMinCellX >= grid_cols)
    return vIndices;

  const int nMaxCellX = min(grid_cols - 1, (int)ceil((x - mnMinX + r) / 8.0f));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY = max(0, (int)floor((y - mnMinY - r) / 8.0f));
  if (nMinCellY >= grid_rows)
    return vIndices;

  const int nMaxCellY = min(grid_rows - 1, (int)ceil((y - mnMinY + r) / 8.0f));
  if (nMaxCellY < 0)
    return vIndices;

  vIndices.reserve(N);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      int16_t idx = occ_grid.at<int16_t>(iy, ix);
      if (idx == -1)
        continue;
      const cv::KeyPoint &kpUn = mvKeysUn[idx];

      const float distx = kpUn.pt.x - x;
      const float disty = kpUn.pt.y - y;

      if (fabs(distx) < r && fabs(disty) < r)
        vIndices.push_back(idx);
    }
  }

  return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const {
  return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i) {
  const float z = mvDepth[i];
  if (z > 0) {
    const float u = mvKeys[i].pt.x;
    const float v = mvKeys[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

    shared_lock<shared_mutex> lock(mMutexPose);
    return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3);
  } else
    return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q) {
  vector<MapPoint *> vpMapPoints;
  cv::Mat Tcw_;
  {
    unique_lock<mutex> lock(mMutexFeatures);
    shared_lock<shared_mutex> lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
  }

  vector<float> vDepths;
  vDepths.reserve(N);
  cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
  Rcw2 = Rcw2.t();
  float zcw = Tcw_.at<float>(2, 3);
  for (int i = 0; i < N; i++) {
    if (mvpMapPoints[i]) {
      MapPoint *pMP = mvpMapPoints[i];
      cv::Mat x3Dw = pMP->GetWorldPos();
      float z = Rcw2.dot(x3Dw) + zcw;
      vDepths.push_back(z);
    }
  }

  sort(vDepths.begin(), vDepths.end());
  scene_depth_median = vDepths[(vDepths.size() - 1) / q];
  scene_depth_min = vDepths[0];
  scene_depth_max = vDepths[vDepths.size() - 1];

  return scene_depth_median;
}

float KeyFrame::ComputeSceneMeanDepth(const int q) {
  vector<MapPoint *> vpMapPoints;
  cv::Mat Tcw_;
  {
    unique_lock<mutex> lock(mMutexFeatures);
    shared_lock<shared_mutex> lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
  }

  vector<float> vDepths;
  float z_sum = 0.0f, n_sum = 0.0f;
  vDepths.reserve(N);
  cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
  Rcw2 = Rcw2.t();
  float zcw = Tcw_.at<float>(2, 3);
  for (int i = 0; i < N; i++) {
    if (mvpMapPoints[i]) {
      MapPoint *pMP = mvpMapPoints[i];
      cv::Mat x3Dw = pMP->GetWorldPos();
      float z = Rcw2.dot(x3Dw) + zcw;
      vDepths.push_back(z);
      z_sum += z;
      n_sum += 1.0f;
    }
  }

  sort(vDepths.begin(), vDepths.end());
  scene_depth_median = vDepths[(vDepths.size() - 1) / q];
  scene_depth_mean = z_sum / n_sum;
  scene_depth_min = vDepths[0];
  scene_depth_max = vDepths[vDepths.size() - 1];

  return scene_depth_median;
}

} // namespace orbslam
