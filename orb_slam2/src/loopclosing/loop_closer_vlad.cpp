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

#include "orb_slam/loopclosing/loop_closer_vlad.h"

#include <mutex>
#include <thread>

#include "orb_slam/global.h"

#include "orb_slam/mapping/optimizer.h"
#include "orb_slam/mapping/sim3_solver.h"

#include "orb_slam/cv/orb_matcher.h"
#include "orb_slam/cv/sp_matcher.h"

#include "orb_slam/utils/converter.h"

#include "orb_slam/utils/timing.h"

namespace orbslam {

using namespace std;

std::vector<KeyFrame *> LoopClosingVLAD::detectLoopCandidates(float minScore) {

  set<KeyFrame *> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

  list<pair<float, KeyFrame *>> lScoreAndMatch;

  cv::Mat &curr_desc = mpCurrentKF->global_desc;
  // Only compare against those keyframes that share enough words
  for (auto &kf : db_frames) {
    if (!spConnectedKeyFrames.count(kf)) {
      float score = curr_desc.dot(kf->global_desc);
      if (score > minScore) {
        // vpCandidateKFs.push_back(kf);
        lScoreAndMatch.push_back(make_pair(score, kf));
        kf->mnLoopQuery = mpCurrentKF->mnId;
        kf->mLoopScore = score;
      }
    }
  }

  int nscores = 0;

  if (lScoreAndMatch.empty())
    return std::vector<KeyFrame *>();

  list<pair<float, KeyFrame *>> lAccScoreAndMatch;
  float bestAccScore = minScore;

  // Lets now accumulate score by covisibility
  for (list<pair<float, KeyFrame *>>::iterator it = lScoreAndMatch.begin(),
                                               itend = lScoreAndMatch.end();
       it != itend; it++) {
    KeyFrame *pKFi = it->second;
    vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = it->first;
    KeyFrame *pBestKF = pKFi;
    for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(),
                                      vend = vpNeighs.end();
         vit != vend; vit++) {
      KeyFrame *pKF2 = *vit;
      if (pKF2->mnLoopQuery == mpCurrentKF->mnId) {
        accScore += pKF2->mLoopScore;
        if (pKF2->mLoopScore > bestScore) {
          pBestKF = pKF2;
          bestScore = pKF2->mLoopScore;
        }
      }
    }

    lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
    if (accScore > bestAccScore)
      bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;

  set<KeyFrame *> spAlreadyAddedKF;
  vector<KeyFrame *> vpLoopCandidates;
  vpLoopCandidates.reserve(lAccScoreAndMatch.size());

  for (list<pair<float, KeyFrame *>>::iterator it = lAccScoreAndMatch.begin(),
                                               itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    if (it->first > minScoreToRetain) {
      KeyFrame *pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpLoopCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpLoopCandidates;
}

bool LoopClosingVLAD::detectLoopVLAD() {
  {
    unique_lock<mutex> lock(mMutexLoopQueue);
    mpCurrentKF = mlpLoopKeyFrameQueue.front();
    mlpLoopKeyFrameQueue.pop_front();
    // Avoid that a keyframe can be erased while it is being process by this
    // thread
    mpCurrentKF->SetNotErase();
  }

  // If the map contains less than 10 KF or less than 10 KF have passed from
  // last loop detection
  if (mpCurrentKF->mnId < mLastLoopKFid + 10) {
    // global::keyframe_db->add(mpCurrentKF);
    mpCurrentKF->SetErase();
    return false;
  }

  vector<KeyFrame *> vpCandidateKFs;
  {
    vector<pair<float, KeyFrame *>> score_candidates;
    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility
    // graph We will impose loop candidates to have a higher similarity than
    // this VIII-A
    const vector<KeyFrame *> vpConnectedKeyFrames =
        mpCurrentKF->GetVectorCovisibleKeyFrames();
    const cv::Mat &curr_desc = mpCurrentKF->global_desc;
    const auto &set_cov_kfs = mpCurrentKF->GetConnectedKeyFrames();

    float minScore = 0.2;
    for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++) {
      KeyFrame *pKF = vpConnectedKeyFrames[i];
      if (pKF->isBad())
        continue;
      const cv::Mat &cov_desc = pKF->global_desc;

      float score = curr_desc.dot(pKF->global_desc);

      if (score < minScore)
        minScore = score;
    }

    // Query the database imposing the minimum score

    minScore = std::max(minScore, 0.2f);

    vpCandidateKFs = detectLoopCandidates(minScore);

    LOG(INFO) << "min score: " << minScore << " norm: " << cv::norm(curr_desc);

    // If there are no loop candidates, just add new keyframe and return false
    if (vpCandidateKFs.empty()) {
      db_frames.push_back(mpCurrentKF);
      mvConsistentGroups.clear();
      mpCurrentKF->SetErase();
      return false;
    }
  }

  LOG(INFO) << "size vpcandidates: " << vpCandidateKFs.size();

  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected to the
  // loop candidate in the covisibility graph) A group is consistent with a
  // previous group if they share at least a keyframe We must detect a
  // consistent loop in several consecutive keyframes to accept it
  mvpEnoughConsistentCandidates.clear();

  vector<ConsistentGroup> vCurrentConsistentGroups;
  vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
  for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
    KeyFrame *pCandidateKF = vpCandidateKFs[i];

    set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
    spCandidateGroup.insert(pCandidateKF);

    bool bEnoughConsistent = false;
    bool bConsistentForSomeGroup = false;

    for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++) {
      set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;

      bool bConsistent = false;
      for (set<KeyFrame *>::iterator sit = spCandidateGroup.begin(),
                                     send = spCandidateGroup.end();
           sit != send; sit++) {
        if (sPreviousGroup.count(*sit)) {
          bConsistent = true;
          bConsistentForSomeGroup = true;
          break;
        }
      }

      if (bConsistent) {
        int nPreviousConsistency = mvConsistentGroups[iG].second;
        int nCurrentConsistency = nPreviousConsistency + 1;
        if (!vbConsistentGroup[iG]) {
          ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
          vCurrentConsistentGroups.push_back(cg);
          vbConsistentGroup[iG] = true;
        }
        if (nCurrentConsistency >= mnCovisibilityConsistencyTh &&
            !bEnoughConsistent) {
          mvpEnoughConsistentCandidates.push_back(pCandidateKF);
          bEnoughConsistent = true;
        }
      }
    }

    // If the group is not consistent with any previous group insert with
    // consistency counter set to zero
    if (!bConsistentForSomeGroup) {
      vCurrentConsistentGroups.clear();

      ConsistentGroup cg = make_pair(spCandidateGroup, 0);
      vCurrentConsistentGroups.push_back(cg);
    }
  }

  // Update Covisibility Consistent Groups
  mvConsistentGroups = vCurrentConsistentGroups;

  // Add Current Keyframe to database
  db_frames.push_back(mpCurrentKF);

  if (mvpEnoughConsistentCandidates.empty()) {
    mpCurrentKF->SetErase();
    return false;
  } else {
    return true;
  }

  mpCurrentKF->SetErase();
  return false;
}

void LoopClosingVLAD::spinOnce() {
  LOG(INFO) << "loopclosing";
  if (CheckNewKeyFrames()) {

    timing::Timer detect_timer("loop/init_detect");
    auto detect_res = detectLoopVLAD();
    detect_timer.Stop();

    // Detect loop candidates and check covisibility consistency
    if (detect_res) {
      LOG(INFO) << "# candidates detected..." << endl;

      timing::Timer correct_timer("loop/correction");
      if (ComputeSim3()) {

        LOG(INFO) << "## compute sim3 successfully..." << endl;
        // Perform loop fusion and pose graph optimization
        CorrectLoop();
      }
      correct_timer.Stop();
    }
  }
}

LoopClosingVLAD::LoopClosingVLAD(const bool bFixScale)
    : mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
      mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false),
      mbFinishedGBA(true), mbStopGBA(false), mpThreadGBA(NULL),
      mbFixScale(bFixScale), mnFullBAIdx(0) {
  mnCovisibilityConsistencyTh = 3;
}

void LoopClosingVLAD::Run() {
  mbFinished = false;

  while (1) {
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      LOG(INFO) << "loop closing";

      // Detect loop candidates and check covisibility consistency
      timing::Timer timer_detect("loop/detection");
      auto res = DetectLoop();
      timer_detect.Stop();
      if (res) {
        // Compute similarity transformation [sR|t]
        // In the stereo/RGBD case s=1
        // continue;
        timing::Timer correct_timer("loop/correction");
        if (ComputeSim3()) {

          // Perform loop fusion and pose graph optimization
          CorrectLoop();
        }
        correct_timer.Stop();
      }
    }

    ResetIfRequested();

    if (CheckFinish())
      break;

    // usleep(5000);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  SetFinish();
}

void LoopClosingVLAD::InsertKeyFrame(KeyFrame *pKF) {
  // cout << "a" << endl;
  unique_lock<mutex> lock(mMutexLoopQueue);
  // cout << "b" << endl;
  if (pKF->mnId != 0)
    mlpLoopKeyFrameQueue.push_back(pKF);
  // cout << "c" << endl;
}

bool LoopClosingVLAD::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mMutexLoopQueue);
  return (!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosingVLAD::DetectLoop() {
  throw std::runtime_error("no implementation");
}

bool LoopClosingVLAD::ComputeSim3() {
  // For each consistent loop candidate we try to compute a Sim3
  const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

  LOG(INFO) << "compute sim3: # init candidates: " << nInitialCandidates;

  // We compute first ORB matches for each candidate
  // If enough matches are found, we setup a Sim3Solver
  // ORBmatcher matcher(0.75, true);
  SPMatcher matcher(0.75);

  vector<Sim3Solver *> vpSim3Solvers;
  vpSim3Solvers.resize(nInitialCandidates);

  vector<vector<MapPoint *>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nInitialCandidates);

  vector<bool> vbDiscarded;
  vbDiscarded.resize(nInitialCandidates);

  int nCandidates = 0; // candidates with enough matches

  for (int i = 0; i < nInitialCandidates; i++) {
    KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

    pKF->SetNotErase();

    if (pKF->isBad()) {
      vbDiscarded[i] = true;
      continue;
    }

    int nmatches =
        matcher.SearchByBruteForce(mpCurrentKF, pKF, vvpMapPointMatches[i]);

    if (nmatches < 20) {
      vbDiscarded[i] = true;
      continue;
    } else {
      Sim3Solver *pSolver =
          new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
      pSolver->SetRansacParameters(0.99, 20, 300);
      vpSim3Solvers[i] = pSolver;
    }

    nCandidates++;
  }

  bool bMatch = false;

  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nInitialCandidates; i++) {
      if (vbDiscarded[i])
        continue;

      KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      Sim3Solver *pSolver = vpSim3Solvers[i];
      cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // If RANSAC returns a Sim3, perform a guided matching and optimize with
      // all correspondences
      if (!Scm.empty()) {
        vector<MapPoint *> vpMapPointMatches(vvpMapPointMatches[i].size(),
                                             static_cast<MapPoint *>(NULL));
        for (size_t j = 0, jend = vbInliers.size(); j < jend; j++) {
          if (vbInliers[j])
            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
        }

        cv::Mat R = pSolver->GetEstimatedRotation();
        cv::Mat t = pSolver->GetEstimatedTranslation();
        const float s = pSolver->GetEstimatedScale();
        matcher.SearchBySim3Override(mpCurrentKF, pKF, vpMapPointMatches, s, R,
                                     t, 7.5);

        g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
        const int nInliers = Optimizer::OptimizeSim3(
            mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);
        LOG(INFO) << "candidate id: " << pKF->mnFrameId
                  << " ninliers: " << nInliers;
        if (nInliers >= 20) {
          bMatch = true;
          mpMatchedKF = pKF;
          g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),
                         Converter::toVector3d(pKF->GetTranslation()), 1.0);
          mg2oScw = gScm * gSmw;
          mScw = Converter::toCvMat(mg2oScw);

          mvpCurrentMatchedPoints = vpMapPointMatches;
          break;
        }
      }
    }
  }

  if (!bMatch) {
    for (int i = 0; i < nInitialCandidates; i++)
      mvpEnoughConsistentCandidates[i]->SetErase();
    mpCurrentKF->SetErase();
    return false;
  }

  // Retrieve MapPoints seen in Loop Keyframe and neighbors
  vector<KeyFrame *> vpLoopConnectedKFs =
      mpMatchedKF->GetVectorCovisibleKeyFrames();
  vpLoopConnectedKFs.push_back(mpMatchedKF);
  mvpLoopMapPoints.clear();
  for (vector<KeyFrame *>::iterator vit = vpLoopConnectedKFs.begin();
       vit != vpLoopConnectedKFs.end(); vit++) {
    KeyFrame *pKF = *vit;
    vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId) {
          mvpLoopMapPoints.push_back(pMP);
          pMP->mnLoopPointForKF = mpCurrentKF->mnId;
        }
      }
    }
  }

  // Find more matches projecting with the computed Sim3
  matcher.SearchByProjectionLoop(mpCurrentKF, mScw, mvpLoopMapPoints,
                                 mvpCurrentMatchedPoints, 10);

  // If enough matches accept Loop
  int nTotalMatches = 0;
  for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) {
    if (mvpCurrentMatchedPoints[i])
      nTotalMatches++;
  }

  LOG(INFO) << "final search #: " << nTotalMatches;

  if (nTotalMatches >= 40) {
    for (int i = 0; i < nInitialCandidates; i++)
      if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
        mvpEnoughConsistentCandidates[i]->SetErase();
    return true;
  } else {
    for (int i = 0; i < nInitialCandidates; i++)
      mvpEnoughConsistentCandidates[i]->SetErase();
    mpCurrentKF->SetErase();
    return false;
  }
}

void LoopClosingVLAD::CorrectLoop() {
  LOG(WARNING) << "Loop detected!";

  // Send a stop signal to Local Mapping
  // Avoid new keyframes are inserted while correcting the loop

  // FIXME: online xx
  global::mapper->RequestStop();

  // If a Global Bundle Adjustment is running, abort it
  if (isRunningGBA()) {
    unique_lock<mutex> lock(mMutexGBA);
    mbStopGBA = true;

    mnFullBAIdx++;

    if (mpThreadGBA) {
      mpThreadGBA->detach();
      delete mpThreadGBA;
    }
  }

  // Wait until Local Mapping has effectively stopped
  while (!global::mapper->isStopped()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Ensure current keyframe is updated
  mpCurrentKF->UpdateConnections();

  // Retrive keyframes connected to the current keyframe and compute corrected
  // Sim3 pose by propagation
  mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
  mvpCurrentConnectedKFs.push_back(mpCurrentKF);

  KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
  CorrectedSim3[mpCurrentKF] = mg2oScw;
  cv::Mat Twc = mpCurrentKF->GetPoseInverse();

  {
    // Get Map Mutex
    unique_lock<mutex> lock(global::map->mMutexMapUpdate);

    for (vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin(),
                                      vend = mvpCurrentConnectedKFs.end();
         vit != vend; vit++) {
      KeyFrame *pKFi = *vit;

      cv::Mat Tiw = pKFi->GetPose();

      if (pKFi != mpCurrentKF) {
        cv::Mat Tic = Tiw * Twc;
        cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
        cv::Mat tic = Tic.rowRange(0, 3).col(3);
        g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic),
                         1.0);
        g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
        // Pose corrected with the Sim3 of the loop closure
        CorrectedSim3[pKFi] = g2oCorrectedSiw;
      }

      cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
      cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
      g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw),
                       1.0);
      // Pose without correction
      NonCorrectedSim3[pKFi] = g2oSiw;
    }

    // Correct all MapPoints obsrved by current keyframe and neighbors, so
    // that they align with the other side of the loop
    for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(),
                                   mend = CorrectedSim3.end();
         mit != mend; mit++) {
      KeyFrame *pKFi = mit->first;
      g2o::Sim3 g2oCorrectedSiw = mit->second;
      g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

      g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

      vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();
      for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++) {
        MapPoint *pMPi = vpMPsi[iMP];
        if (!pMPi)
          continue;
        if (pMPi->isBad())
          continue;
        if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
          continue;

        // Project with non-corrected pose and project back with corrected
        // pose
        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw =
            g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMPi->SetWorldPos(cvCorrectedP3Dw);
        pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
        pMPi->mnCorrectedReference = pKFi->mnId;
        pMPi->UpdateNormalAndDepth();
      }

      // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3
      // (scale translation)
      Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
      Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
      double s = g2oCorrectedSiw.scale();

      eigt *= (1. / s); //[R t/s;0 1]

      cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

      pKFi->SetPose(correctedTiw);

      // Make sure connections are updated
      pKFi->UpdateConnections();
    }

    // Start Loop Fusion
    // Update matched map points and replace if duplicated
    for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) {
      if (mvpCurrentMatchedPoints[i]) {
        MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];
        MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
        if (pCurMP)
          pCurMP->Replace(pLoopMP);
        else {
          mpCurrentKF->AddMapPoint(pLoopMP, i);
          pLoopMP->AddObservation(mpCurrentKF, i);
          pLoopMP->ComputeDistinctiveDescriptors();
        }
      }
    }
  }

  // Project MapPoints observed in the neighborhood of the loop keyframe
  // into the current keyframe and neighbors using corrected poses.
  // Fuse duplications.
  SearchAndFuse(CorrectedSim3);

  // After the MapPoint fusion, new links in the covisibility graph will
  // appear attaching both sides of the loop
  map<KeyFrame *, set<KeyFrame *>> LoopConnections;

  for (vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin(),
                                    vend = mvpCurrentConnectedKFs.end();
       vit != vend; vit++) {
    KeyFrame *pKFi = *vit;
    vector<KeyFrame *> vpPreviousNeighbors =
        pKFi->GetVectorCovisibleKeyFrames();

    // Update connections. Detect new links.
    pKFi->UpdateConnections();
    LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
    for (vector<KeyFrame *>::iterator vit_prev = vpPreviousNeighbors.begin(),
                                      vend_prev = vpPreviousNeighbors.end();
         vit_prev != vend_prev; vit_prev++) {
      LoopConnections[pKFi].erase(*vit_prev);
    }
    for (vector<KeyFrame *>::iterator vit2 = mvpCurrentConnectedKFs.begin(),
                                      vend2 = mvpCurrentConnectedKFs.end();
         vit2 != vend2; vit2++) {
      LoopConnections[pKFi].erase(*vit2);
    }
  }

  // Optimize graph
  Optimizer::OptimizeEssentialGraph(global::map, mpMatchedKF, mpCurrentKF,
                                    NonCorrectedSim3, CorrectedSim3,
                                    LoopConnections, mbFixScale);

  // Add loop edge
  mpMatchedKF->AddLoopEdge(mpCurrentKF);
  mpCurrentKF->AddLoopEdge(mpMatchedKF);

  // Launch a new thread to perform Global Bundle Adjustment
  mbRunningGBA = true;
  mbFinishedGBA = false;
  mbStopGBA = false;

  {
    mpThreadGBA = new thread(&LoopClosingVLAD::RunGlobalBundleAdjustment, this,
                             mpCurrentKF->mnId);

    // RunGlobalBundleAdjustment(mpCurrentKF->mnId);
  }

  // Loop closed. Release Local Mapping.
  global::mapper->Release();

  LOG(WARNING) << "Loop Closed!";

  mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosingVLAD::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap) {
  SPMatcher matcher(0.8);

  for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(),
                                       mend = CorrectedPosesMap.end();
       mit != mend; mit++) {
    KeyFrame *pKF = mit->first;

    g2o::Sim3 g2oScw = mit->second;
    cv::Mat cvScw = Converter::toCvMat(g2oScw);

    vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(),
                                       static_cast<MapPoint *>(NULL));
    matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

    // Get Map Mutex
    unique_lock<mutex> lock(global::map->mMutexMapUpdate);
    const int nLP = mvpLoopMapPoints.size();
    for (int i = 0; i < nLP; i++) {
      MapPoint *pRep = vpReplacePoints[i];
      if (pRep) {
        pRep->Replace(mvpLoopMapPoints[i]);
      }
    }
  }
}

void LoopClosingVLAD::RequestReset() {
  {
    unique_lock<mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  while (1) {
    {
      unique_lock<mutex> lock2(mMutexReset);
      if (!mbResetRequested)
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void LoopClosingVLAD::ResetIfRequested() {
  unique_lock<mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlpLoopKeyFrameQueue.clear();
    mLastLoopKFid = 0;
    mbResetRequested = false;
  }
}

void LoopClosingVLAD::RunGlobalBundleAdjustment(unsigned long nLoopKF) {
  std::cout << "Starting Global Bundle Adjustment" << endl;
  timing::Timer gba_timer("loop/init_gba");

  int idx = mnFullBAIdx;
  Optimizer::GlobalBundleAdjustemnt(global::map, 10, &mbStopGBA, nLoopKF,
                                    false);

  // Update all MapPoints and KeyFrames
  // Local Mapping was active during BA, that means that there might be new
  {
    unique_lock<mutex> lock(mMutexGBA);
    if (idx != mnFullBAIdx)
      return;

    if (!mbStopGBA) {
      std::cout << "Global Bundle Adjustment finished" << endl;
      std::cout << "Updating map ..." << endl;

      // FIXME:
      global::mapper->RequestStop();
      // Wait until Local Mapping has effectively stopped

      while (!global::mapper->isStopped() && !global::mapper->isFinished()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      // Get Map Mutex
      unique_lock<mutex> lock(global::map->mMutexMapUpdate);

      // Correct keyframes starting at map first keyframe
      list<KeyFrame *> lpKFtoCheck(global::map->mvpKeyFrameOrigins.begin(),
                                   global::map->mvpKeyFrameOrigins.end());

      while (!lpKFtoCheck.empty()) {
        KeyFrame *pKF = lpKFtoCheck.front();
        const set<KeyFrame *> sChilds = pKF->GetChilds();
        cv::Mat Twc = pKF->GetPoseInverse();
        for (set<KeyFrame *>::const_iterator sit = sChilds.begin();
             sit != sChilds.end(); sit++) {
          KeyFrame *pChild = *sit;
          if (pChild->mnBAGlobalForKF != nLoopKF) {
            cv::Mat Tchildc = pChild->GetPose() * Twc;
            pChild->mTcwGBA = Tchildc * pKF->mTcwGBA; //*Tcorc*pKF->mTcwGBA;
            pChild->mnBAGlobalForKF = nLoopKF;
          }
          lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);
        lpKFtoCheck.pop_front();
      }

      // Correct MapPoints
      const vector<MapPoint *> vpMPs = global::map->GetAllMapPoints();

      for (size_t i = 0; i < vpMPs.size(); i++) {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
          continue;

        if (pMP->mnBAGlobalForKF == nLoopKF) {
          // If optimized by Global BA, just update
          pMP->SetWorldPos(pMP->mPosGBA);
        } else {
          // Update according to the correction of its reference keyframe
          KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

          if (pRefKF->mnBAGlobalForKF != nLoopKF)
            continue;

          // Map to non-corrected camera
          cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
          cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
          cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

          // Backproject using corrected camera
          cv::Mat Twc = pRefKF->GetPoseInverse();
          cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
          cv::Mat twc = Twc.rowRange(0, 3).col(3);

          pMP->SetWorldPos(Rwc * Xc + twc);
        }
      }

      global::mapper->Release();

      std::cout << "Map updated!" << endl;
    }

    mbFinishedGBA = true;
    mbRunningGBA = false;
  }
  gba_timer.Stop();
}

void LoopClosingVLAD::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool LoopClosingVLAD::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void LoopClosingVLAD::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
}

bool LoopClosingVLAD::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

} // namespace orbslam
