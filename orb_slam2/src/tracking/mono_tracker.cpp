#include "orb_slam/tracking/mono_tracker.h"

#include "orb_slam/cv/orb_matcher.h"
#include "orb_slam/cv/sp_matcher.h"
#include "orb_slam/mapping/optimizer.h"

#include "orb_slam/config.h"
#include "orb_slam/global.h"

namespace orbslam {

using namespace std;

void MonoTracker::setFrameData(const double &timestamp, const cv::Mat &im,
                               const cv::Mat &im_nul) {
  mImGray = im;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mK, mDistCoef,
                          mbf, mThDepth);
  else
    mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mK, mDistCoef,
                          mbf, mThDepth);
}

void MonoTracker::Initialization() {
  static int npt_thresh_detect = 100;
  static int npt_thresh_match = 100;

  if (tracking::extractor_type == tracking::SP) {
    npt_thresh_detect = 40;
    npt_thresh_match = 40;
  }

  if (!mpInitializer) {
    if (mCurrentFrame.mvKeys.size() > npt_thresh_detect) {

      // ADD(debug)
      mImInit = mImGray.clone();

      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      if (mpInitializer)
        delete mpInitializer;

      mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      cout << "Initializer constructed ad id: " << mCurrentFrame.mnId << endl;
      return;
    }
  } else {
    // Try to initialize
    if ((int)mCurrentFrame.mvKeys.size() <= npt_thresh_detect) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer *>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }

    // Find correspondences
    SPMatcher matcher(0.9);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < npt_thresh_match) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer *>(NULL);
      return;
    }

    cv::Mat Rcw;                 // Current Camera Rotation
    cv::Mat tcw;                 // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw,
                                  mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      cout << "Initialize at id: " << mCurrentFrame.mnId << endl;

      // Set Frame Poses
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      CreateInitialMap();
    }
  }
}

void MonoTracker::CreateInitialMap() {
  // Create KeyFrames
  KeyFrame *pKFini = new KeyFrame(mInitialFrame);
  KeyFrame *pKFcur = new KeyFrame(mCurrentFrame);

  // TODO: debug
  pKFcur->mIm = mImGray.clone();
  pKFini->mIm = mImInit.clone();

  // pKFini->ComputeBoW();
  // pKFcur->ComputeBoW();

  // Insert KFs in the map
  global::map->AddKeyFrame(pKFini);
  global::map->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0)
      continue;

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint *pMP = new MapPoint(worldPos, pKFcur);

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->updateDescTrack(pKFcur, mvIniMatches[i]);

    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    global::map->AddMapPoint(pMP);
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "New Map created with " << global::map->MapPointsInMap() << " points"
       << endl;

  Optimizer::GlobalBundleAdjustemnt(global::map, 20);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
    cout << "Wrong initialization, reseting..." << endl;
    global::b_system_reset = true;
    return;
  }

  // Scale initial baseline
  {
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
      if (vpAllMapPoints[iMP]) {
        MapPoint *pMP = vpAllMapPoints[iMP];
        pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
      }
    }
  }
  // ADD(depth_filer)
  // pKFcur->ComputeSceneMeanDepth(2);
  // pKFcur->initializeSeeds();

  std::unique_lock<std::mutex> lock(global::map->mutex_lastkf);
  global::map->pLastKF = pKFcur;

  global::mapper->InsertKeyFrame(pKFini);
  global::mapper->InsertKeyFrame(pKFcur);

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = global::map->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  global::map_drawer->SetCurrentCameraPose(pKFcur->GetPose());

  global::map->mvpKeyFrameOrigins.push_back(pKFini);

  mState = OK;
}

} // namespace orbslam