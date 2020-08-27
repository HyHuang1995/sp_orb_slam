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

#include "orb_slam/viz/map_drawer.h"

#include <mutex>
#include <random>

#include <pangolin/pangolin.h>

#include "orb_slam/config.h"

#include "orb_slam/type/keyframe.h"
#include "orb_slam/type/mappoint.h"

namespace orbslam {

using namespace std;

inline cv::Vec3b makeJet3B(float id) {
  if (id <= 0)
    return cv::Vec3b(128, 0, 0);
  if (id >= 1)
    return cv::Vec3b(0, 0, 128);

  int icP = (id * 8);
  float ifP = (id * 8) - icP;

  if (icP == 0)
    return cv::Vec3b(255 * (0.5 + 0.5 * ifP), 0, 0);
  if (icP == 1)
    return cv::Vec3b(255, 255 * (0.5 * ifP), 0);
  if (icP == 2)
    return cv::Vec3b(255, 255 * (0.5 + 0.5 * ifP), 0);
  if (icP == 3)
    return cv::Vec3b(255 * (1 - 0.5 * ifP), 255, 255 * (0.5 * ifP));
  if (icP == 4)
    return cv::Vec3b(255 * (0.5 - 0.5 * ifP), 255, 255 * (0.5 + 0.5 * ifP));
  if (icP == 5)
    return cv::Vec3b(0, 255 * (1 - 0.5 * ifP), 255);
  if (icP == 6)
    return cv::Vec3b(0, 255 * (0.5 - 0.5 * ifP), 255);
  if (icP == 7)
    return cv::Vec3b(0, 0, 255 * (1 - 0.5 * ifP));
  return cv::Vec3b(255, 255, 255);
}

MapDrawer::MapDrawer() {
  mKeyFrameSize = viewer::kf_size;
  mKeyFrameLineWidth = viewer::kf_line_width;
  mGraphLineWidth = viewer::graph_line_width;
  mPointSize = viewer::point_size;
  mCameraSize = viewer::camera_size;
  mCameraLineWidth = viewer::camera_line_width;
}

void MapDrawer::DrawMapPoints() {
  const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
  const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();

  set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

  if (vpMPs.empty())
    return;

  // for AllMapPoints
  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  glColor3f(0.7529411764705882, 0.7529411764705882, 0.7529411764705882);

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
      continue;
    cv::Mat pos = vpMPs[i]->GetWorldPos();
    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
  }
  glEnd();

  // for ReferenceMapPoints
  glPointSize(mPointSize * 1.6);
  glBegin(GL_POINTS);

  int size = vpRefMPs.size();
  for (int i = 0; i < size; i++) {
    auto &&mp = vpRefMPs[i];
    cv::Mat pos = mp->GetWorldPos();
    auto color = makeJet3B(i * 0.8f / size);
    glColor3f(color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f);
    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
  }
  glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph) {
  const float &w = mKeyFrameSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  vector<KeyFrame *> vpBAKFs = mpMap->getBAKeyFrames();
  sort(vpBAKFs.begin(), vpBAKFs.end(),
       [](const KeyFrame *a, const KeyFrame *b) -> bool {
         return a->mnId > b->mnId;
       });
  set<KeyFrame *> spBAKFs(vpBAKFs.begin(), vpBAKFs.end());
  vector<KeyFrame *> vpBAKFsFixed = mpMap->getBAKeyFramesFixed();
  sort(vpBAKFsFixed.begin(), vpBAKFsFixed.end(),
       [](const KeyFrame *a, const KeyFrame *b) -> bool {
         return a->mnId > b->mnId;
       });
  set<KeyFrame *> spBAKFsFixed(vpBAKFsFixed.begin(), vpBAKFsFixed.end());

  auto dustRef = mpMap->pDustRef;
  if (bDrawKF) {
    for (size_t i = 0; i < vpKFs.size(); i++) {
      KeyFrame *pKF = vpKFs[i];
      cv::Mat Twc = pKF->GetPoseInverse().t();

      glPushMatrix();

      glMultMatrixf(Twc.ptr<GLfloat>(0));

      auto it = find(vpBAKFs.begin(), vpBAKFs.end(), pKF);
      auto it_fixed = find(vpBAKFsFixed.begin(), vpBAKFsFixed.end(), pKF);
      if (it != vpBAKFs.end()) {
        auto color =
            makeJet3B((it - vpBAKFs.begin()) * 0.55f / vpBAKFs.size() + 0.2f);
        glColor3f(color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f);
        glLineWidth(mKeyFrameLineWidth * 2);
      } else if (it_fixed != vpBAKFsFixed.end()) {
        glColor3f(0.0f, 0.7490196078431373f, 1.0f);
        glLineWidth(mKeyFrameLineWidth);
      } else {
        glColor3f(176.0 / 255, 196.0 / 255, 222.0 / 255);
        glLineWidth(mKeyFrameLineWidth);
      }

      glBegin(GL_LINES);
      glVertex3f(0, 0, 0);
      glVertex3f(w, h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, h, z);

      glVertex3f(w, h, z);
      glVertex3f(w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(-w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(w, h, z);

      glVertex3f(-w, -h, z);
      glVertex3f(w, -h, z);
      glEnd();

      glPopMatrix();
    }
    for (size_t i = 0; i < vpBAKFsFixed.size(); i++) {
    }
  }

  if (vpKFs.size() > 0) {
  }

  if (bDrawGraph) {
    glLineWidth(mGraphLineWidth);
    glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
    glBegin(GL_LINES);

    for (size_t i = 0; i < vpKFs.size(); i++) {
      // Covisibility Graph
      const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
      cv::Mat Ow = vpKFs[i]->GetCameraCenter();
      if (!vCovKFs.empty()) {
        for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(),
                                                vend = vCovKFs.end();
             vit != vend; vit++) {
          if ((*vit)->mnId < vpKFs[i]->mnId)
            continue;
          cv::Mat Ow2 = (*vit)->GetCameraCenter();
          glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
          glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
        }
      }

      // Spanning tree
      KeyFrame *pParent = vpKFs[i]->GetParent();
      if (pParent) {
        cv::Mat Owp = pParent->GetCameraCenter();
        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
        glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
      }

      // Loops
      set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
      for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(),
                                     send = sLoopKFs.end();
           sit != send; sit++) {
        if ((*sit)->mnId < vpKFs[i]->mnId)
          continue;
        cv::Mat Owl = (*sit)->GetCameraCenter();
        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
        glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
      }
    }

    glEnd();
  }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) {
  const float &w = mCameraSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  glPushMatrix();

#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(Twc.m);
#endif

  glLineWidth(mCameraLineWidth);
  glColor3f(0.7529411764705882, 0.7529411764705882, 0.7529411764705882);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  glPopMatrix();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw) {
  unique_lock<mutex> lock(mMutexCamera);
  mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M) {
  if (!mCameraPose.empty()) {
    cv::Mat Rwc(3, 3, CV_32F);
    cv::Mat twc(3, 1, CV_32F);
    {
      unique_lock<mutex> lock(mMutexCamera);
      Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
      twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
    }

    M.m[0] = Rwc.at<float>(0, 0);
    M.m[1] = Rwc.at<float>(1, 0);
    M.m[2] = Rwc.at<float>(2, 0);
    M.m[3] = 0.0;

    M.m[4] = Rwc.at<float>(0, 1);
    M.m[5] = Rwc.at<float>(1, 1);
    M.m[6] = Rwc.at<float>(2, 1);
    M.m[7] = 0.0;

    M.m[8] = Rwc.at<float>(0, 2);
    M.m[9] = Rwc.at<float>(1, 2);
    M.m[10] = Rwc.at<float>(2, 2);
    M.m[11] = 0.0;

    M.m[12] = twc.at<float>(0);
    M.m[13] = twc.at<float>(1);
    M.m[14] = twc.at<float>(2);
    M.m[15] = 1.0;
  } else
    M.SetIdentity();
}

} // namespace orbslam
