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

#include "orb_slam/viz/viewer.h"

#include <mutex>

#include <pangolin/pangolin.h>

#include "orb_slam/global.h"

#include "orb_slam/config.h"

namespace orbslam {

using namespace std;

Viewer::Viewer()
    : mbFinishRequested(false), mbFinished(true), mbStopped(false),
      mbStopRequested(false) {
  float fps = camera::fps;
  if (fps < 1)
    fps = 30;
  mT = 1e3 / fps;

  mImageWidth = camera::width;
  mImageHeight = camera::height;
  if (mImageWidth < 1 || mImageHeight < 1) {
    mImageWidth = 752;
    mImageHeight = 480;
  }

  mViewpointX = viewer::viewpoint_x;
  mViewpointY = viewer::viewpoint_y;
  mViewpointZ = viewer::viewpoint_z;
  mViewpointF = viewer::viewpoint_f;
}

void Viewer::Run() {
  mbFinished = false;

  pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);

  glEnable(GL_DEPTH_TEST);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(175));
  pangolin::Var<bool> menu_pause("menu.Pause", false, false);
  pangolin::Var<bool> menu_step("menu.Step", false, false);
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false, true);
  pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
  pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
  pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false,
                                           true);
  pangolin::Var<bool> menuReset("menu.Reset", false, false);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389,
                                 0.1, 1000),
      pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0,
                                0.0, -1.0, 0.0));

  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                         1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::Display("extract")
      .SetAspect(752.0 / 480.0)
      .SetLock(pangolin::LockLeft,
               pangolin::LockBottom); // TODO: check height
  pangolin::Display("track")
      .SetAspect(752.0 / 480.0)
      .SetLock(pangolin::LockLeft,
               pangolin::LockBottom); // TODO: check height
  pangolin::Display("multi")
      .SetBounds(0.0, 1 / 3.0f, pangolin::Attach::Pix(175), 1.0 / 2.0f)
      .SetLayout(pangolin::LayoutEqualHorizontal)
      .SetLock(pangolin::LockLeft,
               pangolin::LockBottom) // TODO: check height
      .AddDisplay(pangolin::Display("extract"))
      .AddDisplay(pangolin::Display("track"));
  pangolin::GlTexture extract_tex(752, 480, GL_RGB, false, 0, GL_RGB,
                                  GL_UNSIGNED_BYTE);
  pangolin::GlTexture track_tex(752, 480, GL_RGB, false, 0, GL_RGB,
                                GL_UNSIGNED_BYTE);

  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  cv::namedWindow("ORB-SLAM2: Current Frame");
  // cv::namedWindow("coarse");

  bool bFollow = true;
  bool bLocalizationMode = false;

  while (1) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);
    global::b_pause = menu_pause;
    if (menu_step) {
      global::b_step = true;
      menu_step = false;
    }

    if (menuFollowCamera && bFollow) {
      s_cam.Follow(Twc);
    } else if (menuFollowCamera && !bFollow) {
      s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
          mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
      s_cam.Follow(Twc);
      bFollow = true;
    } else if (!menuFollowCamera && bFollow) {
      bFollow = false;
    }

    if (menuLocalizationMode && !bLocalizationMode) {
      global::b_local_on = true;
      bLocalizationMode = true;
    } else if (!menuLocalizationMode && bLocalizationMode) {
      global::b_local_off = true;
      bLocalizationMode = false;
    }

    d_cam.Activate(s_cam);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    if (mpFrameDrawer->mState == Tracking::OK) {
      mpMapDrawer->DrawCurrentCamera(Twc);
    }
    if (menuShowKeyFrames || menuShowGraph)
      mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
    if (menuShowPoints)
      mpMapDrawer->DrawMapPoints();

    cv::Mat im = mpFrameDrawer->DrawFrame();
    // cv::Mat imflip;
    // if (!im.empty()) {
    //   cv::flip(im, imflip, 0);
    //   track_tex.Upload(imflip.data, GL_RGB, GL_UNSIGNED_BYTE);
    // }
    // pangolin::Display("track").Activate();
    // glColor3f(1.0, 1.0, 1.0);
    // track_tex.RenderToViewport();

    // if (!mpFrameDrawer->img_sp_viz.empty()) {
    //   cv::flip(mpFrameDrawer->img_sp_viz, imflip, 0);
    //   extract_tex.Upload(imflip.data, GL_RGB, GL_UNSIGNED_BYTE);
    // }
    // pangolin::Display("extract").Activate();
    // glColor3f(1.0, 1.0, 1.0);
    // extract_tex.RenderToViewport();

    pangolin::FinishFrame();

    cv::imshow("ORB-SLAM2: Current Frame", im);
    if (mpFrameDrawer->is_ok)
      cv::imshow("coarse", mpFrameDrawer->mVizCoarse);
    cv::waitKey(mT);

    if (menuReset) {
      menuShowGraph = true;
      menuShowKeyFrames = true;
      menuShowPoints = true;
      menuLocalizationMode = false;
      bLocalizationMode = false;
      bFollow = true;
      menuFollowCamera = true;
      // mpSystem->Reset();
      menuReset = false;

      global::b_system_reset = true;
      if (bLocalizationMode) {
        global::b_local_off = true;
      }
    }

    if (Stop()) {
      while (isStopped()) {
        // usleep(3000);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
    }

    if (CheckFinish())
      break;
  }

  SetFinish();
}

void Viewer::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool Viewer::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void Viewer::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
}

bool Viewer::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

void Viewer::RequestStop() {
  unique_lock<mutex> lock(mMutexStop);
  if (!mbStopped)
    mbStopRequested = true;
}

bool Viewer::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

bool Viewer::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);

  if (mbFinishRequested)
    return false;
  else if (mbStopRequested) {
    mbStopped = true;
    mbStopRequested = false;
    return true;
  }

  return false;
}

void Viewer::Release() {
  unique_lock<mutex> lock(mMutexStop);
  mbStopped = false;
}

} // namespace orbslam
