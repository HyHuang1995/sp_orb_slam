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

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "../type/frame.h"
#include "../type/keyframe.h"
#include "../type/mappoint.h"

namespace orbslam {

class ORBmatcher {

public:
  ORBmatcher(float nnratio = 0.6, bool checkOri = true);

  // Computes the Hamming distance between two ORB descriptors
  static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

  int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints,
                         const float th = 3);

  int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                         const float th, const bool bMono);

  int SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF,
                         const std::set<MapPoint *> &sAlreadyFound,
                         const float th, const int ORBdist);

  int SearchByProjection(KeyFrame *pKF, cv::Mat Scw,
                         const std::vector<MapPoint *> &vpPoints,
                         std::vector<MapPoint *> &vpMatched, int th);

  int SearchByBoW(KeyFrame *pKF, Frame &F,
                  std::vector<MapPoint *> &vpMapPointMatches);
  int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2,
                  std::vector<MapPoint *> &vpMatches12);

  int SearchForInitialization(Frame &F1, Frame &F2,
                              std::vector<cv::Point2f> &vbPrevMatched,
                              std::vector<int> &vnMatches12,
                              int windowSize = 10);

  int SearchForTriangulation(
      KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
      std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
      const bool bOnlyStereo);

  int SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2,
                   std::vector<MapPoint *> &vpMatches12, const float &s12,
                   const cv::Mat &R12, const cv::Mat &t12, const float th);

  int Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints,
           const float th = 3.0);

  int Fuse(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints,
           float th, std::vector<MapPoint *> &vpReplacePoint);

public:
  static const int TH_LOW;
  static const int TH_HIGH;
  static const int HISTO_LENGTH;

protected:
  bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                             const cv::Mat &F12, const KeyFrame *pKF);

  float RadiusByViewingCos(const float &viewCos);

  void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1,
                          int &ind2, int &ind3);

  float mfNNratio;
  bool mbCheckOrientation;
};

} // namespace orbslam
