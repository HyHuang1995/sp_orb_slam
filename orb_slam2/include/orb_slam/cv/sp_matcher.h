#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "../type/frame.h"
#include "../type/keyframe.h"
#include "../type/mappoint.h"

namespace orbslam {

class SPMatcher {

public:
  SPMatcher(float nnratio = 0.6);

  // Computes the Hamming distance between two ORB descriptors
  static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

  int SearchByFlann(KeyFrame *kf_db_ptr, KeyFrame *kf_qry_ptr,
                    std::vector<std::pair<size_t, size_t>> &vMatchesPairs);

  int SearchForTriByFlann(
      KeyFrame *kf_db_ptr, KeyFrame *kf_qry_ptr, cv::Mat F12,
      std::vector<std::pair<size_t, size_t>> &vMatchesPairs);

  int SearchForTriByEpi(KeyFrame *kf_db_ptr, KeyFrame *kf_qry_ptr, cv::Mat F12,
                        std::vector<std::pair<size_t, size_t>> &vMatchesPairs);

  // int SearchByFlann(Frame &F, const std::vector<MapPoint *> &vpMapPoints,
  //                   const float th = 3, const float th_dist = TH_HIGH);

  int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints,
                         const float th = 3, const float th_dist = TH_HIGH);

  int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                         const float th, const bool bMono);

  int SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF,
                         const std::set<MapPoint *> &sAlreadyFound,
                         const float th, const int ORBdist);

  int SearchByProjection(KeyFrame *pKF, cv::Mat Scw,
                         const std::vector<MapPoint *> &vpPoints,
                         std::vector<MapPoint *> &vpMatched, int th);

  int SearchByBruteForce(KeyFrame *curr_kf, Frame &kf,
                         std::vector<MapPoint *> &vpMapPointsMatches);

  // int SearchByBoW(KeyFrame *pKF, Frame &F,
  //                 std::vector<MapPoint *> &vpMapPointMatches);
  // int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2,
  //                 std::vector<MapPoint *> &vpMatches12);

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

  // ADD(Loop)
  int SearchByProjectionLoop(KeyFrame *pKF, cv::Mat Scw,
                             const std::vector<MapPoint *> &vpPoints,
                             std::vector<MapPoint *> &vpMatched, int th);

  int SearchBySim3Override(KeyFrame *pKF1, KeyFrame *pKF2,
                           std::vector<MapPoint *> &vpMatches12,
                           const float &s12, const cv::Mat &R12,
                           const cv::Mat &t12, const float th);

  int SearchByBruteForce(KeyFrame *curr_kf, KeyFrame *kf,
                         std::vector<MapPoint *> &vpMapPoints);
  // ADD(Loop)

public:
  static const float TH_LOW;
  static const float TH_HIGH;
  static const int HISTO_LENGTH;

protected:
  bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                             const cv::Mat &F12, const KeyFrame *pKF,
                             const int idx);

  float RadiusByViewingCos(const float &viewCos);

  float mfNNratio;
};

} // namespace orbslam
