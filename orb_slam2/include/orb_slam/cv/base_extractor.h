#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace orbslam {
class BaseExtractor {
public:
  BaseExtractor() = delete;

  BaseExtractor(int _nfeatures, float _scaleFactor, int _nlevels,
                int _iniThFAST, int _minThFAST)
      : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        iniThFAST(_iniThFAST), minThFAST(_minThFAST) {
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++) {
      mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
      mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++) {
      mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
      mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale =
        nfeatures * (1 - factor) /
        (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++) {
      mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
      sumFeatures += mnFeaturesPerLevel[level];
      nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
  }

  virtual ~BaseExtractor() = default;

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  virtual void operator()(cv::InputArray image, cv::InputArray mask,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::OutputArray descriptors) = 0;

  int inline GetLevels() { return nlevels; }

  float inline GetScaleFactor() { return scaleFactor; }

  std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

  std::vector<float> inline GetInverseScaleFactors() {
    return mvInvScaleFactor;
  }

  std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return mvInvLevelSigma2;
  }

  std::vector<cv::Mat> mvImagePyramid;

protected:
  void ComputePyramid(cv::Mat image);

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  std::vector<int> mnFeaturesPerLevel;

  std::vector<int> umax;

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
};

} // namespace orbslam
