#pragma once

#include <list>
#include <vector>

#include <Eigen/Dense>
#include <opencv/cv.h>

#include <torch/script.h>
#include <torch/torch.h>

#include "base_extractor.h"

namespace orbslam {

struct SPFrontend : torch::nn::Module {
  // public:
  SPFrontend(const float conf_thresh, const int height, const int width,
             const int cell_size);

  torch::Tensor grid;
  float conf_thresh;
  int h, w, hc, wc, c;

  // torch::Device device_;

  std::vector<torch::Tensor> forward(torch::Tensor x);

  torch::nn::Conv2d conv1a;
  torch::nn::Conv2d conv1b;

  torch::nn::Conv2d conv2a;
  torch::nn::Conv2d conv2b;

  torch::nn::Conv2d conv3a;
  torch::nn::Conv2d conv3b;

  torch::nn::Conv2d conv4a;
  torch::nn::Conv2d conv4b;

  torch::nn::Conv2d convPa;
  torch::nn::Conv2d convPb;

  // descriptor
  torch::nn::Conv2d convDa;
  torch::nn::Conv2d convDb;
};

class SPExtractor : public BaseExtractor {
public:
  SPExtractor(int nfeatures);
  // SPExtractor(int nfeatures, std::string weight_path,
  //             float conf_thresh = 0.007);

  virtual ~SPExtractor() = default;

  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint> &keypoints,
                  cv::OutputArray descriptors) override;

  cv::Mat getMask() { return mask_; }

  cv::Mat getHeatMap() { return heat_; }

  const std::vector<Eigen::Vector2f> getCov() { return cov2_; }

  const std::vector<Eigen::Vector2f> getCov2Inv() { return cov2_inv_; }

  cv::Mat semi_dust_, dense_dust_;

  cv::Mat mask_, heat_, heat_inv_;

  cv::Mat occ_grid_;

protected:
  std::vector<Eigen::Vector2f> cov2_, cov2_inv_;
  std::vector<Eigen::Matrix2f> info_mat_;
  // std::vector<cv::Point2f> cov2_inv_;

  int num_feature_;

  float conf_thresh = 0.015;

  // std::shared_ptr<torch::jit::script::Module> model_;
  torch::Device device_;

  std::shared_ptr<SPFrontend> model_;
};

} // namespace orbslam
