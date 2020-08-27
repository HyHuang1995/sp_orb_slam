#pragma once

#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

namespace orbslam {

struct NetVLAD : torch::nn::Module {
public:
  std::vector<torch::Tensor> forward(torch::Tensor x);

};

} // namespace orbslam
