#pragma once

#include <unordered_map>

#include <tensorboard_logger/tensorboard_logger.h>

#include <iostream>
using namespace std;

namespace orbslam {
class SummaryWriter {
public:
  SummaryWriter() = default;

  explicit SummaryWriter(const std::string &log_dir) : log_dir_(log_dir) {
    if (log_dir_.back() != '/')
      log_dir_ += '/';
  }

  ~SummaryWriter() = default;

  bool addScalar(const std::string &tag, const int step, const float scalar,
                 const std::string &name = "scalar") {

    if (!map_.count(tag)) {
      map_[tag] = std::shared_ptr<TensorBoardLogger>(
          new TensorBoardLogger((log_dir_ + tag).c_str()));
    }

    map_[tag]->add_scalar(name, step, scalar);
  }

  bool addHist(const std::string &tag, const int step,
                 std::vector<float> &hist,
                 const std::string &name = "scalar") {

    if (!map_.count(tag)) {
      map_[tag] = std::shared_ptr<TensorBoardLogger>(
          new TensorBoardLogger((log_dir_ + tag).c_str()));
    }

    map_[tag]->add_histogram(name, step, hist);
  }

  std::string log_dir_;

  std::unordered_map<std::string, std::shared_ptr<TensorBoardLogger>> map_;
};

} // namespace orbslam
