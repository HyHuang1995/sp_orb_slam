#pragma once

#include <cstdint>
#include <memory>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

namespace orbslam {

enum class DataType : std::uint8_t {
  Mono = 1,
  Stereo = 1 << 1,
  Depth = 1 << 2,
  IMU = 1 << 3,
  Lidar = 1 << 4,
  Odom = 1 << 5,

  Feat = 1 << 6,

  GT = 1 << 7

};

inline DataType operator|(DataType a, DataType b) {
  return static_cast<DataType>(static_cast<uint8_t>(a) |
                               static_cast<uint8_t>(b));
}

inline uint8_t operator&(DataType a, DataType b) {
  return static_cast<uint8_t>(a) & static_cast<uint8_t>(b);
}

struct DataFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  uint32_t idx = 0;

  using Ptr = std::shared_ptr<DataFrame>;

  using ConstPtr = std::shared_ptr<const DataFrame>;

  cv::Mat mono, depth;
  double timestamp;

  bool feature_extracted = false;

  Eigen::Vector3d t_w_c;
  Eigen::Quaterniond rot_w_c;

  // for direct test
  size_t num_pts;
  cv::Mat pts, desc;
  cv::Mat semi, semi_half, dense;
  cv::Mat global_desc;

  cv::Mat dense_sm;
};

class Dataloader {
public:
  using Ptr = std::shared_ptr<Dataloader>;

  using ConstPtr = std::shared_ptr<const Dataloader>;

  Dataloader() = default;

  Dataloader(const std::string &str, DataType cfg);

  virtual ~Dataloader() = default;

  virtual DataFrame::Ptr getNextFrame() = 0;

  virtual DataFrame::Ptr getFrameByIndex(size_t idx) = 0;

  // virtual bool getTrajectory(
  //     std::vector<Eigen::Quaterniond,
  //                 Eigen::aligned_allocator<Eigen::Quaterniond>> &rot,
  //     std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
  //         &trans) = 0;

  bool getTrajectory(
      std::vector<Eigen::Quaterniond,
                  Eigen::aligned_allocator<Eigen::Quaterniond>> &rot,
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          &trans) {

    std::copy(rot_.begin(), rot_.end(), back_inserter(rot));
    std::copy(trans_.begin(), trans_.end(), back_inserter(trans));
  }
  // virtual DataFrame::Ptr getFrameByIndex(size_t idx) = 0;

  const size_t getSize() const { return num_; }

protected:
  size_t num_ = 0, idx_ = 0;
  DataType cfg_;

  std::string base_path_;

  std::vector<std::string> mono_file_, depth_file_;
  std::vector<double> time_stamp_;

  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>
      rot_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      trans_;
};

class DataloaderKITTIExport : public Dataloader {
public:
  using Ptr = std::shared_ptr<DataloaderKITTIExport>;

  using ConstPtr = std::shared_ptr<const DataloaderKITTIExport>;

  DataloaderKITTIExport(const std::string &str, const std::string &seq,
                        DataType cfg);

  ~DataloaderKITTIExport() = default;

  DataFrame::Ptr getNextFrame() override;

  const size_t getSize() const { return num_; }

  DataFrame::Ptr getFrameByIndex(size_t idx) override;

private:
  void loadTrajectory(const std::string &traj_file);

  void loadImages(const std::string &ass_file);

  void loadGlobalDesc(const std::string &base_path);

private:
  std::vector<std::string> global_desc_files;

  std::string seq_str;
};

class DataloaderTsukuba : public Dataloader {
public:
  using Ptr = std::shared_ptr<DataloaderTsukuba>;

  using ConstPtr = std::shared_ptr<const DataloaderTsukuba>;

  DataloaderTsukuba(const std::string &str, const std::string &seq,
                    DataType cfg);

  ~DataloaderTsukuba() = default;

  DataFrame::Ptr getNextFrame() override;

  const size_t getSize() const { return num_; }

  DataFrame::Ptr getFrameByIndex(size_t idx) override;

private:
  // void loadTrajectory(const std::string &traj_file);

  void loadImages(const std::string &ass_file);

  // void loadGlobalDesc(const std::string &base_path);

private:
  std::vector<std::string> global_desc_files;
};

class DataloaderEuRoC : public Dataloader {
public:
  using Ptr = std::shared_ptr<DataloaderEuRoC>;

  using ConstPtr = std::shared_ptr<const DataloaderEuRoC>;

  DataloaderEuRoC(const std::string &str, DataType cfg);

  ~DataloaderEuRoC() = default;

  DataFrame::Ptr getNextFrame() override;

  DataFrame::Ptr getFrameByIndex(size_t idx) override;

private:
  void loadTrajectory(const std::string &traj_file);

  void loadImages(const std::string &ass_file);

  cv::Mat m1, m2;
};

class DataloaderEuRoCExport : public Dataloader {
public:
  using Ptr = std::shared_ptr<DataloaderEuRoCExport>;

  using ConstPtr = std::shared_ptr<const DataloaderEuRoCExport>;

  DataloaderEuRoCExport(const std::string &str, DataType cfg);

  ~DataloaderEuRoCExport() = default;

  DataFrame::Ptr getNextFrame() override;

  const size_t getSize() const { return num_; }

  DataFrame::Ptr getFrameByIndex(size_t idx) override;

private:
  void loadTrajectory(const std::string &traj_file);

  void loadImages(const std::string &ass_file);

  void loadGlobalDesc(const std::string &base_path);

private:
  std::vector<std::string> global_desc_files;
};

} // namespace orbslam