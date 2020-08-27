#include "orb_slam/io/data_loader.h"

#include <glog/logging.h>

#include <fstream>
#include <iostream>

#include <sys/stat.h>

namespace orbslam {

using namespace std;

bool isPathExist(const std::string &s) {
  struct stat buffer;
  return (stat(s.c_str(), &buffer) == 0);
}

cv::Mat readDepthBinary(const std::string &file_name, const int height = 480,
                        const int width = 640) {

  // cout << file_name << endl;
  // allocate buffer
  const int buffer_size_ = sizeof(float) * height * width;
  char buffer_[buffer_size_];

  // open filestream && read buffer
  std::ifstream fs_bin_(file_name.c_str(), std::ios::binary);
  fs_bin_.read(buffer_, buffer_size_);
  fs_bin_.close();

  // construct depth map && return
  cv::Mat out = cv::Mat(cv::Size(width, height), CV_32FC1);
  std::memcpy(out.data, buffer_, buffer_size_);

  return out.clone();
}

cv::Mat readMatBinary(const std::string &file_name, const int height = 30,
                      const int width = 40) {

  const int buffer_size_ = sizeof(float) * height * width * 3;
  char buffer_[buffer_size_];

  // open filestream && read buffer
  std::ifstream fs_bin_(file_name.c_str(), std::ios::binary);
  fs_bin_.read(buffer_, buffer_size_);
  fs_bin_.close();

  // construct depth map && return
  cv::Mat out = cv::Mat(cv::Size(width, height), CV_32FC3);
  std::memcpy(out.data, buffer_, buffer_size_);

  return out.clone();
}

Dataloader::Dataloader(const std::string &str, DataType cfg)
    : cfg_(cfg), base_path_(str) {}

DataloaderEuRoCExport::DataloaderEuRoCExport(const std::string &str,
                                             DataType cfg)
    : Dataloader(str, cfg) {

  if (!isPathExist(base_path_))
    throw std::runtime_error("base path not exists: " + base_path_);

  if (cfg_ & DataType::GT) {
    string traj_file = base_path_ + "/state_groundtruth_estimate0/traj.txt";

    if (!isPathExist(traj_file))
      throw std::runtime_error("traj_file not exists: " + traj_file);

    loadTrajectory(traj_file);
    cout << "load trajectory: " << traj_file << endl;
    num_ = rot_.size();
  }

  if (cfg_ & DataType::Mono) {
    loadImages(base_path_);
    cout << "load images: " << base_path_ << endl;
    num_ = mono_file_.size();

    loadGlobalDesc(base_path_);
  }
}

void DataloaderEuRoCExport::loadGlobalDesc(const std::string &base_path) {
  cout << "load global desc" << endl;

  ifstream fTimes;
  string strPathTimeFile = base_path + "/cam0/data.csv";
  string strPathDesc = base_path + "/vlad/";

  fTimes.open(strPathTimeFile.c_str());
  string s;
  getline(fTimes, s);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      int index = s.find_first_of(",");
      string t = s.substr(0, index);

      // time_stamp_.push_back(stod(t) / 10.0e8);
      global_desc_files.push_back(strPathDesc + t + ".bin");
      // depth_file_.push_back(strPrefixRight + t + ".png");
    }
  }
}

DataFrame::Ptr DataloaderEuRoCExport::getNextFrame() {
  if (idx_ >= num_) {
    LOG(INFO) << "return nullptr";
    return nullptr;
  }
  LOG(INFO) << "retrieval idx: " << idx_;

  auto ptr = getFrameByIndex(idx_);

  idx_++;
  return ptr;
}

DataFrame::Ptr DataloaderEuRoCExport::getFrameByIndex(size_t idx) {
  if (idx >= num_)
    return nullptr;

  auto ptr = DataFrame::Ptr(new DataFrame());

  string bin_path = base_path_ + "/features/";

  stringstream ss;
  ss << setw(6) << setfill('0') << idx << '/';
  bin_path += ss.str();
  {
    // load size from file
    string str;
    std::ifstream ifs(bin_path + "size");
    getline(ifs, str);
    ifs.close();

    stringstream ss(str);
    size_t num_pts;
    ss >> num_pts;
    ptr->num_pts = num_pts;
    // cout << "size " << num_pts << endl;
  }

  if (cfg_ & DataType::GT) {
    ptr->rot_w_c = rot_[idx];
    ptr->t_w_c = trans_[idx];
  }
  //   ptr->mono = cv::

  if (cfg_ & DataType::Mono) {
    if (isPathExist(mono_file_[idx])) {
      ptr->mono = cv::imread(mono_file_[idx]);
    } else {
      ptr->mono = cv::Mat();
    }

    ptr->timestamp = time_stamp_[idx];
  }
  {
    // ptr->pts = readDepthBinary(bin_path + "pts.bin", ptr->num_pts, 3);

    // ptr->desc = readDepthBinary(bin_path + "desc.bin", ptr->num_pts, 256);

    // cout << global_desc_files[idx] << endl;
    ptr->global_desc = readDepthBinary(global_desc_files[idx], 1, 4096);

    if (cfg_ & DataType::Feat) {
      ptr->pts = readDepthBinary(bin_path + "pts.bin", ptr->num_pts, 3);
      // ptr->semi_half = readDepthBinary(bin_path + "dust_half.bin", 30, 40);
      ptr->semi = readDepthBinary(bin_path + "dust.bin", 60, 94);
      ptr->dense = readDepthBinary(bin_path + "dense.bin", 464, 736);
      ptr->dense_sm = readDepthBinary(bin_path + "dense_sm.bin", 480, 752);
      ptr->desc = readDepthBinary(bin_path + "desc_mat.bin", ptr->num_pts, 256);
    }
  }

  ptr->feature_extracted = true;

  ptr->idx = idx;
  return ptr;
}

void DataloaderEuRoCExport::loadImages(const string &base_path) {
  ifstream fTimes;
  string strPathTimeFile = base_path + "/cam0/data.csv";
  string strPrefixLeft = base_path + "/cam0/data/";
  string strPrefixRight = base_path + "/cam1/data/";

  fTimes.open(strPathTimeFile.c_str());
  string s;
  getline(fTimes, s);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      int index = s.find_first_of(",");
      string t = s.substr(0, index);

      time_stamp_.push_back(stod(t) / 10.0e8);
      mono_file_.push_back(strPrefixLeft + t + ".png");
      depth_file_.push_back(strPrefixRight + t + ".png");
    }
  }
}

void DataloaderEuRoCExport::loadTrajectory(const std::string &traj_file) {
  ifstream fs_mean(traj_file.c_str());
  string str_line;
  while (getline(fs_mean, str_line) && !fs_mean.eof()) {
    stringstream ss(str_line);
    double time, x, y, z, qx, qy, qz, qw;
    ss >> time;
    ss >> x;
    ss >> y;
    ss >> z;
    ss >> qx;
    ss >> qy;
    ss >> qz;
    ss >> qw;

    Eigen::Vector3d trans_tmp(x, y, z);
    Eigen::Quaterniond q_tmp(qw, qx, qy, qz);
    trans_.push_back(trans_tmp);
    rot_.push_back(q_tmp);
  }
}

DataloaderKITTIExport::DataloaderKITTIExport(const std::string &str,
                                             const std::string &seq,
                                             DataType cfg)
    : Dataloader(str, cfg) {

  // stringstream ss;
  // ss << setw(2) << setfill('0') << seq;

  seq_str = seq;

  if (!isPathExist(base_path_))
    throw std::runtime_error("base path not exists: " + base_path_);

  // TODO:
  // if (cfg_ & DataType::GT) {
  //   string traj_file = base_path_ + "/state_groundtruth_estimate0/traj.txt";

  //   if (!isPathExist(traj_file))
  //     throw std::runtime_error("traj_file not exists: " + traj_file);

  //   loadTrajectory(traj_file);
  //   cout << "load trajectory: " << traj_file << endl;
  //   num_ = rot_.size();
  // }

  if (cfg_ & DataType::Mono) {
    loadImages(base_path_);
    cout << "load images: " << base_path_ << endl;
    num_ = mono_file_.size();

    loadGlobalDesc(base_path_);
  }
}

void DataloaderKITTIExport::loadGlobalDesc(const std::string &base_path) {
  string strPrefixLeft = base_path_ + "/vlad/sequences/" + seq_str + '/';

  const int nTimes = time_stamp_.size();
  global_desc_files.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    global_desc_files[i] = strPrefixLeft + ss.str() + ".bin";
  }
}

DataFrame::Ptr DataloaderKITTIExport::getNextFrame() {
  if (idx_ >= num_)
    return nullptr;

  auto ptr = getFrameByIndex(idx_);

  idx_++;
  return ptr;
}

DataFrame::Ptr DataloaderKITTIExport::getFrameByIndex(size_t idx) {
  if (idx >= num_)
    return nullptr;

  auto ptr = DataFrame::Ptr(new DataFrame());

  string bin_path = base_path_ + "/features/";

  stringstream ss;
  ss << setw(6) << setfill('0') << idx << '/';
  bin_path += ss.str();

  // if (cfg_ & DataType::GT) {
  //   ptr->rot_w_c = rot_[idx];
  //   ptr->t_w_c = trans_[idx];
  // }
  //   ptr->mono = cv::

  // cout << mono_file_[idx] << endl;
  if (cfg_ & DataType::Mono) {
    if (isPathExist(mono_file_[idx])) {
      ptr->mono = cv::imread(mono_file_[idx]);
    } else {
      ptr->mono = cv::Mat();
    }

    ptr->timestamp = time_stamp_[idx];
  }

  // ptr->feature_extracted = true;

  ptr->idx = idx;
  return ptr;
}

void DataloaderKITTIExport::loadImages(const string &base_path) {
  ifstream fTimes;
  string strPathTimeFile =
      base_path_ + "/gray/sequences/" + seq_str + "/times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      time_stamp_.push_back(t);
    }
  }

  string strPrefixLeft =
      base_path_ + "/gray/sequences/" + seq_str + "/image_0/";

  const int nTimes = time_stamp_.size();
  mono_file_.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    mono_file_[i] = strPrefixLeft + ss.str() + ".png";
  }
}

void DataloaderKITTIExport::loadTrajectory(const std::string &traj_file) {
  // ifstream fs_mean(traj_file.c_str());
  // string str_line;
  // while (getline(fs_mean, str_line) && !fs_mean.eof()) {
  //   stringstream ss(str_line);
  //   double time, x, y, z, qx, qy, qz, qw;
  //   ss >> time;
  //   ss >> x;
  //   ss >> y;
  //   ss >> z;
  //   ss >> qx;
  //   ss >> qy;
  //   ss >> qz;
  //   ss >> qw;

  //   Eigen::Vector3d trans_tmp(x, y, z);
  //   Eigen::Quaterniond q_tmp(qw, qx, qy, qz);
  //   trans_.push_back(trans_tmp);
  //   rot_.push_back(q_tmp);
  // }
}

DataloaderTsukuba::DataloaderTsukuba(const std::string &str,
                                     const std::string &seq, DataType cfg)
    : Dataloader(str, cfg) {

  num_ = 1724;

  if (!isPathExist(base_path_)) {
    throw std::runtime_error("base path not exists: " + base_path_);
  }

  if (cfg_ & DataType::GT) {
  }

  if (cfg_ & DataType::Mono) {
    loadImages(base_path_ + '/' + seq);
    cout << "load images: " << base_path_ << endl;
    num_ = mono_file_.size();
  }
}

DataFrame::Ptr DataloaderTsukuba::getNextFrame() {
  if (idx_ >= num_) {
    LOG(INFO) << "return nullptr";
    return nullptr;
  }
  LOG(INFO) << "retrieval idx: " << idx_;

  auto ptr = getFrameByIndex(idx_);

  idx_++;
  return ptr;
}

DataFrame::Ptr DataloaderTsukuba::getFrameByIndex(size_t idx) {
  if (idx >= num_)
    return nullptr;

  auto ptr = DataFrame::Ptr(new DataFrame());

  if (cfg_ & DataType::Mono) {
    if (isPathExist(mono_file_[idx])) {
      ptr->mono = cv::imread(mono_file_[idx]);
    } else {
      ptr->mono = cv::Mat();
    }

    ptr->timestamp = time_stamp_[idx];
  }

  ptr->idx = idx;
  return ptr;
}

void DataloaderTsukuba::loadImages(const string &path) {
  time_stamp_.resize(num_);
  mono_file_.resize(num_);
  for (size_t i = 0; i < num_; i++) {

    stringstream ss;
    ss << "/left/frame_" << i + 1 << ".png";

    time_stamp_[i] = i * 0.05;
    mono_file_[i] = path + '/' + ss.str();
  }
}

DataloaderEuRoC::DataloaderEuRoC(const std::string &str, DataType cfg)
    : Dataloader(str, cfg) {

  if (!isPathExist(base_path_)) {
    throw std::runtime_error("base path not exists: " + base_path_);
  }

  if (cfg_ & DataType::GT) {
    // string traj_file = base_path_ + "/state_groundtruth_estimate0/traj.txt";

    // if (!isPathExist(traj_file))
    //   throw std::runtime_error("traj_file not exists: " + traj_file);

    // loadTrajectory(traj_file);
    // cout << "load trajectory: " << traj_file << endl;
    // num_ = rot_.size();
  }

  if (cfg_ & DataType::Mono) {
    cout << "load images: " << base_path_ << endl;
    loadImages(base_path_);
    num_ = mono_file_.size();
    cout << num_ << endl;
  }

  // if (common::dataset == "euroc")
  // pre-rectification
  {
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = 458.654;
    K.at<float>(1, 1) = 457.296;
    K.at<float>(0, 2) = 367.215;
    K.at<float>(1, 2) = 248.375;

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = -0.28340811;
    DistCoef.at<float>(1) = 0.07395907;
    DistCoef.at<float>(2) = 0.00019359;
    DistCoef.at<float>(3) = 1.76187114e-05;

    cv::Mat Knew =
        cv::getOptimalNewCameraMatrix(K, DistCoef, cv::Size(752, 480), 0);

    cv::initUndistortRectifyMap(K, DistCoef, cv::Mat(), Knew,
                                cv::Size(752, 480), CV_32FC1, m1, m2);
    // cv::getOptimalNewCameraMatrix(K, D, (752, 480), 0);
  }
}

DataFrame::Ptr DataloaderEuRoC::getNextFrame() {
  if (idx_ >= num_) {
    LOG(INFO) << "return nullptr";
    return nullptr;
  }
  LOG(INFO) << "retrieval idx: " << idx_;

  auto ptr = getFrameByIndex(idx_);

  idx_++;
  return ptr;
}

DataFrame::Ptr DataloaderEuRoC::getFrameByIndex(size_t idx) {
  if (idx >= num_)
    return nullptr;

  auto ptr = DataFrame::Ptr(new DataFrame());

  if (cfg_ & DataType::GT) {
    // ptr->rot_w_c = rot_[idx];
    // ptr->t_w_c = trans_[idx];
  }
  //   ptr->mono = cv::

  if (cfg_ & DataType::Mono) {
    if (isPathExist(mono_file_[idx])) {
      ptr->mono = cv::imread(mono_file_[idx]);

      cv::remap(ptr->mono, ptr->mono, m1, m2, cv::INTER_LINEAR);
    } else {
      ptr->mono = cv::Mat();
    }

    ptr->timestamp = time_stamp_[idx];
  }

  ptr->idx = idx;
  return ptr;
}

void DataloaderEuRoC::loadImages(const string &base_path) {
  ifstream fTimes;
  string strPathTimeFile = base_path + "/cam0/data.csv";
  string strPrefixLeft = base_path + "/cam0/data/";
  string strPrefixRight = base_path + "/cam1/data/";

  fTimes.open(strPathTimeFile.c_str());
  string s;
  getline(fTimes, s);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      int index = s.find_first_of(",");
      string t = s.substr(0, index);

      time_stamp_.push_back(stod(t) / 10.0e8);
      mono_file_.push_back(strPrefixLeft + t + ".png");
      depth_file_.push_back(strPrefixRight + t + ".png");
      // cout << strPrefixLeft + t + ".png" << endl;
    }
  }
}

} // namespace orbslam
