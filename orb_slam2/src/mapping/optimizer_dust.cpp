#include "orb_slam/mapping/optimizer.h"

#include <mutex>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "orb_slam/optimization/types_dust_tracking.h"

#include <Eigen/StdVector>

#include "orb_slam/config.h"
#include "orb_slam/global.h"
#include "orb_slam/utils/converter.h"

#include <glog/logging.h>

namespace orbslam {

using namespace std;

template <typename T> void saveVector(string file_name, const vector<T> &vec) {
  ofstream fs(file_name.c_str());
  for (auto &&e : vec) {
    fs << e << endl;
  }

  fs.close();
}

int Optimizer::PoseOptimizationDustPost(Frame *pFrame) {
  g2o::SparseOptimizer optimizer;

  g2o::OptimizationAlgorithmLevenberg *solver;
  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  int nInitialCorrespondences = 0;

  // Set Frame vertex
  // 步骤2：添加顶点：待优化当前帧的Tcw
  g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = pFrame->N;

  // for Monocular
  vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vector<float> chi2_mono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  const float deltaMono = sqrt(5.991);

  //   步骤3：添加一元边：相机投影模型
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint *pMP = pFrame->mvpMapPoints[i];
      if (pMP) {
        // Monocular observation
        nInitialCorrespondences++;
        pFrame->mvbOutlier[i] = false;

        Eigen::Matrix<double, 2, 1> obs;
        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectXYZOnlyPose *e =
            new g2o::EdgeSE3ProjectXYZOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(0)));
        e->setMeasurement(obs);

        {
          const auto inv_sigma2 = pFrame->cov2_inv_[i];
          Eigen::Matrix2d info = Eigen::Matrix2d::Identity();
          info(0, 0) *= inv_sigma2.x();
          info(1, 1) *= inv_sigma2.y();
          e->setInformation(info);
        }

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(deltaMono);

        e->fx = pFrame->fx;
        e->fy = pFrame->fy;
        e->cx = pFrame->cx;
        e->cy = pFrame->cy;
        cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vnIndexEdgeMono.push_back(i);
      }
    }
  }

  if (nInitialCorrespondences < 3)
    return 0;

  chi2_mono.resize(vpEdgesMono.size());

  const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
  // const float chi2Mono[4] = {7.378, 7.378, 7.378, 7.378};
  // const float chi2Mono[4] = {9.21, 9.21, 9.21, 9.21};

  int nBad = 0;
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  optimizer.initializeOptimization(0); // 对level为0的边进行优化
  optimizer.optimize(10);
  for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
    g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

    const size_t idx = vnIndexEdgeMono[i];

    // if (pFrame->mvbOutlier[idx]) {
    e->computeError();
    // }

    const float chi2 = e->chi2();
    chi2_mono[i] = chi2;
    // TODO: use histogram to filter outlier
    if (chi2 > 7.378) {
      pFrame->mvbOutlier[idx] = true;
      e->setLevel(1);
      nBad++;
    } else {
      pFrame->mvbOutlier[idx] = false;
      e->setLevel(0);
    }

    // if (it == 2)
    e->setRobustKernel(0);
  }
  optimizer.initializeOptimization(0);
  optimizer.optimize(10);

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vSE3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  pFrame->SetPose(pose);

  return nInitialCorrespondences - nBad;

  //   return 1;
}

int Optimizer::PoseOptimizationDust(Frame *pFrame,
                                    const std::vector<MapPoint *> &mps,
                                    std::vector<bool> &is_visible) {
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg *solver;

  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  // int nInitialCorrespondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = mps.size();

  // for Monocular
  vector<g2o::EdgeSE3ProjectDustOnlyPose *> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  int n_to_opt = mps.size();
  // auto &&mps = pLastFrame->GetMapPointMatches();
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint *pMP = mps[i];
      // if (pMP && (!pMP->isBad())) {
      // n_to_opt++;

      g2o::EdgeSE3ProjectDustOnlyPose *e =
          new g2o::EdgeSE3ProjectDustOnlyPose();

      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                          optimizer.vertex(0)));

      // TODO: weight the information by confidence
      e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

      g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
      e->setRobustKernel(rk);
      rk->setDelta(0.9);

      e->fx = pFrame->fx / 8.0f;
      e->fy = pFrame->fy / 8.0f;
      e->cx = (pFrame->cx - 3.5) / 8.0f;
      e->cy = (pFrame->cy - 3.5) / 8.0f;

      e->setDustData(&(pFrame->dust_));

      cv::Mat Xw = pMP->GetWorldPos();
      e->Xw[0] = Xw.at<float>(0);
      e->Xw[1] = Xw.at<float>(1);
      e->Xw[2] = Xw.at<float>(2);

      optimizer.addEdge(e);

      vpEdgesMono.push_back(e);
      vnIndexEdgeMono.push_back(i);
    }
    // }
  }
  // cout << "# points for opt " << n_to_opt << endl;

  optimizer.initializeOptimization(0);
  // optimizer.optimize(1);
  const int iter = 40;
  // for (size_t i = 0; i < iter; i++) {
  //   optimizer.optimize(1);
  //   cout << "error: " << optimizer.chi2() << endl;
  // }
  const int n_iter_final = optimizer.optimize(iter);

  int n_inlier = n_to_opt;
  for (size_t i = 0; i < vpEdgesMono.size(); i++) {
    const auto &e = vpEdgesMono[i];
    if (e->level() == 1 || e->chi2() > 0.9)
    // if (e->level() == 1)
    {
      n_inlier--;
    } else {
      auto idx = vnIndexEdgeMono[i];
      is_visible[idx] = true;
      mps[idx]->in_view = true;
      mps[idx]->dust_proj_u = e->u_;
      mps[idx]->dust_proj_v = e->v_;
      mps[idx]->dust_proj_v = e->v_;
    }
  }

  if (common::verbose) {
    LOG(INFO) << "#iter: " << n_iter_final << " #to_opt: " << n_to_opt
              << " #inliers: " << n_inlier;
  }

  // for (const auto &e : vpEdgesMono) {
  //   if (e->level() == 1 || e->chi2() > 0.9)
  //   // if (e->level() == 1)
  //   {
  //     n_inlier--;
  //   }
  // }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vSE3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  // cout << pose << endl;
  pFrame->SetPose(pose);

  // return nInitialCorrespondences - nBad;

  return n_inlier;
}

int Optimizer::PoseOptimizationDust(Frame *pFrame, KeyFrame *pLastFrame) {
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg *solver;

  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  // int nInitialCorrespondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = pLastFrame->N;

  // for Monocular
  vector<g2o::EdgeSE3ProjectDustOnlyPose *> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  int n_to_opt = 0;
  auto &&mps = pLastFrame->GetMapPointMatches();
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint *pMP = mps[i];
      if (pMP && (!pMP->isBad())) {
        n_to_opt++;

        g2o::EdgeSE3ProjectDustOnlyPose *e =
            new g2o::EdgeSE3ProjectDustOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(0)));

        // TODO: weight the information by confidence
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(0.9);

        e->fx = pFrame->fx / 8.0f;
        e->fy = pFrame->fy / 8.0f;
        e->cx = (pFrame->cx - 3.5) / 8.0f;
        e->cy = (pFrame->cy - 3.5) / 8.0f;

        e->setDustData(&(pFrame->dust_));

        cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vnIndexEdgeMono.push_back(i);
      }
    }
  }
  // cout << "# points for opt " << n_to_opt << endl;

  optimizer.initializeOptimization(0);
  // optimizer.optimize(1);
  const int iter = 40;
  // for (size_t i = 0; i < iter; i++) {
  //   optimizer.optimize(1);
  //   cout << "error: " << optimizer.chi2() << endl;
  // }
  const int n_iter_final = optimizer.optimize(iter);

  int n_inlier = n_to_opt;
  for (size_t i = 0; i < vpEdgesMono.size(); i++) {
    const auto &e = vpEdgesMono[i];
    if (e->level() == 1 || e->chi2() > 0.9)
    // if (e->level() == 1)
    {
      n_inlier--;
    } else {
      pLastFrame->is_mp_visible_[vnIndexEdgeMono[i]] = true;
    }
  }

  if (common::verbose) {
    LOG(INFO) << "#iter: " << n_iter_final << " #to_opt: " << n_to_opt
              << " #inliers: " << n_inlier;
  }

  // for (const auto &e : vpEdgesMono) {
  //   if (e->level() == 1 || e->chi2() > 0.9)
  //   // if (e->level() == 1)
  //   {
  //     n_inlier--;
  //   }
  // }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vSE3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  // cout << pose << endl;
  pFrame->SetPose(pose);

  // return nInitialCorrespondences - nBad;

  return n_inlier;
}

int Optimizer::PoseOptimizationHeat(Frame *pFrame, Frame *pLastFrame) {
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg *solver;

  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  // int nInitialCorrespondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = pLastFrame->N;

  // for Monocular
  vector<g2o::EdgeSE3ProjectDustOnlyPose *> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  // const float deltaMono = sqrt(5.991);
  // const float deltaStereo = sqrt(7.815);

  int n_to_opt = 0;
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint *pMP = pLastFrame->mvpMapPoints[i];
      if (pMP && (!pMP->isBad())) {
        n_to_opt++;
        // pFrame->mvbOutlier[i] = false;

        // Eigen::Matrix<double, 2, 1> obs;
        // const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
        // obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectDustOnlyPose *e =
            new g2o::EdgeSE3ProjectDustOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(0)));

        // TODO: weight the information by confidence
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(0.9);

        e->fx = pFrame->fx;
        e->fy = pFrame->fy;
        e->cx = pFrame->cx;
        e->cy = pFrame->cy;

        e->setDustData(&(pFrame->heat_));

        cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vnIndexEdgeMono.push_back(i);
      }
    }
  }
  cout << "# points for opt " << n_to_opt << endl;

  optimizer.initializeOptimization(0);
  // optimizer.optimize(1);
  const int iter = 40;
  // for (size_t i = 0; i < iter; i++) {
  //   optimizer.optimize(1);
  //   cout << "error: " << optimizer.chi2() << endl;
  // }
  const int what = optimizer.optimize(iter);
  cout << "heat: " << what << endl;

  int n_inlier = n_to_opt;
  for (const auto &e : vpEdgesMono) {
    if (e->level() == 1 || e->chi2() > 0.02) {
      n_inlier--;
    }
  }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vSE3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  // cout << pose << endl;
  pFrame->SetPose(pose);

  // return nInitialCorrespondences - nBad;

  return n_inlier;
}

int Optimizer::PoseOptimizationDust(Frame *pFrame,
                                    const vector<MapPoint *> &mps) {
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg *solver;

  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  // int nInitialCorrespondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = mps.size();

  // for Monocular
  vector<g2o::EdgeSE3ProjectDustOnlyPose *> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  // const float deltaMono = sqrt(5.991);
  // const float deltaStereo = sqrt(7.815);

  int n_to_opt = 0;
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint *pMP = mps[i];
      if (pMP && (!pMP->isBad())) {
        n_to_opt++;
        // pFrame->mvbOutlier[i] = false;

        // Eigen::Matrix<double, 2, 1> obs;
        // const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
        // obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectDustOnlyPose *e =
            new g2o::EdgeSE3ProjectDustOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(0)));

        // TODO: weight the information by confidence
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(0.9);

        e->fx = pFrame->fx / 8.0f;
        e->fy = pFrame->fy / 8.0f;
        e->cx = (pFrame->cx - 3.5) / 8.0f;
        e->cy = (pFrame->cy - 3.5) / 8.0f;

        e->setDustData(&(pFrame->dust_));

        cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vnIndexEdgeMono.push_back(i);
      }
    }
  }
  cout << "# points for opt " << n_to_opt << endl;

  optimizer.initializeOptimization(0);
  // optimizer.optimize(1);
  const int iter = 40;
  // for (size_t i = 0; i < iter; i++) {
  //   optimizer.optimize(1);
  //   cout << "error: " << optimizer.chi2() << endl;
  // }
  int what = optimizer.optimize(40);
  cout << "dust " << what << endl;

  int n_inlier = n_to_opt;
  for (const auto &e : vpEdgesMono) {
    if (e->level() == 1 || e->chi2() > 0.9) {
      n_inlier--;
    }
  }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vSE3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  // cout << pose << endl;
  pFrame->SetPose(pose);

  return n_inlier;
}

int Optimizer::PoseOptimizationDust(Frame *pFrame, Frame *pLastFrame) {
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg *solver;

  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  // int nInitialCorrespondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = pLastFrame->N;

  // for Monocular
  vector<g2o::EdgeSE3ProjectDustOnlyPose *> vpEdgesMono;
  vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  // const float deltaMono = sqrt(5.991);
  // const float deltaStereo = sqrt(7.815);

  int n_to_opt = 0;
  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (int i = 0; i < N; i++) {
      MapPoint *pMP = pLastFrame->mvpMapPoints[i];
      if (pMP && (!pMP->isBad())) {
        n_to_opt++;
        // pFrame->mvbOutlier[i] = false;

        // Eigen::Matrix<double, 2, 1> obs;
        // const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
        // obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectDustOnlyPose *e =
            new g2o::EdgeSE3ProjectDustOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(0)));

        // TODO: weight the information by confidence
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(0.9);

        e->fx = pFrame->fx / 8.0f;
        e->fy = pFrame->fy / 8.0f;
        e->cx = (pFrame->cx - 3.5) / 8.0f;
        e->cy = (pFrame->cy - 3.5) / 8.0f;

        e->setDustData(&(pFrame->dust_));

        cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vnIndexEdgeMono.push_back(i);
      }
    }
  }
  // cout << "# points for opt " << n_to_opt << endl;

  optimizer.initializeOptimization(0);
  // optimizer.optimize(1);
  const int iter = 40;
  // for (size_t i = 0; i < iter; i++) {
  //   optimizer.optimize(1);
  //   cout << "error: " << optimizer.chi2() << endl;
  // }
  const int n_iter_final = optimizer.optimize(iter);

  // if (nInitialCorrespondences < 3)
  //   return 0;


  //   for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
  //     g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

  //     const size_t idx = vnIndexEdgeStereo[i];

  //     if (pFrame->mvbOutlier[idx]) {
  //       e->computeError();
  //     }

  //     const float chi2 = e->chi2();

  //     if (chi2 > chi2Stereo[it]) {
  //       pFrame->mvbOutlier[idx] = true;
  //       e->setLevel(1);
  //       nBad++;
  //     } else {
  //       e->setLevel(0);
  //       pFrame->mvbOutlier[idx] = false;
  //     }

  //     if (it == 2)
  //       e->setRobustKernel(0);
  //   }

  //   if (optimizer.edges().size() < 10)
  //     break;
  // }
  int n_inlier = n_to_opt;
  for (size_t i = 0; i < vpEdgesMono.size(); i++) {
    const auto &e = vpEdgesMono[i];
    if (e->level() == 1 || e->chi2() > 0.9)
    // if (e->level() == 1)
    {
      n_inlier--;
    } else {
      pLastFrame->is_mp_visible_[vnIndexEdgeMono[i]] = true;
    }
  }

  if (common::verbose) {
    LOG(INFO) << "#iter: " << n_iter_final << " #to_opt: " << n_to_opt
              << " #inliers: " << n_inlier;
  }

  // for (const auto &e : vpEdgesMono) {
  //   if (e->level() == 1 || e->chi2() > 0.9)
  //   // if (e->level() == 1)
  //   {
  //     n_inlier--;
  //   }
  // }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vSE3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  // cout << pose << endl;
  pFrame->SetPose(pose);

  // return nInitialCorrespondences - nBad;

  return n_inlier;
}

} // namespace orbslam
