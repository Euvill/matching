#pragma once

#include <memory>

#include <string>

#include <vector>
#include <deque>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lidar_localization/sensor_data/key_frame.hpp"

#include <ceres/ceres.h>

#include "lidar_localization/models/ceres_back_end/params/param_pr.hpp"

#include "lidar_localization/models/ceres_back_end/factors/factor_map_matching_pose.hpp"
#include "lidar_localization/models/ceres_back_end/factors/factor_relative_pose.hpp"

namespace lidar_localization {

class CeresBackEnd {
public:

    struct OptimizedKeyFrame {
      double time;
      double pr[7];
      bool fixed = false;
    };
    
    struct ResidualMapMatchingPose {
      int param_index;

      Eigen::Vector3d    m_pos;
      Eigen::Quaterniond m_ori;
      Eigen::MatrixXd I;
    };

    struct ResidualRelativePose {
      int param_index_i;
      int param_index_j;

      Eigen::Vector3d    m_pos;
      Eigen::Quaterniond m_ori;
      Eigen::MatrixXd I;
    };

    CeresBackEnd(const int N);
    ~CeresBackEnd();

    /**
     * @brief  add parameter block for LIO key frame
     * @param  lio_key_frame, LIO key frame with (pos, ori, vel, b_a and b_g)
     * @param  fixed, shall the param block be fixed to eliminate trajectory estimation ambiguity
     * @return true if success false otherwise
     */
    void AddPRParam(const KeyFrame &lio_key_frame, const bool fixed);

    /**
     * @brief  add residual block for relative pose constraint from lidar frontend
     * @param  param_index_i, param block ID of previous key frame
     * @param  param_index_j, param block ID of current key frame
     * @param  relative_pose, relative pose measurement
     * @param  noise, relative pose measurement noise
     * @return void
     */
    void AddRelativePoseFactor(const int param_index_i, const int param_index_j, const Eigen::Matrix4d &relative_pose, const Eigen::VectorXd &noise);

    /**
     * @brief  add residual block for prior pose constraint from map matching
     * @param  param_index, param block ID of current key frame
     * @param  prior_pose, prior pose measurement
     * @param  noise, prior pose measurement noise
     * @return void
     */
    void AddMapMatchingPoseFactor(const int param_index, const Eigen::Matrix4d &prior_pose, const Eigen::VectorXd &noise);

    // do optimization
    bool Optimize();

    // get num. of parameter blocks:
    int GetNumParamBlocks();

    /**
     * @brief  get optimized odometry estimation
     * @param  optimized_key_frame, output latest optimized key frame
     * @return true if success false otherwise
     */
    bool GetLatestOptimizedKeyFrame(KeyFrame &optimized_key_frame);

    /**
     * @brief  get optimized LIO key frame state estimation
     * @param  optimized_key_frames, output optimized LIO key frames
     * @return true if success false otherwise
     */
    bool GetOptimizedKeyFrames(std::deque<KeyFrame> &optimized_key_frames);

private:
    /**
     * @brief  create information matrix from measurement noise specification
     * @param  noise, measurement noise covariances
     * @return information matrix as square Eigen::MatrixXd
     */
    Eigen::MatrixXd GetInformationMatrix(Eigen::VectorXd noise);

    const int kWindowSize;

    struct {
      std::unique_ptr<ceres::LossFunction> loss_function_ptr;
      ceres::Solver::Options options;
    } config_;

    std::vector<OptimizedKeyFrame> optimized_key_frames_;

    FactorMapMatchingPose *GetResMapMatchingPose(const ResidualMapMatchingPose &res_map_matching_pose);
    FactorRelativePose    *GetResRelativePose(const ResidualRelativePose &res_relative_pose);

    struct {
      std::deque<ResidualMapMatchingPose> map_matching_pose;
      std::deque<ResidualRelativePose> relative_pose;
    } residual_blocks_;
};

} // namespace lidar_localization