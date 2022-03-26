#pragma once

#include <memory>

#include <string>

#include <vector>
#include <deque>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lidar_localization/sensor_data/key_frame.hpp"
#include "lidar_localization/sensor_data/speed_bias.hpp"

#include <ceres/ceres.h>

#include "lidar_localization/models/ceres_back_end/params/param_pr.hpp"

#include "lidar_localization/models/ceres_back_end/factors/factor_map_matching_pose.hpp"
#include "lidar_localization/models/ceres_back_end/factors/factor_relative_pose.hpp"
#include "lidar_localization/models/ceres_back_end/factors/factor_imu_pre_integration.hpp"
#include "lidar_localization/models/pre_integrator/imu_pre_integrator.hpp"

namespace lidar_localization {

class CeresBackEnd {
public:

    struct OptimizedKeyFrame {
      double time;
      double pr[7];
      bool fixed = false;
    };
    
    struct OptimizedSpeedBias {
      double time;
      double vag[9];
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

    
    struct ResidualIMUPreIntegration {
      int param_index_i;
      int param_index_j;

      int speedbias_index_i;
      int speedbias_index_j;

      double T;

      Eigen::Vector3d g;
      Eigen::MatrixXd I;
      Eigen::MatrixXd J;

      Eigen::Vector3d alpha;
      Eigen::Vector3d beta;
      Eigen::Quaterniond theta;
    };

    CeresBackEnd(const int N);
    ~CeresBackEnd();

    void AddPRParam(const KeyFrame &lio_key_frame, const bool fixed);
    void AddVAGParam(const SpeedBias &speed_bias, const bool fixed);
    
    void AddRelativePoseFactor(const int param_index_i, const int param_index_j, const Eigen::Matrix4d &relative_pose, const Eigen::VectorXd &noise);
    void AddMapMatchingPoseFactor(const int param_index, const Eigen::Matrix4d &prior_pose, const Eigen::VectorXd &noise);

    void AddIMUPreIntegrationFactor(const int param_index_i, const int param_index_j, const int speedbias_index_i, const int speedbias_index_j,
                                    const IMUPreIntegrator::IMUPreIntegration &imu_pre_integration);
    
    bool Optimize();
    int  GetNumParamBlocks();
    bool GetLatestOptimizedKeyFrame(KeyFrame &optimized_key_frame);
    bool GetOptimizedKeyFrames(std::deque<KeyFrame> &optimized_key_frames);

private:
    Eigen::MatrixXd GetInformationMatrix(Eigen::VectorXd noise);

    const int kWindowSize;

    struct {
      std::unique_ptr<ceres::LossFunction> loss_function_ptr;
      ceres::Solver::Options options;
    } config_;

    std::vector<OptimizedKeyFrame>  optimized_key_frames_;
    std::vector<OptimizedSpeedBias> optimized_speed_bias_;

    FactorMapMatchingPose   *GetResMapMatchingPose(const ResidualMapMatchingPose &res_map_matching_pose);
    FactorRelativePose      *GetResRelativePose(const ResidualRelativePose &res_relative_pose);
    FactorIMUPreIntegration *GetResIMUPreIntegration(const ResidualIMUPreIntegration &res_imu_pre_integration);

    struct {
      std::deque<ResidualMapMatchingPose> map_matching_pose;
      std::deque<ResidualRelativePose> relative_pose;
      std::deque<ResidualIMUPreIntegration> imu_pre_integration;
    } residual_blocks_;
};

} // namespace lidar_localization