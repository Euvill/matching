#include "lidar_localization/models/ceres_back_end/ceres_back_end.hpp"

#include <chrono>

#include "glog/logging.h"

namespace lidar_localization {

CeresBackEnd::CeresBackEnd(const int N) : kWindowSize(N) 
{   
    config_.loss_function_ptr = std::make_unique<ceres::CauchyLoss>(1.0);

    config_.options.linear_solver_type = ceres::DENSE_SCHUR;
    // config_.options.use_explicit_schur_complement = true;
    config_.options.trust_region_strategy_type = ceres::DOGLEG;
    // config_.options.use_nonmonotonic_steps = true;
    config_.options.num_threads = 2;
    config_.options.max_num_iterations = 1000;
    config_.options.max_solver_time_in_seconds = 0.10;
    // config_.options.minimizer_progress_to_stdout = true;

    optimized_key_frames_.clear();
    optimized_speed_bias_.clear();

    residual_blocks_.relative_pose.clear();
    residual_blocks_.map_matching_pose.clear();
}

CeresBackEnd::~CeresBackEnd() {
}

void CeresBackEnd::AddPRParam(const KeyFrame &lio_key_frame, const bool fixed) {
    OptimizedKeyFrame optimized_key_frame;

    optimized_key_frame.time = lio_key_frame.time;
    optimized_key_frame.fixed = fixed;

    Eigen::Map<Eigen::Vector3d>    pos(optimized_key_frame.pr + 0);
    Eigen::Map<Eigen::Quaterniond> ori(optimized_key_frame.pr + 3);

    pos = lio_key_frame.pose.block<3, 1>(0, 3).cast<double>();
    ori = Eigen::Quaterniond(lio_key_frame.pose.block<3, 3>(0, 0).cast<double>());

    optimized_key_frames_.push_back(optimized_key_frame);
}

void CeresBackEnd::AddVAGParam(const SpeedBias &speed_bias, const bool fixed) {
    OptimizedSpeedBias optimized_speed_bias;

    optimized_speed_bias.time  = speed_bias.time;
    optimized_speed_bias.fixed = fixed;

    Eigen::Map<Eigen::Vector3d> vel(optimized_speed_bias.vag + 0);
    Eigen::Map<Eigen::Vector3d> ba(optimized_speed_bias.vag + 3);
    Eigen::Map<Eigen::Vector3d> bg(optimized_speed_bias.vag + 6);

    vel = speed_bias.vel;
    ba  = speed_bias.ba;
    bg  = speed_bias.bg;

    optimized_speed_bias_.push_back(optimized_speed_bias);
}

void CeresBackEnd::AddRelativePoseFactor(const int param_index_i, const int param_index_j, const Eigen::Matrix4d &relative_pose, const Eigen::VectorXd &noise) {
    ResidualRelativePose residual_relative_pose;

    residual_relative_pose.param_index_i = param_index_i;
    residual_relative_pose.param_index_j = param_index_j;
    residual_relative_pose.m_pos = relative_pose.block<3, 1>(0, 3);
    residual_relative_pose.m_ori = Eigen::Quaterniond(relative_pose.block<3, 3>(0, 0).cast<double>());
    residual_relative_pose.I = GetInformationMatrix(noise);

    residual_blocks_.relative_pose.push_back(residual_relative_pose);
}

void CeresBackEnd::AddMapMatchingPoseFactor(const int param_index, const Eigen::Matrix4d &prior_pose, const Eigen::VectorXd &noise) {
    ResidualMapMatchingPose residual_map_matching_pose;

    residual_map_matching_pose.param_index = param_index;
    residual_map_matching_pose.m_pos = prior_pose.block<3, 1>(0, 3);
    residual_map_matching_pose.m_ori = Eigen::Quaterniond(prior_pose.block<3, 3>(0, 0).cast<double>());
    residual_map_matching_pose.I = GetInformationMatrix(noise);

    residual_blocks_.map_matching_pose.push_back(residual_map_matching_pose);
}

void CeresBackEnd::AddIMUPreIntegrationFactor(const int param_index_i, const int param_index_j, const int speedbias_index_i, const int speedbias_index_j,
                                              const IMUPreIntegrator::IMUPreIntegration &imu_pre_integration) {
    ResidualIMUPreIntegration residual_imu_pre_integration;

    residual_imu_pre_integration.param_index_i = param_index_i;
    residual_imu_pre_integration.param_index_j = param_index_j;

    residual_imu_pre_integration.speedbias_index_i = speedbias_index_i;
    residual_imu_pre_integration.speedbias_index_j = speedbias_index_j;

    residual_imu_pre_integration.T = imu_pre_integration.GetT();
    residual_imu_pre_integration.g = imu_pre_integration.GetGravity();

    residual_imu_pre_integration.alpha = imu_pre_integration.GetMeasurement_alpha();
    residual_imu_pre_integration.beta  = imu_pre_integration.GetMeasurement_beta();
    residual_imu_pre_integration.theta = imu_pre_integration.GetMeasurement_theta();

    residual_imu_pre_integration.I = imu_pre_integration.GetInformation();
    residual_imu_pre_integration.J = imu_pre_integration.GetJacobian();

    // add to data buffer:
    residual_blocks_.imu_pre_integration.push_back(residual_imu_pre_integration);
}

FactorMapMatchingPose *CeresBackEnd::GetResMapMatchingPose(const CeresBackEnd::ResidualMapMatchingPose &res_map_matching_pose) {
    FactorMapMatchingPose *factor_map_matching_pose = new FactorMapMatchingPose();
    
    factor_map_matching_pose->SetMeasurement(res_map_matching_pose.m_pos, res_map_matching_pose.m_ori);
    factor_map_matching_pose->SetInformation(res_map_matching_pose.I);

    return factor_map_matching_pose;
}

FactorRelativePose *CeresBackEnd::GetResRelativePose(const CeresBackEnd::ResidualRelativePose &res_relative_pose) {
    FactorRelativePose *factor_relative_pose = new FactorRelativePose();
    
    factor_relative_pose->SetMeasurement(res_relative_pose.m_pos, res_relative_pose.m_ori);
    factor_relative_pose->SetInformation(res_relative_pose.I);   

    return factor_relative_pose;
}

FactorIMUPreIntegration *CeresBackEnd::GetResIMUPreIntegration(const CeresBackEnd::ResidualIMUPreIntegration &res_imu_pre_integration) {
    FactorIMUPreIntegration *factor_imu_pre_integration = new FactorIMUPreIntegration();

    factor_imu_pre_integration->SetT(res_imu_pre_integration.T);
    factor_imu_pre_integration->SetGravitiy(res_imu_pre_integration.g);
    factor_imu_pre_integration->SetMeasurement(res_imu_pre_integration.alpha, res_imu_pre_integration.theta, res_imu_pre_integration.beta);
    factor_imu_pre_integration->SetInformation(res_imu_pre_integration.I);
    factor_imu_pre_integration->SetJacobian(res_imu_pre_integration.J);   

    return factor_imu_pre_integration;
}


bool CeresBackEnd::Optimize() {
    static int optimization_count = 0;
    
    const int N = GetNumParamBlocks();

    if (kWindowSize + 1 <= N){
        ceres::Problem problem;

        for (int i = 1; i <= kWindowSize + 1; ++i) {
            auto &target_key_frame  = optimized_key_frames_.at(N - i);
            auto &target_speed_bias = optimized_speed_bias_.at(N - i);

            ceres::LocalParameterization *local_parameterization = new ParamPR();

            problem.AddParameterBlock(target_key_frame.pr, 7, local_parameterization);
            problem.AddParameterBlock(target_speed_bias.vag, 9);

            if (target_key_frame.fixed) {
                problem.SetParameterBlockConstant(target_key_frame.pr);
                problem.SetParameterBlockConstant(target_key_frame.pr);
            }
        }
        
        if (!residual_blocks_.map_matching_pose.empty() && !residual_blocks_.relative_pose.empty() && !residual_blocks_.imu_pre_integration.empty()) {
            auto &key_frame_m = optimized_key_frames_.at(N - kWindowSize - 1);
            auto &key_frame_r = optimized_key_frames_.at(N - kWindowSize - 0);

            auto &speed_bias_m = optimized_speed_bias_.at(N - kWindowSize - 1);
            auto &speed_bias_r = optimized_speed_bias_.at(N - kWindowSize - 0);

            const ceres::CostFunction *factor_map_matching_pose   = GetResMapMatchingPose(residual_blocks_.map_matching_pose.front());
            const ceres::CostFunction *factor_relative_pose       = GetResRelativePose(residual_blocks_.relative_pose.front());
            const ceres::CostFunction *factor_imu_pre_integration = GetResIMUPreIntegration(residual_blocks_.imu_pre_integration.front());

            FactorMarginalization *factor_marginalization = new FactorMarginalization();

            factor_marginalization->SetResMapMatchingPose(factor_map_matching_pose, std::vector<double *>{key_frame_m.pr});
            factor_marginalization->SetResRelativePose(factor_relative_pose, std::vector<double *>{key_frame_m.pr, key_frame_r.pr});
            factor_marginalization->SetResIMUPreIntegration(factor_imu_pre_integration, std::vector<double *>{key_frame_m.pr, speed_bias_m.vag, key_frame_r.pr, speed_bias_r.vag});
            factor_marginalization->Marginalize(key_frame_r.pr, speed_bias_r.vag);

            // add marginalization factor into sliding window
            problem.AddResidualBlock(factor_marginalization, NULL, key_frame_r.pr, speed_bias_r.vag);

            residual_blocks_.map_matching_pose.pop_front();
            residual_blocks_.relative_pose.pop_front();
            residual_blocks_.imu_pre_integration.pop_front();
        }

        if (!residual_blocks_.map_matching_pose.empty()) {
            for (const auto &residual_map_matching_pose: residual_blocks_.map_matching_pose) {

                auto &key_frame = optimized_key_frames_.at(residual_map_matching_pose.param_index);

                FactorMapMatchingPose *factor_map_matching_pose = GetResMapMatchingPose(residual_map_matching_pose);

                problem.AddResidualBlock(factor_map_matching_pose, NULL, key_frame.pr);
            }            
        }

        if (!residual_blocks_.relative_pose.empty()) {
            for (const auto &residual_relative_pose: residual_blocks_.relative_pose) {

                auto &key_frame_i = optimized_key_frames_.at(residual_relative_pose.param_index_i);
                auto &key_frame_j = optimized_key_frames_.at(residual_relative_pose.param_index_j);

                FactorRelativePose *factor_relative_pose = GetResRelativePose(residual_relative_pose);

                problem.AddResidualBlock(factor_relative_pose, NULL, key_frame_i.pr, key_frame_j.pr);
            } 
        }

        if (!residual_blocks_.imu_pre_integration.empty()) {
            for (const auto &residual_imu_pre_integration: residual_blocks_.imu_pre_integration) {

                auto &key_frame_i  = optimized_key_frames_.at(residual_imu_pre_integration.param_index_i);
                auto &key_frame_j  = optimized_key_frames_.at(residual_imu_pre_integration.param_index_j);

                auto &speed_bias_i = optimized_speed_bias_.at(residual_imu_pre_integration.speedbias_index_i);
                auto &speed_bias_j = optimized_speed_bias_.at(residual_imu_pre_integration.speedbias_index_j);

                FactorIMUPreIntegration *factor_imu_pre_integration = GetResIMUPreIntegration(residual_imu_pre_integration);

                problem.AddResidualBlock(factor_imu_pre_integration, NULL, key_frame_i.pr, speed_bias_i.vag, key_frame_j.pr, speed_bias_j.vag);
            }
        }

        // solve:
        ceres::Solver::Summary summary;

        auto start = std::chrono::steady_clock::now();
        ceres::Solve(config_.options, &problem, &summary);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = end-start;

        // prompt:
        LOG(INFO) << "------ Finish Iteration " << optimization_count << " of Sliding Window Optimization -------" << std::endl
                << "Time Used: " << time_used.count() << " seconds." << std::endl
                << "Cost Reduced: " << summary.initial_cost - summary.final_cost << std::endl
                << summary.BriefReport() << std::endl
                << std::endl;

        return true;
    }

    return false;
 
}

int CeresBackEnd::GetNumParamBlocks() {
    return static_cast<int>(optimized_key_frames_.size());
}

bool CeresBackEnd::GetLatestOptimizedKeyFrame(KeyFrame &optimized_key_frame) {
    const int N = GetNumParamBlocks();
    if ( 0 == N ) return false;

    const auto &latest_optimized_key_frame = optimized_key_frames_.back();

    optimized_key_frame = KeyFrame(N-1, latest_optimized_key_frame.time, latest_optimized_key_frame.pr);

    return true;
}

bool CeresBackEnd::GetOptimizedKeyFrames(std::deque<KeyFrame> &optimized_key_frames) {
    optimized_key_frames.clear();

    const int N = GetNumParamBlocks();
    if ( 0 == N ) return false;

    for (int param_id = 0; param_id < N; param_id++) {
        const auto &optimized_key_frame = optimized_key_frames_.at(param_id);

        optimized_key_frames.emplace_back( param_id, optimized_key_frame.time, optimized_key_frame.pr);
    }

    return true;
}

Eigen::MatrixXd CeresBackEnd::GetInformationMatrix(Eigen::VectorXd noise) {

    Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(noise.rows(), noise.rows());

    for (int i = 0; i < noise.rows(); i++) {
        information_matrix(i, i) /= noise(i);
    }

    return information_matrix;
}

} // namespace graph_ptr_optimization