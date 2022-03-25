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

    // add to data buffer:
    optimized_key_frames_.push_back(optimized_key_frame);
}

void CeresBackEnd::AddRelativePoseFactor(const int param_index_i, const int param_index_j, const Eigen::Matrix4d &relative_pose, const Eigen::VectorXd &noise) {
    // create new residual block:
    ResidualRelativePose residual_relative_pose;

    // a. set param block IDs:
    residual_relative_pose.param_index_i = param_index_i;
    residual_relative_pose.param_index_j = param_index_j;

    // b.1. position:
    residual_relative_pose.m_pos = relative_pose.block<3, 1>(0, 3);
    // b.2. orientation, so3:
    residual_relative_pose.m_ori = Eigen::Quaterniond(relative_pose.block<3, 3>(0, 0).cast<double>());

    // c. set information matrix:
    residual_relative_pose.I = GetInformationMatrix(noise);

    // add to data buffer:
    residual_blocks_.relative_pose.push_back(residual_relative_pose);
}

void CeresBackEnd::AddMapMatchingPoseFactor(const int param_index, const Eigen::Matrix4d &prior_pose, const Eigen::VectorXd &noise) {
    // create new residual block:
    ResidualMapMatchingPose residual_map_matching_pose;

    // a. set param block ID:
    residual_map_matching_pose.param_index = param_index;

    // b.1. position:
    residual_map_matching_pose.m_pos = prior_pose.block<3, 1>(0, 3);
    // b.2. orientation, so3:
    residual_map_matching_pose.m_ori = Eigen::Quaterniond(prior_pose.block<3, 3>(0, 0).cast<double>());

    // c. set information matrix:
    residual_map_matching_pose.I = GetInformationMatrix(noise);

    // add to data buffer:
    residual_blocks_.map_matching_pose.push_back(residual_map_matching_pose);
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

bool CeresBackEnd::Optimize() {
    static int optimization_count = 0;
    
    // get key frames count:
    const int N = GetNumParamBlocks();

    if (N <= 1)
        return false;

    ceres::Problem problem;

    // a. add parameter blocks:
    for (int i = 0; i < N; ++i) {
        auto &target_key_frame = optimized_key_frames_.at(i);

        ceres::LocalParameterization *local_parameterization = new ParamPR();

        problem.AddParameterBlock(target_key_frame.pr, 7, local_parameterization);

        if (target_key_frame.fixed) {
            problem.SetParameterBlockConstant(target_key_frame.pr);
        }
    }

    // b.1. map matching pose constraint:
    if (!residual_blocks_.map_matching_pose.empty()) {
        for (const auto &residual_map_matching_pose: residual_blocks_.map_matching_pose) {
            auto &key_frame = optimized_key_frames_.at(residual_map_matching_pose.param_index);

            FactorMapMatchingPose *factor_map_matching_pose = GetResMapMatchingPose(residual_map_matching_pose);

            // add map matching factor into sliding window
            problem.AddResidualBlock(factor_map_matching_pose, NULL, key_frame.pr);
        }            
    }

    // b.2. relative pose constraint:
    if (!residual_blocks_.relative_pose.empty()) {
        for (const auto &residual_relative_pose: residual_blocks_.relative_pose) {
            auto &key_frame_i = optimized_key_frames_.at(residual_relative_pose.param_index_i);
            auto &key_frame_j = optimized_key_frames_.at(residual_relative_pose.param_index_j);

            FactorRelativePose *factor_relative_pose = GetResRelativePose(residual_relative_pose);

            // add relative pose factor into sliding window
            problem.AddResidualBlock(factor_relative_pose, NULL, key_frame_i.pr, key_frame_j.pr);
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