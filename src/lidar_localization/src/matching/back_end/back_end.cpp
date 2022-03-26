#include "lidar_localization/matching/back_end/back_end.hpp"

#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include "glog/logging.h"

#include "lidar_localization/sensor_data/speed_bias.hpp"
#include "lidar_localization/global_defination/global_defination.h"
#include "lidar_localization/tools/file_manager.hpp"

namespace lidar_localization {

BackEnd::BackEnd() {
    InitWithConfig();
}

bool BackEnd::InitWithConfig() {
    //
    // load lio localization backend config file:
    //
    std::string config_file_path = WORK_SPACE_PATH + "/config/matching/back_end.yaml";
    YAML::Node config_node = YAML::LoadFile(config_file_path);

    std::cout << "-----------------Init LIO Localization, Backend-------------------" << std::endl;

    InitDataPath(config_node);
    InitKeyFrameSelection(config_node);
    InitOptimizer(config_node);
    InitIMUPreIntegrator(config_node);

    return true;
}

bool BackEnd::InitDataPath(const YAML::Node& config_node) {
    std::string data_path = config_node["data_path"].as<std::string>();
    if (data_path == "./") {
        data_path = WORK_SPACE_PATH;
    }

    if (!FileManager::CreateDirectory(data_path + "/slam_data"))
        return false;

    trajectory_path_ = data_path + "/slam_data/trajectory";
    if (!FileManager::InitDirectory(trajectory_path_, "Estimated Trajectory"))
        return false;

    return true;
}

bool BackEnd::InitKeyFrameSelection(const YAML::Node& config_node) {
    key_frame_config_.max_distance = config_node["key_frame"]["max_distance"].as<float>();
    key_frame_config_.max_interval = config_node["key_frame"]["max_interval"].as<float>();
    key_frame_config_.max_key_frame_interval = config_node["key_frame"]["max_key_frame_interval"].as<int>();

    return true;
}

bool BackEnd::InitOptimizer(const YAML::Node& config_node) {
    // init sliding window:
    const int sliding_window_size = config_node["sliding_window_size"].as<int>();
    ceres_back_end_ptr_ = std::make_shared<CeresBackEnd>(sliding_window_size);

    // select measurements:
    measurement_config_.source.map_matching = config_node["measurements"]["map_matching"].as<bool>();
    measurement_config_.source.imu_pre_integration = config_node["measurements"]["imu_pre_integration"].as<bool>();

    // get measurement noises, pose:
    measurement_config_.noise.lidar_odometry.resize(6);
    measurement_config_.noise.map_matching.resize(6);
    for (int i = 0; i < 6; ++i) {
        measurement_config_.noise.lidar_odometry(i) = config_node["lidar_odometry"]["noise"][i].as<double>();
        measurement_config_.noise.map_matching(i) = config_node["map_matching"]["noise"][i].as<double>();
    }

    // get measurement noises, position:
    measurement_config_.noise.gnss_position.resize(3);
    for (int i = 0; i < 3; i++) {
        measurement_config_.noise.gnss_position(i) =
            config_node["gnss_position"]["noise"][i].as<double>();
    }

    return true;
}

bool BackEnd::InitIMUPreIntegrator(const YAML::Node& config_node) {
    imu_pre_integrator_ptr_ = nullptr;
    
    if (measurement_config_.source.imu_pre_integration) {
        imu_pre_integrator_ptr_ = std::make_shared<IMUPreIntegrator>(config_node["imu_pre_integration"]);
    }

    return true;
}

bool BackEnd::UpdateIMUPreIntegration(const IMUData &imu_data) {
    if (!measurement_config_.source.imu_pre_integration || nullptr == imu_pre_integrator_ptr_ ){
        return false;
    }

    if(!imu_pre_integrator_ptr_->IsInited()){
        if (imu_pre_integrator_ptr_) 
            imu_pre_integrator_ptr_->Init(imu_data, imu_pre_integration_);
    }
    
    if (imu_pre_integrator_ptr_->Update(imu_data, imu_pre_integration_)) { 
        return true;
    }

    return false;
}

bool BackEnd::Update(const PoseData &laser_odom, const PoseData &map_matching_odom, const IMUData &imu_data, const PoseData& gnss_pose) {
    ResetParam();

    if (MaybeNewKeyFrame(laser_odom, map_matching_odom, imu_data, gnss_pose)) {
        UpdateOptimizer();
        MaybeOptimized();
    }

    return true;
}

bool BackEnd::HasNewKeyFrame() {
    return has_new_key_frame_;
}

bool BackEnd::HasNewOptimized() {
    return has_new_optimized_;
}

void BackEnd::GetLatestKeyFrame(KeyFrame& key_frame) {
    key_frame = current_key_frame_;
}

void BackEnd::GetLatestKeyGNSS(KeyFrame& key_frame) {
    key_frame = current_key_gnss_;
}

void BackEnd::GetLatestOptimizedOdometry(KeyFrame& key_frame) {
    ceres_back_end_ptr_->GetLatestOptimizedKeyFrame(key_frame);
}

void BackEnd::GetOptimizedKeyFrames(std::deque<KeyFrame>& key_frames_deque) {
    ceres_back_end_ptr_->GetOptimizedKeyFrames(key_frames_deque);
}

bool BackEnd::SavePose(std::ofstream& ofs, const Eigen::Matrix4f& pose) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ofs << pose(i, j);
            
            if (i == 2 && j == 3) {
                ofs << std::endl;
            } else {
                ofs << " ";
            }
        }
    }

    return true;
}

bool BackEnd::SaveOptimizedTrajectory() {
    static Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();

    if (ceres_back_end_ptr_->GetNumParamBlocks() == 0)
        return false;

    // create output files:
    std::ofstream laser_odom_ofs, optimized_ofs, ground_truth_ofs;
    if (
        !FileManager::CreateFile(laser_odom_ofs, trajectory_path_ + "/laser_odom.txt") ||
        !FileManager::CreateFile(optimized_ofs, trajectory_path_ + "/optimized.txt") ||
        !FileManager::CreateFile(ground_truth_ofs, trajectory_path_ + "/ground_truth.txt")  
    ) {
        return false;
    }

    // load optimized key frames:
    GetOptimizedKeyFrames(key_frames_.optimized);

    for (size_t i = 0; i < key_frames_.optimized.size(); ++i) {
        // a. lidar odometry:
        current_pose = key_frames_.lidar.at(i).pose;
        SavePose(laser_odom_ofs, current_pose);
        // b. sliding window optimized odometry:
        current_pose = key_frames_.optimized.at(i).pose;
        SavePose(optimized_ofs, current_pose);
        // c. IMU/GNSS position reference as ground truth:
        current_pose = key_frames_.reference.at(i).pose;
        SavePose(ground_truth_ofs, current_pose);
    }

    return true;
}

void BackEnd::ResetParam() {
    has_new_key_frame_ = false;
    has_new_optimized_ = false;
}

bool BackEnd::MaybeNewKeyFrame(const PoseData &laser_odom, const PoseData &map_matching_odom, const IMUData &imu_data, const PoseData &gnss_odom) {
    static KeyFrame last_key_frame;

    if (key_frames_.lidar.empty()) 
    {
        has_new_key_frame_ = true;
        if (imu_pre_integrator_ptr_) {
            imu_pre_integrator_ptr_->Init(imu_data, imu_pre_integration_);
        }
    } 
    else if ((laser_odom.pose.block<3,1>(0, 3) - last_key_frame.pose.block<3,1>(0, 3)).lpNorm<1>() > key_frame_config_.max_distance || (laser_odom.time - last_key_frame.time) > key_frame_config_.max_interval)
    {
        if (imu_pre_integrator_ptr_) {
            imu_pre_integrator_ptr_->Reset(imu_data, imu_pre_integration_); 
        }
        has_new_key_frame_ = true;
    } 
    else
    {
        has_new_key_frame_ = false;
    }

    if (has_new_key_frame_) {
        current_key_frame_.time  = laser_odom.time;
        current_key_frame_.index = key_frames_.lidar.size();
        current_key_frame_.pose  = laser_odom.pose;

        current_key_frame_.vel.v = gnss_odom.vel.v;
        current_key_frame_.vel.w = gnss_odom.vel.w;

        current_map_matching_pose_ = map_matching_odom;

        current_key_gnss_.time  = current_key_frame_.time;
        current_key_gnss_.index = current_key_frame_.index;
        current_key_gnss_.pose  = gnss_odom.pose;
        current_key_gnss_.vel.v = gnss_odom.vel.v;
        current_key_gnss_.vel.w = gnss_odom.vel.w;

        key_frames_.lidar.push_back(current_key_frame_);
        key_frames_.reference.push_back(current_key_gnss_);

        last_key_frame = current_key_frame_;

        ++counter_.key_frame;
    }

    return has_new_key_frame_;
}

bool BackEnd::UpdateOptimizer(void) {
    static KeyFrame last_key_frame_ = current_key_frame_;

    if (ceres_back_end_ptr_->GetNumParamBlocks() == 0) {
        ceres_back_end_ptr_->AddPRParam(current_key_frame_, true);

        SpeedBias speed_bias;
        speed_bias.time  = current_key_frame_.time;
        speed_bias.index = current_key_frame_.index;
        speed_bias.vel   = current_key_frame_.vel.v.cast<double>();
        speed_bias.ba    = imu_pre_integration_.linearized_ba;
        speed_bias.bg    = imu_pre_integration_.linearized_bg;

        ceres_back_end_ptr_->AddVAGParam(speed_bias, false);
    }
    else {
        ceres_back_end_ptr_->AddPRParam(current_key_frame_, false);
        
        SpeedBias speed_bias;
        speed_bias.time  = current_key_frame_.time;
        speed_bias.index = current_key_frame_.index;
        speed_bias.vel   = current_key_frame_.vel.v.cast<double>();
        speed_bias.ba    = imu_pre_integration_.linearized_ba;
        speed_bias.bg    = imu_pre_integration_.linearized_bg;

        ceres_back_end_ptr_->AddVAGParam(speed_bias, false);
    }

    const int N = ceres_back_end_ptr_->GetNumParamBlocks();
    const int param_index_j = N - 1;

    if (N > 0 && measurement_config_.source.map_matching) {
        Eigen::Matrix4d prior_pose = current_map_matching_pose_.pose.cast<double>();
        ceres_back_end_ptr_->AddMapMatchingPoseFactor(param_index_j, prior_pose, measurement_config_.noise.map_matching);
    }   
    
    if (N > 1) {
        const int param_index_i = N - 2;
        Eigen::Matrix4d relative_pose = (last_key_frame_.pose.inverse() * current_key_frame_.pose).cast<double>();
        ceres_back_end_ptr_->AddRelativePoseFactor(param_index_i, param_index_j, relative_pose, measurement_config_.noise.lidar_odometry);
        
        if (measurement_config_.source.imu_pre_integration) 
            ceres_back_end_ptr_->AddIMUPreIntegrationFactor(param_index_i, param_index_j, param_index_i, param_index_j, imu_pre_integration_);
    }

    last_key_frame_ = current_key_frame_;

    return true;
}

bool BackEnd::MaybeOptimized() {    
   bool need_optimize = false; 

    if (counter_.HasEnoughKeyFrames(key_frame_config_.max_key_frame_interval))
        need_optimize = true;
    
    if (need_optimize && ceres_back_end_ptr_->Optimize()) {
        has_new_optimized_ = true;
        return true;
    }
    
    return false;
}

} // namespace lidar_localization