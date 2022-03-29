#include "lidar_localization/matching/back_end/back_end_flow.hpp"

#include "glog/logging.h"

#include "lidar_localization/tools/file_manager.hpp"
#include "lidar_localization/global_defination/global_defination.h"

namespace lidar_localization {

BackEndFlow::BackEndFlow(ros::NodeHandle& nh) {
    //
    // subscribers:
    //
    // a. lidar odometry:
    laser_odom_sub_ptr_ = std::make_shared<OdometrySubscriber>(nh, "/laser_odometry", 100000);
    // b. map matching odometry:
    map_matching_odom_sub_ptr_ = std::make_shared<OdometrySubscriber>(nh, "/map_matching_odometry", 100000);
    // c. IMU measurement, for pre-integration:
    imu_raw_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/kitti/oxts/imu", 1000000);
    imu_synced_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/synced_imu", 100000);
    // d. GNSS position:
    gnss_pose_sub_ptr_ = std::make_shared<OdometrySubscriber>(nh, "/synced_gnss", 100000);

    // 
    //  publishers:
    // 
    // a. current lidar key frame:
    key_frame_pub_ptr_ = std::make_shared<KeyFramePublisher>(nh, "/key_frame", "/n_frame", 100);
    // b. current reference GNSS frame:
    key_gnss_pub_ptr_ = std::make_shared<KeyFramePublisher>(nh, "/key_gnss", "/n_frame", 100);
    // c. optimized odometry:
    optimized_odom_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "/optimized_odometry", "/n_frame", "/lidar_frame", 100);
    key_frames_pub_ptr_ = std::make_shared<KeyFramesPublisher>(nh, "/optimized_key_frames", "/n_frame", 100);
    //
    // backend:
    //
    back_end_ptr_ = std::make_shared<BackEnd>();
}

bool BackEndFlow::Run() {
    // load messages into buffer:
    if (!ReadData())
        return false;
    
    while(HasData()) {
        // make sure all the measurements are synced:
        if (!ValidData())
            continue;

        UpdateBackEnd();
        PublishData();
    }

    return true;
}

bool BackEndFlow::SaveOptimizedTrajectory() {
    back_end_ptr_ -> SaveOptimizedTrajectory();

    return true;
}

bool BackEndFlow::ReadData() {
    // a. lidar odometry:
    laser_odom_sub_ptr_->ParseData(laser_odom_data_buff_);
    // b. map matching odometry:
    map_matching_odom_sub_ptr_->ParseData(map_matching_odom_data_buff_);
    // c. IMU measurement, for pre-integration:
    imu_raw_sub_ptr_->ParseData(imu_raw_data_buff_);
    imu_synced_sub_ptr_->ParseData(imu_synced_data_buff_);
    // d. GNSS position:
    gnss_pose_sub_ptr_->ParseData(gnss_pose_data_buff_);

    return true;
}

bool BackEndFlow::HasData() {
    if (laser_odom_data_buff_.empty() || map_matching_odom_data_buff_.empty() || imu_synced_data_buff_.empty() || gnss_pose_data_buff_.empty()) 
        return false;

    return true;
}

bool BackEndFlow::ValidData() {
    current_laser_odom_data_ = laser_odom_data_buff_.front();
    current_map_matching_odom_data_ = map_matching_odom_data_buff_.front();
    current_imu_data_ = imu_synced_data_buff_.front();
    current_gnss_pose_data_ = gnss_pose_data_buff_.front();

    double diff_map_matching_odom_time = current_laser_odom_data_.time - current_map_matching_odom_data_.time;
    double diff_imu_time = current_laser_odom_data_.time - current_imu_data_.time;
    double diff_gnss_pose_time = current_laser_odom_data_.time - current_gnss_pose_data_.time;

    if ( diff_map_matching_odom_time < -0.05 || diff_imu_time < -0.05 || diff_gnss_pose_time < -0.05 ) {
        laser_odom_data_buff_.pop_front();
        return false;
    }

    if ( diff_map_matching_odom_time > 0.05 ) {
        map_matching_odom_data_buff_.pop_front();
        return false;
    }

    if ( diff_imu_time > 0.05 ) {
        imu_synced_data_buff_.pop_front();
        return false;
    }

    if ( diff_gnss_pose_time > 0.05 ) {
        gnss_pose_data_buff_.pop_front();
        return false;
    }

    laser_odom_data_buff_.pop_front();
    map_matching_odom_data_buff_.pop_front();
    imu_synced_data_buff_.pop_front();
    gnss_pose_data_buff_.pop_front();

    return true;
}

bool BackEndFlow::UpdateIMUPreIntegration(void) {

    while (!imu_raw_data_buff_.empty() && 
            imu_raw_data_buff_.front().time < current_imu_data_.time && 
            back_end_ptr_->UpdateIMUPreIntegration(imu_raw_data_buff_.front())) {
        imu_raw_data_buff_.pop_front();
    }

    return true;
}

bool BackEndFlow::UpdateBackEnd() {
    static bool odometry_inited = false;
    static Eigen::Matrix4f odom_init_pose = Eigen::Matrix4f::Identity();

    if (!odometry_inited) {
        // the origin of lidar odometry frame in map frame as init pose:
        odom_init_pose = current_gnss_pose_data_.pose * current_laser_odom_data_.pose.inverse();

        odometry_inited = true;
    }
    
    // update IMU pre-integration:
    UpdateIMUPreIntegration();
    
    // current lidar odometry in map frame:
    current_laser_odom_data_.pose = odom_init_pose * current_laser_odom_data_.pose;

    // optimization is carried out in map frame:
    return back_end_ptr_->Update(
        current_laser_odom_data_, 
        current_map_matching_odom_data_,
        current_imu_data_,
        current_gnss_pose_data_
    );
}

bool BackEndFlow::PublishData() {
    if (back_end_ptr_->HasNewKeyFrame()) {        
        KeyFrame key_frame;

        back_end_ptr_->GetLatestKeyFrame(key_frame);
        key_frame_pub_ptr_->Publish(key_frame);

        back_end_ptr_->GetLatestKeyGNSS(key_frame);
        key_gnss_pub_ptr_->Publish(key_frame);
    }

    if (back_end_ptr_->HasNewOptimized()) {
        KeyFrame key_frame;
        back_end_ptr_->GetLatestOptimizedOdometry(key_frame);
        optimized_odom_pub_ptr_->Publish(key_frame.pose, key_frame.time);

        //std::deque<KeyFrame> optimized_key_frames;
        //back_end_ptr_->GetOptimizedKeyFrames(optimized_key_frames);
        //key_frames_pub_ptr_->Publish(optimized_key_frames);
    }

    return true;
}

} // namespace lidar_localization