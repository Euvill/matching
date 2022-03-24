#include "lidar_localization/matching/front_end/matching_flow.hpp"

#include "glog/logging.h"
#include "lidar_localization/global_defination/global_defination.h"

namespace lidar_localization {

MatchingFlow::MatchingFlow(ros::NodeHandle& nh) {
    //
    // subscribers:
    // 
    // a. undistorted Velodyne measurement: 
    cloud_sub_ptr_ = std::make_shared<CloudSubscriber>(nh, "/synced_cloud", 100000);
    // b. lidar pose in map frame:
    gnss_sub_ptr_ = std::make_shared<OdometrySubscriber>(nh, "/synced_gnss", 100000);

    //
    // publishers:
    // 
    // a. global point cloud map:
    global_map_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "/global_map", "/n_frame", 100);
    // b. local point cloud map:
    local_map_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "/local_map", "/n_frame", 100);
    // c. current scan:
    current_scan_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "/current_scan", "/n_frame", 100);
    
    // d. estimated lidar pose in map frame, lidar frontend:
    laser_odom_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "/laser_odometry", "/n_frame", "/lidar_frame", 100);
    // e. estimated lidar pose in map frame, map matching:
    map_matching_odom_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "/map_matching_odometry", "/n_frame", "/lidar_frame", 100);

    // TODO: move lidar TF to sliding window backend
    laser_tf_pub_ptr_ = std::make_shared<TFBroadCaster>("/n_frame", "/velo_link");

    matching_ptr_ = std::make_shared<Matching>();
}

bool MatchingFlow::Run() {
    // update global map if necessary:
    if (
        matching_ptr_->HasNewGlobalMap() && 
        global_map_pub_ptr_->HasSubscribers()
    ) {
        global_map_pub_ptr_->Publish( matching_ptr_->GetGlobalMap() );
    }

    // update local map if necessary:
    if (
        matching_ptr_->HasNewLocalMap() && 
        local_map_pub_ptr_->HasSubscribers()
    ) {
        local_map_pub_ptr_->Publish( matching_ptr_->GetLocalMap() );
    }

    // read inputs:
    ReadData();

    while(HasData()) {
        if (!ValidData()) {
            LOG(INFO) << "Invalid data. Skip matching" << std::endl;
            continue;
        }

        if (UpdateMatching()) {
            PublishData();
        }
    }

    return true;
}

bool MatchingFlow::ReadData() {
    // pipe lidar measurements and reference pose into buffer:
    cloud_sub_ptr_->ParseData(cloud_data_buff_);
    gnss_sub_ptr_->ParseData(gnss_data_buff_);

    return true;
}

bool MatchingFlow::HasData() {
    if (cloud_data_buff_.empty() ||  gnss_data_buff_.empty())
        return false;
    
    if (matching_ptr_->HasInited())
        return true;
        
    return true;
}

bool MatchingFlow::ValidData() {
    current_cloud_data_ = cloud_data_buff_.front();

    /*if  ( matching_ptr_->HasInited() ) {
        cloud_data_buff_.pop_front();
        //gnss_data_buff_.clear();
        return true;
    }*/

    current_gnss_data_ = gnss_data_buff_.front();

    double diff_time = current_cloud_data_.time - current_gnss_data_.time;

    //
    // here assumes that the freq. of lidar mseasurements is 10Hz:
    //
    if (diff_time < -0.05) {
        cloud_data_buff_.pop_front();
        return false;
    }

    if (diff_time > 0.05) {
        gnss_data_buff_.pop_front();
        return false;
    }

    cloud_data_buff_.pop_front();
    gnss_data_buff_.pop_front();

    return true;
}

bool MatchingFlow::UpdateMatching() {
    //if (!matching_ptr_->HasInited()) 
    
    matching_ptr_->SetGNSSPose(current_gnss_data_.pose);

    return matching_ptr_->Update(
        current_cloud_data_, 
        laser_odometry_, map_matching_odometry_
    );
}

bool MatchingFlow::PublishData() {
    const double &timestamp_synced = current_cloud_data_.time;

    laser_tf_pub_ptr_->SendTransform(laser_odometry_, timestamp_synced);

    laser_odom_pub_ptr_->Publish(laser_odometry_, timestamp_synced);

    map_matching_odom_pub_ptr_->Publish(map_matching_odometry_, timestamp_synced);

    current_scan_pub_ptr_->Publish(matching_ptr_->GetCurrentScan());

    return true;
}

} // namespace lidar_localization