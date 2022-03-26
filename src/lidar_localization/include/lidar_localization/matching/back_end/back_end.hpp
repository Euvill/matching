#ifndef LIDAR_LOCALIZATION_MATCHING_BACK_END_SLIDING_WINDOW_HPP_
#define LIDAR_LOCALIZATION_MATCHING_BACK_END_SLIDING_WINDOW_HPP_

#include <string>
#include <deque>
#include <yaml-cpp/yaml.h>
#include <fstream>

#include "lidar_localization/sensor_data/pose_data.hpp"
#include "lidar_localization/sensor_data/imu_data.hpp"

#include "lidar_localization/sensor_data/key_frame.hpp"

#include "lidar_localization/models/ceres_back_end/ceres_back_end.hpp"

namespace lidar_localization {

class BackEnd {
  public:
    BackEnd();
    bool UpdateIMUPreIntegration(const IMUData &imu_data);

    bool Update(
      const PoseData &laser_odom,
      const PoseData &map_matching_odom,
      const IMUData &imu_data, 
      const PoseData& gnss_pose
    );

    bool HasNewKeyFrame();
    bool HasNewOptimized();
    
    void GetLatestKeyFrame(KeyFrame& key_frame);
    void GetLatestKeyGNSS(KeyFrame& key_frame);
    void GetLatestOptimizedOdometry(KeyFrame& key_frame);
    void GetOptimizedKeyFrames(std::deque<KeyFrame>& key_frames_deque);
    bool SaveOptimizedTrajectory();

  private:
    bool InitWithConfig();

    bool InitDataPath(const YAML::Node& config_node);
    bool InitKeyFrameSelection(const YAML::Node& config_node);
    bool InitOptimizer(const YAML::Node& config_node);
    bool InitIMUPreIntegrator(const YAML::Node& config_node);

    void ResetParam();
    bool SavePose(std::ofstream& ofs, const Eigen::Matrix4f& pose);
    
    bool MaybeNewKeyFrame(
      const PoseData& laser_odom, 
      const PoseData &map_matching_odom,
      const IMUData &imu_data,
      const PoseData& gnss_pose
    );
    
    bool UpdateOptimizer(void);
    bool MaybeOptimized();

  private:
    std::string trajectory_path_ = "";

    bool has_new_key_frame_ = false;
    bool has_new_optimized_ = false;

    KeyFrame current_key_frame_;
    PoseData current_map_matching_pose_;
    KeyFrame current_key_gnss_;

    // key frame buffer:
    struct {
      std::deque<KeyFrame> lidar;
      std::deque<KeyFrame> optimized;
      std::deque<KeyFrame> reference;
    } key_frames_;

    // pre-integrator:
    std::shared_ptr<IMUPreIntegrator> imu_pre_integrator_ptr_;
    IMUPreIntegrator::IMUPreIntegration imu_pre_integration_;

    // key frame config:
    struct {
      float max_distance;
      float max_interval;
      int   max_key_frame_interval;
    } key_frame_config_;

    // optimizer:
    std::shared_ptr<CeresBackEnd> ceres_back_end_ptr_;

    // measurement config:
    struct MeasurementConfig {
        struct {
          bool map_matching = false;
          bool imu_pre_integration = false;
        } source;

        struct {
          Eigen::VectorXd lidar_odometry;
          Eigen::VectorXd map_matching;
          Eigen::VectorXd gnss_position;
        } noise;
    };
    MeasurementConfig measurement_config_;

    struct {
      int key_frame = 0;

      bool HasEnoughKeyFrames(const int num_key_frames_thresh) {
        if ( key_frame >= num_key_frames_thresh ) {
          key_frame = 0;
          return true;
        }

        return false;
      }
    } counter_;
};

} // namespace lidar_localization

#endif // LIDAR_LOCALIZATION_MATCHING_BACK_END_SLIDING_WINDOW_HPP_