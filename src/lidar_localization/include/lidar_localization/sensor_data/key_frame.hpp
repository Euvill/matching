#ifndef LIDAR_LOCALIZATION_SENSOR_DATA_KEY_FRAME_HPP_
#define LIDAR_LOCALIZATION_SENSOR_DATA_KEY_FRAME_HPP_

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/so3.hpp>


namespace lidar_localization {

struct KeyFrame {
public:
    double time = 0.0;

    // key frame ID:
    unsigned int index = 0;
    
    // a. position & orientation:
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    // b. velocity:
    struct {
      Eigen::Vector3f v = Eigen::Vector3f::Zero();
      Eigen::Vector3f w = Eigen::Vector3f::Zero();
    } vel;
    // c. bias:
    struct {
      // c.1. accelerometer:
      Eigen::Vector3f accel = Eigen::Vector3f::Zero();
      // c.2. gyroscope:
      Eigen::Vector3f gyro = Eigen::Vector3f::Zero();
    } bias;

    KeyFrame() {}

    explicit KeyFrame(const int param_index, const double &T, const double *pr) {
      // set time:
      time = T;
      // set seq. ID:
      index = param_index;
      // set state:
      Eigen::Map<const Eigen::Vector3d>    pos(pr + 0);
      Eigen::Map<const Eigen::Quaterniond> ori(pr + 3);

      pose.block<3, 1>(0, 3) = pos.cast<float>();
      pose.block<3, 3>(0, 0) = ori.toRotationMatrix().cast<float>();
    }

    Eigen::Quaternionf GetQuaternion() const;
    Eigen::Vector3f GetTranslation() const;
};

}
#endif