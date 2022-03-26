#pragma once

#include <Eigen/Dense>

namespace lidar_localization {
class SpeedBias {
public:
    
    double time = 0.0;
    
    unsigned int index = 0;

    Eigen::Vector3d vel;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;

};
}