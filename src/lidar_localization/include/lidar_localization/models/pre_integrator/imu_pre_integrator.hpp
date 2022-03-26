#pragma once

#include "lidar_localization/models/pre_integrator/pre_integrator.hpp"

#include "lidar_localization/sensor_data/imu_data.hpp"

#include <random>

namespace lidar_localization {

using Eigen::Vector3d;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;

class IMUPreIntegrator : public PreIntegrator {
public:
    struct IMUPreIntegration {

        double T;
        
        Vector3d g;

        Vector3d    alpha_ij;
        Quaterniond theta_ij;
        Vector3d    beta_ij;

        Vector3d linearized_ba;
        Vector3d linearized_bg;

        Matrix<double, 15, 15> P;
        Matrix<double, 15, 15> J;

        double GetT(void) const { return T; }
        
        Vector3d GetGravity(void) const { return g; }

        Vector3d GetMeasurement_alpha(void) const{
            return alpha_ij;
        }

        Quaterniond GetMeasurement_theta(void) const{
            return theta_ij;
        }

        Vector3d GetMeasurement_beta(void) const{
            return beta_ij;
        }

        MatrixXd GetInformation(void) const {
            return P.inverse();
        }

        MatrixXd GetJacobian(void) const {
            return J;
        }
    };

    IMUPreIntegrator(const YAML::Node& node);

    bool Init(const IMUData &init_imu_data, IMUPreIntegration &imu_pre_integratio);
    bool Update(const IMUData &imu_data, IMUPreIntegration &imu_pre_integratio);
    bool Reset(const IMUData &init_imu_data, IMUPreIntegration &imu_pre_integration);

private:
    double sum_dt_ = 0;

    std::deque<IMUData> imu_data_buff_;

    struct {
        double GRAVITY_MAGNITUDE;
    } EARTH;

    struct {
        struct {
            double ACCEL;
            double GYRO;
        } RANDOM_WALK;
        struct {
            double ACCEL;
            double GYRO;
        } MEASUREMENT;
    } COV;

    struct {
        double ACCEL;
        double GYRO;
    } BIAS;

    Matrix<double, 15, 15> jacobian_, covariance_;
    Matrix<double, 18, 18> noise_;
    Vector3d g_vec_;

    Vector3d linearized_ba_;
    Vector3d linearized_bg_;

    void ResetState(const IMUData &init_imu_data, IMUPreIntegration &imu_pre_integration);
    void showIMUPreIntegration(const IMUPreIntegration &imu_pre_integration);
    void MidPointIntegration(double dt,
                             const Vector3d &acc0, const Vector3d &gyr0,
                             const Vector3d &acc1, const Vector3d &gyr1,
                             const Vector3d &delta_p, const Quaterniond &delta_q,
                             const Vector3d &delta_v, const Vector3d &linearized_ba,
                             const Vector3d &linearized_bg, Vector3d &result_delta_p,
                             Quaterniond &result_delta_q, Vector3d &result_delta_v,
                             Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian);
};

} // namespace lidar_localization
