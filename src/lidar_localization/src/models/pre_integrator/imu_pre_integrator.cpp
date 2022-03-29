#include "lidar_localization/models/pre_integrator/imu_pre_integrator.hpp"

#include "lidar_localization/global_defination/global_defination.h"

#include "lidar_localization/tools/math_utils.h"

#include "glog/logging.h"

namespace lidar_localization {

using namespace mathutils;
using Eigen::Vector3d;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;

IMUPreIntegrator::IMUPreIntegrator(const YAML::Node& node) {
    //
    // parse config:
    // 
    // a. earth constants:
    EARTH.GRAVITY_MAGNITUDE = node["earth"]["gravity_magnitude"].as<double>();
    // b. process noise:
    COV.MEASUREMENT.ACCEL = node["covariance"]["measurement"]["accel"].as<double>();
    COV.MEASUREMENT.GYRO = node["covariance"]["measurement"]["gyro"].as<double>();
    COV.RANDOM_WALK.ACCEL = node["covariance"]["random_walk"]["accel"].as<double>();
    COV.RANDOM_WALK.GYRO = node["covariance"]["random_walk"]["gyro"].as<double>();  
    // c. process bias:
    BIAS.ACCEL =  node["bias"]["accel"].as<double>();
    BIAS.GYRO  =  node["bias"]["gyro"].as<double>();

    // prompt:
    LOG(INFO) << std::endl 
              << "IMU Pre-Integration params:" << std::endl
              << "\tgravity magnitude: " << EARTH.GRAVITY_MAGNITUDE << std::endl
              << std::endl
              << "\tprocess noise:" << std::endl
              << "\t\tmeasurement:" << std::endl
              << "\t\t\taccel.: " << COV.MEASUREMENT.ACCEL << std::endl
              << "\t\t\tgyro.: " << COV.MEASUREMENT.GYRO << std::endl
              << "\t\trandom_walk:" << std::endl
              << "\t\t\taccel.: " << COV.RANDOM_WALK.ACCEL << std::endl
              << "\t\t\tgyro.: " << COV.RANDOM_WALK.GYRO << std::endl
              << "\t\tzero_bias:" << std::endl
              << "\t\t\taccel.: " << BIAS.ACCEL << std::endl
              << "\t\t\tgyro.: " << BIAS.GYRO << std::endl
              << std::endl;

    // a. gravity constant:
    g_vec_ = Vector3d(0.0,  0.0,  EARTH.GRAVITY_MAGNITUDE);

    // b. process noise:
    noise_.block<3, 3>(0, 0)   = (COV.MEASUREMENT.ACCEL * COV.MEASUREMENT.ACCEL) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(3, 3)   = (COV.MEASUREMENT.GYRO * COV.MEASUREMENT.GYRO) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(6, 6)   = (COV.MEASUREMENT.ACCEL * COV.MEASUREMENT.ACCEL) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(9, 9)   = (COV.MEASUREMENT.GYRO * COV.MEASUREMENT.GYRO) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(12, 12) = (COV.RANDOM_WALK.ACCEL * COV.RANDOM_WALK.ACCEL) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(15, 15) = (COV.RANDOM_WALK.GYRO * COV.RANDOM_WALK.GYRO) * Eigen::Matrix3d::Identity();

    // c. process bias:
    linearized_ba_ = Vector3d(BIAS.ACCEL, BIAS.ACCEL, BIAS.ACCEL);
    linearized_bg_ = Vector3d( BIAS.GYRO,  BIAS.GYRO,  BIAS.GYRO);
}


bool IMUPreIntegrator::Init(const IMUData &init_imu_data, IMUPreIntegration &imu_pre_integration) {
    ResetState(init_imu_data, imu_pre_integration);
    
    is_inited_ = true;

    return true;
}

bool IMUPreIntegrator::Update(const IMUData &imu_data, IMUPreIntegration &imu_pre_integration) {

    if (imu_data_buff_.size() >= 1 && imu_data_buff_.front().time < imu_data.time) {
        imu_data_buff_.push_back(imu_data);
        
        Vector3d    result_delta_p;
        Quaterniond result_delta_q;
        Vector3d    result_delta_v;
        Vector3d    result_linearized_ba;
        Vector3d    result_linearized_bg;

        const IMUData &prev_imu_data = imu_data_buff_.at(0);
        const IMUData &curr_imu_data = imu_data_buff_.at(1);

        double dt = curr_imu_data.time - prev_imu_data.time;

        Vector3d acc0_ = Vector3d(prev_imu_data.linear_acceleration.x, prev_imu_data.linear_acceleration.y, prev_imu_data.linear_acceleration.z);
        Vector3d gyr0_ = Vector3d(prev_imu_data.angular_velocity.x,    prev_imu_data.angular_velocity.y,    prev_imu_data.angular_velocity.z);
        Vector3d acc1_ = Vector3d(curr_imu_data.linear_acceleration.x, curr_imu_data.linear_acceleration.y, curr_imu_data.linear_acceleration.z);
        Vector3d gyr1_ = Vector3d(curr_imu_data.angular_velocity.x,    curr_imu_data.angular_velocity.y,    curr_imu_data.angular_velocity.z);

        Vector3d    delta_p_ = imu_pre_integration.alpha_ij;
        Quaterniond delta_q_ = imu_pre_integration.theta_ij;
        Vector3d    delta_v_ = imu_pre_integration.beta_ij;

        MidPointIntegration(dt, acc0_, gyr0_, acc1_, gyr1_, delta_p_, delta_q_, delta_v_,
                            linearized_ba_, linearized_bg_,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, true);

        sum_dt_ = sum_dt_ + dt;

        imu_pre_integration.P = jacobian_;
        imu_pre_integration.J = covariance_; 
        imu_pre_integration.alpha_ij = result_delta_p;
        imu_pre_integration.theta_ij = result_delta_q;
        imu_pre_integration.beta_ij  = result_delta_v;

        linearized_ba_ = result_linearized_ba;
        linearized_bg_ = result_linearized_bg;

        imu_pre_integration.g = g_vec_;
        imu_pre_integration.T = sum_dt_;
        
        imu_data_buff_.pop_front();
    }

    return true;
}


bool IMUPreIntegrator::Reset(const IMUData &init_imu_data, IMUPreIntegration &imu_pre_integration) {

    Update(init_imu_data, imu_pre_integration);

    //std::cout << std::endl;
    //std::cout << "IMUPreIntegrator Reset: " << std::endl;
    //showIMUPreIntegration(imu_pre_integration);
    //std::cout << std::endl;

    ResetState(init_imu_data, imu_pre_integration);

    return true;
}


void IMUPreIntegrator::ResetState(const IMUData &init_imu_data, IMUPreIntegration &imu_pre_integration) {

    imu_pre_integration.alpha_ij = Vector3d::Zero();
    imu_pre_integration.theta_ij = Quaterniond::Identity();
    imu_pre_integration.beta_ij  = Vector3d::Zero();

    imu_pre_integration.linearized_ba = linearized_ba_;
    imu_pre_integration.linearized_bg = linearized_bg_;

    jacobian_.setIdentity();
    covariance_.setZero();

    sum_dt_ = 0;

    imu_data_buff_.clear();
    imu_data_buff_.push_back(init_imu_data);
}

void IMUPreIntegrator::MidPointIntegration(double dt,
                                           const Vector3d &acc0, const Vector3d &gyr0,
                                           const Vector3d &acc1, const Vector3d &gyr1,
                                           const Vector3d &delta_p, const Quaterniond &delta_q,
                                           const Vector3d &delta_v, const Vector3d &linearized_ba,
                                           const Vector3d &linearized_bg, Vector3d &result_delta_p,
                                           Quaterniond &result_delta_q, Vector3d &result_delta_v,
                                           Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian) {

    // NOTE: the un_acc here is different from the un_acc in the Estimator
    Vector3d un_acc_0 = delta_q * (acc0 - linearized_ba);
    Vector3d un_gyr = 0.5 * (gyr0 + gyr1) - linearized_bg;

    result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);

    Vector3d un_acc_1 = result_delta_q * (acc1 - linearized_ba);
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;
    result_delta_v = delta_v + un_acc * dt;

    double bia_ = rng_.gaussian(COV.RANDOM_WALK.ACCEL);
    double big_ = rng_.gaussian(COV.RANDOM_WALK.GYRO);

    Vector3d bia(bia_, bia_, bia_);
    Vector3d big(big_, big_, big_);

    result_linearized_ba = linearized_ba + bia;
    result_linearized_bg = linearized_bg + big;

    if (update_jacobian) {
        Vector3d w_x = 0.5 * (gyr0 + gyr1) - linearized_bg;
        Vector3d a_0_x = acc0 - linearized_ba;
        Vector3d a_1_x = acc1 - linearized_ba;
        Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x << 0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
        R_a_0_x << 0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
        R_a_1_x << 0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

        // NOTE: F = Fd = \Phi = I + dF * dt
        MatrixXd F = MatrixXd::Zero(15, 15);
        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * dt * dt +
                              -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * dt) * dt * dt;
        F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * dt;
        F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt * dt;
        F.block<3, 3>(0, 12) = -0.1667 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * -dt;
        
        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * dt;
        F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * dt;
        
        F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * dt +
                              -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * dt) * dt;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt;
        F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * -dt;

        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        // NOTE: V = Fd * G_c
        // FIXME: verify if it is right, the 0.25 part
        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.5 * delta_q.toRotationMatrix() * dt * dt;
        V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * 0.5 * dt;
        V.block<3, 3>(0, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt * dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        
        V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * dt;
        V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * dt;
        
        V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * dt;
        V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt * 0.5 * dt;
        V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        
        V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * dt;
        V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * dt;

        jacobian_ = F * jacobian_;
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
    }

  }

void IMUPreIntegrator::showIMUPreIntegration(const IMUPreIntegration &imu_pre_integration){

    std::cout << std::endl;

    std::cout << "IMUPreIntegration:" << std::endl
              << "Time: " << imu_pre_integration.T << std::endl
              << "gravity: " << imu_pre_integration.g(0) << "," << imu_pre_integration.g(1) << "," <<imu_pre_integration.g(2) << std::endl
              << "alpha:   " << imu_pre_integration.alpha_ij(0) << "," << imu_pre_integration.alpha_ij(1) << "," <<imu_pre_integration.alpha_ij(2) << std::endl
              << "theta:   " << imu_pre_integration.theta_ij.w()<< "," << imu_pre_integration.theta_ij.x() << "," 
                             <<imu_pre_integration.theta_ij.y() << "," << imu_pre_integration.theta_ij.z() << std::endl
              << "beta:    " << imu_pre_integration.beta_ij(0)  << "," << imu_pre_integration.beta_ij(1)  << "," <<imu_pre_integration.beta_ij(2)  << std::endl
              << "linearized_ba: " << imu_pre_integration.linearized_ba(0) << "," << imu_pre_integration.linearized_ba(1) << ","  <<imu_pre_integration.linearized_ba(2) << std::endl
              << "linearized_bg: " << imu_pre_integration.linearized_bg(0) << "," << imu_pre_integration.linearized_bg(1) << ","  <<imu_pre_integration.linearized_bg(2) << std::endl;
}

} // namespace lidar_localization