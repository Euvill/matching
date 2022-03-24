/*
 * @Description: IMU pre-integrator for LIO mapping, implementation
 * @Author: Ge Yao
 * @Date: 2020-11-29 15:47:49
 */

#include "lidar_localization/models/pre_integrator/imu_pre_integrator.hpp"

#include "lidar_localization/global_defination/global_defination.h"

#include "glog/logging.h"

namespace lidar_localization {

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
              << std::endl;

    // a. gravity constant:
    pre_int_state.g_ = Eigen::Vector3d(
        0.0, 
        0.0, 
        EARTH.GRAVITY_MAGNITUDE
    );

    // b. process noise:
    Q_.block<3, 3>(INDEX_M_ACC_PREV, INDEX_M_ACC_PREV) = Q_.block<3, 3>(INDEX_M_ACC_CURR, INDEX_M_ACC_CURR) = COV.MEASUREMENT.ACCEL * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_M_GYR_PREV, INDEX_M_GYR_PREV) = Q_.block<3, 3>(INDEX_M_GYR_CURR, INDEX_M_GYR_CURR) = COV.MEASUREMENT.GYRO * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_R_ACC_PREV, INDEX_R_ACC_PREV) = COV.RANDOM_WALK.ACCEL * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_R_GYR_PREV, INDEX_R_GYR_PREV) = COV.RANDOM_WALK.GYRO * Eigen::Matrix3d::Identity();

    // c. process equation, state propagation:
    F_.block<3, 3>(INDEX_ALPHA,  INDEX_BETA) =  Eigen::Matrix3d::Identity();
    F_.block<3, 3>(INDEX_THETA,   INDEX_B_G) = -Eigen::Matrix3d::Identity();

    // d. process equation, noise input:
    B_.block<3, 3>(INDEX_THETA, INDEX_M_GYR_PREV) = B_.block<3, 3>(INDEX_THETA, INDEX_M_GYR_CURR) = 0.50 * Eigen::Matrix3d::Identity();
    B_.block<3, 3>(INDEX_B_A, INDEX_R_ACC_PREV) = B_.block<3, 3>(INDEX_B_G, INDEX_R_GYR_PREV) = Eigen::Matrix3d::Identity();
}

/**
 * @brief  reset IMU pre-integrator
 * @param  init_imu_data, init IMU measurements
 * @return true if success false otherwise
 */
bool IMUPreIntegrator::Init(const IMUData &init_imu_data) {
    // reset pre-integrator state:
    ResetState(init_imu_data);
    
    // mark as inited:
    is_inited_ = true;

    return true;
}

/**
 * @brief  update IMU pre-integrator
 * @param  imu_data, current IMU measurements
 * @return true if success false otherwise
 */
bool IMUPreIntegrator::Update(const IMUData &imu_data) {
    if ( imu_data_buff_.front().time < imu_data.time ) {
        // set buffer:
        imu_data_buff_.push_back(imu_data);

        // update state mean, covariance and Jacobian:
        UpdateState();

        // move forward:
        imu_data_buff_.pop_front();
    }

    return true;
}

/**
 * @brief  reset IMU pre-integrator using new init IMU measurement
 * @param  init_imu_data, new init IMU measurements
 * @param  output pre-integration result for constraint building as IMUPreIntegration
 * @return true if success false otherwise
 */
bool IMUPreIntegrator::Reset(
    const IMUData &init_imu_data, 
    IMUPreIntegration &imu_pre_integration
) {
    // one last update:
    Update(init_imu_data);

    // set output IMU pre-integration:
    imu_pre_integration.T_ = init_imu_data.time - time_;

    // set gravity constant:
    imu_pre_integration.g_ = pre_int_state.g_;

    // set measurement:
    imu_pre_integration.alpha_ij_ = pre_int_state.alpha_ij_;
    imu_pre_integration.theta_ij_ = pre_int_state.theta_ij_;
    imu_pre_integration.beta_ij_  = pre_int_state.beta_ij_;
    imu_pre_integration.b_a_i_    = pre_int_state.b_a_i_;
    imu_pre_integration.b_g_i_    = pre_int_state.b_g_i_;
    // set information:
    imu_pre_integration.P_ = P_;
    // set Jacobian:
    imu_pre_integration.J_ = J_;

    // reset:
    ResetState(init_imu_data);

    return true;
}

/**
 * @brief  reset pre-integrator state using IMU measurements
 * @param  void
 * @return void
 */
void IMUPreIntegrator::ResetState(const IMUData &init_imu_data) {
    // reset time:
    time_ = init_imu_data.time;

    // a. reset relative translation:
    pre_int_state.alpha_ij_ = Eigen::Vector3d::Zero();
    // b. reset relative orientation:
    pre_int_state.theta_ij_ = Eigen::Quaterniond::Identity();

    static int count = 0;
    std::cout << " IMUPreIntegrator::ResetState " << ++count << std::endl;

    // c. reset relative velocity:
    pre_int_state.beta_ij_ = Eigen::Vector3d::Zero();
    // d. set init bias, acceleometer:
    pre_int_state.b_a_i_ = Eigen::Vector3d(
        init_imu_data.accel_bias.x,
        init_imu_data.accel_bias.y,
        init_imu_data.accel_bias.z
    );
    // d. set init bias, gyroscope:
    pre_int_state.b_g_i_ = Eigen::Vector3d(
        init_imu_data.gyro_bias.x,
        init_imu_data.gyro_bias.y,
        init_imu_data.gyro_bias.z
    );

    // reset state covariance:
    P_ = MatrixP::Zero();

    // reset Jacobian:
    J_ = MatrixJ::Identity();

    // reset buffer:
    imu_data_buff_.clear();
    imu_data_buff_.push_back(init_imu_data);
}

/**
 * @brief  update pre-integrator state: mean, covariance and Jacobian
 * @param  void
 * @return void
 */
void IMUPreIntegrator::UpdateState(void) {
    static double T = 0.0;

    static Eigen::Vector3d w_mid = Eigen::Vector3d::Zero();
    static Eigen::Vector3d a_mid = Eigen::Vector3d::Zero();

    static Eigen::Quaterniond prev_theta_ij = Eigen::Quaterniond::Identity();
    static Eigen::Quaterniond curr_theta_ij = Eigen::Quaterniond::Identity();
    static Eigen::Quaterniond d_theta_ij = Eigen::Quaterniond::Identity();
    
    static int count = 0;
    std::cout << " IMUPreIntegrator::UpdateState" << ++count << std::endl;

    //
    // parse measurements:
    //
    // get measurement handlers:
    const IMUData &prev_imu_data = imu_data_buff_.at(0);
    const IMUData &curr_imu_data = imu_data_buff_.at(1);

    // get time delta:
    T = curr_imu_data.time - prev_imu_data.time;

    // get measurements:
    Eigen::Vector3d prev_w(
        prev_imu_data.angular_velocity.x - pre_int_state.b_g_i_.x(),
        prev_imu_data.angular_velocity.y - pre_int_state.b_g_i_.y(),
        prev_imu_data.angular_velocity.z - pre_int_state.b_g_i_.z()
    );
    Eigen::Vector3d curr_w(
        curr_imu_data.angular_velocity.x - pre_int_state.b_g_i_.x(),
        curr_imu_data.angular_velocity.y - pre_int_state.b_g_i_.y(),
        curr_imu_data.angular_velocity.z - pre_int_state.b_g_i_.z()
    );

    Eigen::Vector3d prev_a(
        prev_imu_data.linear_acceleration.x - pre_int_state.b_a_i_.x(),
        prev_imu_data.linear_acceleration.y - pre_int_state.b_a_i_.y(),
        prev_imu_data.linear_acceleration.z - pre_int_state.b_a_i_.z()
    );
    Eigen::Vector3d curr_a(
        curr_imu_data.linear_acceleration.x - pre_int_state.b_a_i_.x(),
        curr_imu_data.linear_acceleration.y - pre_int_state.b_a_i_.y(),
        curr_imu_data.linear_acceleration.z - pre_int_state.b_a_i_.z()
    );

    /*std::cout << std::endl;
    std::cout << "bias accel: " <<
                 pre_int_state.b_a_i_.x() << " " << 
                 pre_int_state.b_a_i_.y() << " " << 
                 pre_int_state.b_a_i_.z() << " " << std::endl;
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "bias gyro: " <<
                 pre_int_state.b_g_i_.x() << " " << 
                 pre_int_state.b_g_i_.y() << " " << 
                 pre_int_state.b_g_i_.z() << " " << std::endl;
    std::cout << std::endl;*/

    //
    // a. update mean:
    //
    // 1. get w_mid:
    w_mid = 0.5 * (prev_w + curr_w);

    // 2. update relative orientation:
    prev_theta_ij = pre_int_state.theta_ij_;
    d_theta_ij = Eigen::Quaterniond(1.0, w_mid(0) * T / 2, w_mid(1) * T / 2, w_mid(2) * T / 2);
    pre_int_state.theta_ij_ = d_theta_ij * pre_int_state.theta_ij_;
    curr_theta_ij = pre_int_state.theta_ij_;
    
    // 3. get a_mid:
    a_mid = 0.5 * ( prev_theta_ij * prev_a + curr_theta_ij * curr_a );
    
    // 4. update relative translation:
    pre_int_state.alpha_ij_ += (pre_int_state.beta_ij_ + 0.5 * a_mid * T) * T;

    // 5. update relative velocity:
    pre_int_state.beta_ij_ += a_mid * T;

    //
    // b. update covariance:
    //
    // 1. intermediate results:
    // dR_inv = d_theta_ij.inverse().matrix();
    Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

    R_w_x <<     0.0, -w_mid(2),  w_mid(1),
            w_mid(2),       0.0, -w_mid(0),
           -w_mid(1),  w_mid(0),       0.0;

    R_a_0_x <<       0.0, -prev_a(2),  prev_a(1),
               prev_a(2),        0.0, -prev_a(0),
              -prev_a(1),  prev_a(0),        0.0;

    R_a_1_x <<       0.0, -curr_a(2),  curr_a(1),
               curr_a(2),        0.0, -curr_a(0),
              -curr_a(1),  curr_a(0),        0.0;

    double T2 = T * T;
    double T3 = T2 * T;
    //
    // 2. set up F:
    //
    F_.block<3, 3>(0, 0)  = Eigen::Matrix3d::Identity();
    F_.block<3, 3>(0, 3)  = -0.25 * prev_theta_ij.toRotationMatrix() * R_a_0_x * T2 + 
                            -0.25 * curr_theta_ij.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * T) * T2;
    F_.block<3, 3>(0, 6)  = Eigen::MatrixXd::Identity(3,3) * T;
    F_.block<3, 3>(0, 9)  = -0.25 * (prev_theta_ij.toRotationMatrix() + curr_theta_ij.toRotationMatrix()) * T2;
    F_.block<3, 3>(0, 12) = 0.25 * curr_theta_ij.toRotationMatrix() * R_a_1_x * T3;
    F_.block<3, 3>(3, 3)  = Eigen::Matrix3d::Identity() - R_w_x * T;
    F_.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3,3) * T;
    F_.block<3, 3>(6, 3)  = -0.5 * prev_theta_ij.toRotationMatrix() * R_a_0_x * T + 
                            -0.5 * curr_theta_ij.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * T) * T;
    F_.block<3, 3>(6, 6)   = Eigen::Matrix3d::Identity();
    F_.block<3, 3>(6, 9)   = -0.5 * (prev_theta_ij.toRotationMatrix() + curr_theta_ij.toRotationMatrix()) * T;
    F_.block<3, 3>(6, 12)  = 0.5 * curr_theta_ij.toRotationMatrix() * R_a_1_x * T2;
    F_.block<3, 3>(9, 9)   = Eigen::Matrix3d::Identity();
    F_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

    //
    // 3. set up G:
    //
    // G11 & G31:
    B_.block<3, 3>(0, 0) =  0.25 * prev_theta_ij.toRotationMatrix() * T2;
    B_.block<3, 3>(0, 3) =  0.25 * -curr_theta_ij.toRotationMatrix() * R_a_1_x * 0.5 * T3;
    B_.block<3, 3>(0, 6) =  0.25 * curr_theta_ij.toRotationMatrix() * T2;
    B_.block<3, 3>(0, 9) =  B_.block<3, 3>(0, 3);
    B_.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * T;
    B_.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * T;
    B_.block<3, 3>(6, 0) =  0.5 * prev_theta_ij.toRotationMatrix() * T;
    B_.block<3, 3>(6, 3) =  0.5 * -curr_theta_ij.toRotationMatrix() * R_a_1_x * 0.5 * T2;
    B_.block<3, 3>(6, 6) =  0.5 * curr_theta_ij.toRotationMatrix() * T;
    B_.block<3, 3>(6, 9) =  B_.block<3, 3>(6, 3);
    B_.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3,3) * T;
    B_.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * T;

    // 4. update P_:
    //MatrixF F = MatrixF::Identity() + T * F_;
    //MatrixB B = T * B_;

    P_ = F_ * P_ * F_.transpose() + B_ * Q_ * B_.transpose();

    // 
    // c. update Jacobian:
    //
    J_ = F_ * J_;
}

} // namespace lidar_localization