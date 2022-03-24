#include "lidar_localization/models/pre_integrator/odo_pre_integrator.hpp"

#include "lidar_localization/global_defination/global_defination.h"

#include "glog/logging.h"

namespace lidar_localization {

ODOPreIntegrator::ODOPreIntegrator(const YAML::Node& node) {
    //
    // parse config:
    // 
    // a. process noise:
    COV.MEASUREMENT.L_VELO = node["covariance"]["measurement"]["linear"].as<double>();
    COV.MEASUREMENT.W_VELO = node["covariance"]["measurement"]["angular"].as<double>(); 

    // prompt:
    LOG(INFO) << std::endl 
              << "ODO Pre-Integration params:" << std::endl
              << std::endl
              << "\tprocess noise:" << std::endl
              << "\t\tmeasurement:" << std::endl
              << "\t\t\tlinear.: " << COV.MEASUREMENT.L_VELO << std::endl
              << "\t\t\tangular.: " << COV.MEASUREMENT.W_VELO << std::endl
              << std::endl;

    // a. process noise:
    Q_.block<3, 3>(0, 0) = COV.MEASUREMENT.L_VELO * COV.MEASUREMENT.L_VELO * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(3, 3) = COV.MEASUREMENT.W_VELO * COV.MEASUREMENT.W_VELO * Eigen::Matrix3d::Identity();

    // b. process equation, state propagation:
    F_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

    // c. process equation, noise input:
    B_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    B_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
}

/**
 * @brief  reset ODO pre-integrator
 * @param  init_odo_data, init ODO measurements
 * @return true if success false otherwise
 */
bool ODOPreIntegrator::Init(const VelocityData &init_odo_data) {
    // reset pre-integrator state:
    ResetState(init_odo_data);
    
    // mark as inited:
    is_inited_ = true;

    return true;
}

/**
 * @brief  update ODO pre-integrator
 * @param  odo_data, current ODO measurements
 * @return true if success false otherwise
 */
bool ODOPreIntegrator::Update(const VelocityData &odo_data) {
    if ( odo_data_buff_.front().time < odo_data.time ) {
        // set buffer:
        odo_data_buff_.push_back(odo_data);

        // update state mean, covariance and Jacobian:
        UpdateState();

        // move forward:
        odo_data_buff_.pop_front();
    }

    return true;
}

/**
 * @brief  reset ODO pre-integrator using new init ODO measurement
 * @param  init_odo_data, new init ODO measurements
 * @param  output pre-integration result for constraint building as ODOPreIntegration
 * @return true if success false otherwise
 */
bool ODOPreIntegrator::Reset(
    const VelocityData &init_odo_data, 
    ODOPreIntegration &odo_pre_integration
) {
    // one last update:
    Update(init_odo_data);

    // set output ODO pre-integration:
    odo_pre_integration.T_ = init_odo_data.time - time_;

    // set measurement:
    odo_pre_integration.alpha_ij_ = pre_int_state.alpha_ij_;
    odo_pre_integration.theta_ij_ = pre_int_state.theta_ij_;

    // set information:
    odo_pre_integration.P_ = P_;

    // reset:
    ResetState(init_odo_data);

    return true;
}

/**
 * @brief  reset pre-integrator state using ODO measurements
 * @param  void
 * @return void
 */
void ODOPreIntegrator::ResetState(const VelocityData &init_odo_data) {
    // reset time:
    time_ = init_odo_data.time;

    // a. reset relative translation:
    pre_int_state.alpha_ij_ = Eigen::Vector3d::Zero();
    // b. reset relative orientation:
    pre_int_state.theta_ij_ = Sophus::SO3d();

    // reset state covariance:
    P_ = MatrixP::Zero();

    // reset buffer:
    odo_data_buff_.clear();
    odo_data_buff_.push_back(init_odo_data);
}

/**
 * @brief  update pre-integrator state: mean, covariance and Jacobian
 * @param  void
 * @return void
 */
void ODOPreIntegrator::UpdateState(void) {
    static double T = 0.0;

    static Eigen::Vector3d w_mid = Eigen::Vector3d::Zero();
    static Eigen::Vector3d v_mid = Eigen::Vector3d::Zero();

    static Sophus::SO3d prev_theta_ij = Sophus::SO3d();
    static Sophus::SO3d curr_theta_ij = Sophus::SO3d();
    static Sophus::SO3d d_theta_ij = Sophus::SO3d();

    static Eigen::Matrix3d dR_inv = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d prev_R = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d curr_R = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d prev_R_a_hat = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d curr_R_a_hat = Eigen::Matrix3d::Zero();

    //
    // parse measurements:
    //
    // get measurement handlers:
    const VelocityData &prev_odo_data = odo_data_buff_.at(0);
    const VelocityData &curr_odo_data = odo_data_buff_.at(1);

    // get time delta:
    T = curr_odo_data.time - prev_odo_data.time;

    // get measurements:
    Eigen::Vector3d prev_w(
        prev_odo_data.angular_velocity.x,
        prev_odo_data.angular_velocity.y,
        prev_odo_data.angular_velocity.z
    );
    Eigen::Vector3d curr_w(
        curr_odo_data.angular_velocity.x,
        curr_odo_data.angular_velocity.y,
        curr_odo_data.angular_velocity.z
    );

    Eigen::Vector3d prev_v(
        prev_odo_data.linear_velocity.x,
        prev_odo_data.linear_velocity.y,
        prev_odo_data.linear_velocity.z
    );
    Eigen::Vector3d curr_v(
        curr_odo_data.linear_velocity.x,
        curr_odo_data.linear_velocity.y,
        curr_odo_data.linear_velocity.z
    );

    //
    // a. update mean:
    //
    // 1. get w_mid:
    w_mid = 0.5 * ( prev_w + curr_w );
    // 2. update relative orientation, so3:
    prev_theta_ij = pre_int_state.theta_ij_;
    d_theta_ij = Sophus::SO3d::exp(w_mid * T);
    pre_int_state.theta_ij_ = pre_int_state.theta_ij_ * d_theta_ij;
    curr_theta_ij = pre_int_state.theta_ij_;
    
    // 3. get v_mid:
    v_mid = 0.5 * ( prev_theta_ij * prev_v + curr_theta_ij * curr_v );
    
    // 4. update relative translation:
    pre_int_state.alpha_ij_ += v_mid * T;

    //
    // b. update covariance:
    //
    // 1. intermediate results:
    dR_inv = Eigen::Matrix3d::Identity() - Sophus::SO3d::hat(w_mid) * T;
    prev_R = prev_theta_ij.matrix();
    curr_R = curr_theta_ij.matrix();
    prev_R_a_hat = prev_R * Sophus::SO3d::hat(prev_v);
    curr_R_a_hat = curr_R * Sophus::SO3d::hat(curr_v);

    //
    // 2. set up F:
    //
    // F12 & F22:
    F_.block<3, 3>(0, 3) = -0.50 * (prev_R_a_hat + curr_R_a_hat * dR_inv);
    F_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - Sophus::SO3d::hat(w_mid) * T;

    //
    // 3. set up G:
    //
    B_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * T;
    B_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * T;

    // 4. update P_:
    MatrixF F = F_;
    MatrixB B = B_;

    P_ = F * P_ * F.transpose() + B * Q_ * B.transpose();
}

} // namespace lidar_localization