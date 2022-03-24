#ifndef LIDAR_LOCALIZATION_MODELS_PRE_INTEGRATOR_ODO_PRE_INTEGRATOR_HPP_
#define LIDAR_LOCALIZATION_MODELS_PRE_INTEGRATOR_ODO_PRE_INTEGRATOR_HPP_

#include "lidar_localization/models/pre_integrator/pre_integrator.hpp"

#include "lidar_localization/sensor_data/velocity_data.hpp"

#include <sophus/so3.hpp>

namespace lidar_localization {

class ODOPreIntegrator : public PreIntegrator {
public:

    typedef Eigen::Matrix<double, 6, 6> MatrixP;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    struct ODOPreIntegration {
        // time delta:
        double T_;

        // a. measurement:
        // a.1. relative translation:
        Eigen::Vector3d alpha_ij_;
        // a.2. relative orientation:
        Sophus::SO3d theta_ij_;

        // b. information:
        MatrixP P_;

        double GetT(void) const { return T_; }
        
        Vector6d GetMeasurement(void) const {
            Vector6d measurement = Vector6d::Zero();

            measurement.block<3, 1>(0, 0) = alpha_ij_;
            measurement.block<3, 1>(3, 0) = theta_ij_.log();

            return measurement;
        }

        Eigen::MatrixXd GetInformation(void) const {
            return P_.inverse();
        }

    };

    ODOPreIntegrator(const YAML::Node& node);

    /**
     * @brief  init ODO pre-integrator
     * @param  init_odo_data, init ODO measurements
     * @return true if success false otherwise
     */
    bool Init(const VelocityData &init_odo_data);

    /**
     * @brief  update ODO pre-integrator
     * @param  odo_data, current ODO measurements
     * @return true if success false otherwise
     */
    bool Update(const VelocityData &odo_data);

    /**
     * @brief  reset ODO pre-integrator using new init ODO measurement
     * @param  init_odo_data, new init ODO measurements
     * @param  output pre-integration result for constraint building as ODOPreIntegration
     * @return true if success false otherwise
     */
    bool Reset(const VelocityData &init_odo_data, ODOPreIntegration &odo_pre_integration);

private:
    typedef Eigen::Matrix<double, 6, 6> MatrixF;
    typedef Eigen::Matrix<double, 6, 6> MatrixB;
    typedef Eigen::Matrix<double, 6, 6> MatrixQ;

    // data buff:
    std::deque<VelocityData> odo_data_buff_;
    // b. prior state covariance, process & measurement noise:
    struct {
        struct {
            double L_VELO;
            double W_VELO;
        } MEASUREMENT;
    } COV;

    // pre-integration state:
    struct {
        // a. relative translation:
        Eigen::Vector3d alpha_ij_;
        // b. relative orientation:
        Sophus::SO3d theta_ij_;
    } pre_int_state;

    // state covariance:
    MatrixP P_ = MatrixP::Zero();

    // process noise:
    MatrixQ Q_ = MatrixQ::Zero();

    // process equation:
    MatrixF F_ = MatrixF::Zero();
    MatrixB B_ = MatrixB::Zero();

    /**
     * @brief  reset pre-integrator state using ODO measurements
     * @param  void
     * @return void
     */
    void ResetState(const VelocityData &init_odo_data);

    /**
     * @brief  update pre-integrator state: mean, covariance and Jacobian
     * @param  void
     * @return void
     */
    void UpdateState(void);
};

} // namespace lidar_localization

#endif // LIDAR_LOCALIZATION_MODELS_PRE_INTEGRATOR_ODO_PRE_INTEGRATOR_HPP_