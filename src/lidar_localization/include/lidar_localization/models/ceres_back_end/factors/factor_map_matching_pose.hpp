#pragma once

#include <ceres/ceres.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lidar_localization/tools/math_utils.h"
#include "glog/logging.h"

namespace lidar_localization {

using namespace mathutils;

class FactorMapMatchingPose : public ceres::SizedCostFunction<6, 7> {
public:
    FactorMapMatchingPose(void) {};

    void SetMeasurement(const Eigen::Vector3d &m_pos, Eigen::Quaterniond m_ori) {
        m_pos_ = m_pos;
        m_ori_ = m_ori;
    }

    void SetInformation(const Eigen::MatrixXd &I) {
        I_ = I;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Vector3d P{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(I_).matrixL().transpose();

        const Eigen::Vector3d&    pos_ = m_pos_;
        const Eigen::Quaterniond& rot_ = m_ori_;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.topRows<3>() = P - pos_;
        residual.bottomRows<3>() = 2 * (rot_.inverse() * Q).coeffs().head<3>();

        // FIXME: info
        residual = sqrt_info * residual;
        //DLOG(INFO) << "residual: " << residual.transpose();

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > jacobian_prior(jacobians[0]);
                
                Eigen::Matrix<double, 6, 6> jaco_prior;
                jaco_prior.setIdentity();
                jaco_prior.bottomRightCorner<3, 3>() = LeftQuatMatrix(Q.inverse() * rot_).topLeftCorner<3, 3>();

                // FIXME: info
                jacobian_prior.setZero();
                jacobian_prior.leftCols<6>() = sqrt_info * jaco_prior;
                jacobian_prior.rightCols<1>().setZero();
            }
        }

        return true;
    }


private:
    Eigen::Vector3d    m_pos_;
    Eigen::Quaterniond m_ori_;
    Eigen::MatrixXd I_;
};

}