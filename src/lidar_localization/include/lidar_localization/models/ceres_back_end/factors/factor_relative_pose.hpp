#pragma once

#include <ceres/ceres.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lidar_localization/tools/math_utils.h"
#include "glog/logging.h"

namespace lidar_localization {

using namespace mathutils;

class FactorRelativePose : public ceres::SizedCostFunction<6, 7, 7> {
public:

    FactorRelativePose(void) {};

    void SetMeasurement(const Eigen::Vector3d &m_pos, Eigen::Quaterniond m_ori) {
        m_pos_ = m_pos;
        m_ori_ = m_ori;
    }

    void SetInformation(const Eigen::MatrixXd &I) {
        I_ = I;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // pose i
        Eigen::Vector3d Pi{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        // pose j
        Eigen::Vector3d Pj{parameters[1][0], parameters[1][1], parameters[1][2]};
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(I_).matrixL().transpose();

        const Eigen::Vector3d&    pos_ij = m_pos_;
        const Eigen::Quaterniond& rot_ij = m_ori_;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        Eigen::Matrix3d r_Qi = Qi.inverse().toRotationMatrix();
        residual.topRows<3>() = r_Qi * (Pj - Pi) - pos_ij;
        residual.bottomRows<3>() = 2 * (rot_ij.inverse() * Qi.inverse() * Qj).vec();

        // FIXME: info
        residual = sqrt_info * residual;
        //DLOG(INFO) << "residual: " << residual.transpose();

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_i(jacobians[0]);

                jacobian_i.setZero();
                jacobian_i.block<3, 3>(0, 0) = -r_Qi;
                jacobian_i.block<3, 3>(0, 3) = -SkewSymmetric(Qi.inverse() * (Pj - Pi) - pos_ij);
                jacobian_i.block<3, 3>(3, 3) = -(LeftQuatMatrix(Qj.inverse() * Qi) * RightQuatMatrix(rot_ij)).topLeftCorner<3, 3>();

                jacobian_i = sqrt_info * jacobian_i;
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_j(jacobians[1]);

                jacobian_j.setZero();
                jacobian_j.block<3, 3>(0, 0) = r_Qi;
                jacobian_j.block<3, 3>(3, 3) = LeftQuatMatrix(rot_ij.inverse() * Qi.inverse() * Qj).topLeftCorner<3, 3>();

                jacobian_j = sqrt_info * jacobian_j;
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