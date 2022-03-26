#pragma once

#include <ceres/ceres.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lidar_localization/tools/math_utils.h"
#include "glog/logging.h"

namespace lidar_localization {

using namespace mathutils;

class FactorIMUPreIntegration : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
    enum StateOrder {
        O_P = 0,
        O_R = 3,
        O_V = 6,
        O_BA = 9,
        O_BG = 12,
    };

    FactorIMUPreIntegration(void) {};

	void SetT(const double &T) {
		T_ = T;
	}

	void SetGravitiy(const Eigen::Vector3d &g) {
		g_vec_ = g;
	}

    void SetMeasurement(const Eigen::Vector3d &alpha, const Eigen::Quaterniond &theta, const Eigen::Vector3d &beta) {
		alpha_ = alpha;
        theta_ = theta;
        beta_  = beta;
	}

    void SetInformation(const Eigen::MatrixXd &I) {
        I_ = I;
    }

	void SetLinearizedBa(const Eigen::Vector3d &linearized_ba) {
		linearized_ba_ = linearized_ba;
	}

    void SetLinearizedBg(const Eigen::Vector3d &linearized_bg) {
		linearized_bg_ = linearized_bg;
	}

	void SetJacobian(const Eigen::MatrixXd &J) {
		J_ = J;
	}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        // residual = RedidualEvaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix3d dp_dba = J_.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = J_.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = J_.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = J_.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = J_.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba_;
        Eigen::Vector3d dbg = Bgi - linearized_bg_; 

        Eigen::Quaterniond corrected_delta_q = theta_ * DeltaQ(dq_dbg * dbg);
        Eigen::Vector3d    corrected_delta_v = beta_ + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d    corrected_delta_p = alpha_ + dp_dba * dba + dp_dbg * dbg;

        residual.block<3, 1>(O_P, 0) = Qi.inverse() * (-0.5 * g_vec_ * T_ * T_ + Pj - Pi - Vi * T_) - corrected_delta_p;
        residual.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residual.block<3, 1>(O_V, 0) = Qi.inverse() * (-g_vec_ * T_ + Vj - Vi) - corrected_delta_v;
        residual.block<3, 1>(O_BA, 0) = Baj - Bai;
        residual.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(I_).matrixL().transpose();

        residual = sqrt_info * residual;

        if (jacobians) {
            Eigen::Matrix3d dp_dba = J_.block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = J_.block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = J_.block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = J_.block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = J_.block<3, 3>(O_V, O_BG);

            if (J_.maxCoeff() > 1e8 || J_.minCoeff() < -1e8) {
               // ROS_DEBUG("numerical unstable in preintegration");
            }

            if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();

                    jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                    jacobian_pose_i.block<3, 3>(O_P, O_R) = SkewSymmetric(Qi.inverse() * (-0.5 * g_vec_ * T_ * T_ + Pj - Pi - Vi * T_));

                    Eigen::Quaterniond corrected_delta_q  = theta_ * DeltaQ(dq_dbg * (Bgi - linearized_bg_));
                    jacobian_pose_i.block<3, 3>(O_R, O_R) = -(LeftQuatMatrix(Qj.inverse() * Qi) * RightQuatMatrix(corrected_delta_q)).topLeftCorner<3, 3>();
                    jacobian_pose_i.block<3, 3>(O_V, O_R) = SkewSymmetric(Qi.inverse() * (-g_vec_ * T_ + Vj - Vi));

                    jacobian_pose_i = sqrt_info * jacobian_pose_i;

                    if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
                        //ROS_DEBUG("numerical unstable in preintegration");
                    }
            }
            
            if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                    jacobian_speedbias_i.setZero();
                    jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * T_;
                    jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                    jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

                    Eigen::Quaterniond corrected_delta_q = theta_ * DeltaQ(dq_dbg * (Bgi - linearized_bg_));
                    jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -LeftQuatMatrix(Qj.inverse() * Qi * corrected_delta_q).topLeftCorner<3, 3>() * dq_dbg;

                    jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                    jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                    jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
                    jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
                    jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
                    jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
            }

            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
                Eigen::Quaterniond corrected_delta_q = theta_ * DeltaQ(dq_dbg * (Bgi - linearized_bg_));
                jacobian_pose_j.block<3, 3>(O_R, O_R) = LeftQuatMatrix(corrected_delta_q.inverse() * Qi.inverse() * Qj).topLeftCorner<3, 3>();
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }

            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();
                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

            }
    }

    return true;
  }

private:

    double T_ = 0.0;

	Eigen::Vector3d g_vec_ = Eigen::Vector3d::Zero();

    Eigen::Vector3d linearized_ba_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d linearized_bg_ = Eigen::Vector3d::Zero();

    Eigen::Vector3d    alpha_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond theta_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d    beta_  = Eigen::Vector3d::Zero();

    Eigen::MatrixXd I_;

    Eigen::MatrixXd J_;
};

} // namespace 

