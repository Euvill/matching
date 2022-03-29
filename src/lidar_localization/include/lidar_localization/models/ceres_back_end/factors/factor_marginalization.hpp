#pragma once

#include <ceres/ceres.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace lidar_localization {

class FactorMarginalization : public ceres::SizedCostFunction<15, 7, 9> {
public:
  FactorMarginalization(void) {
    H_ = Eigen::MatrixXd::Zero(14, 14);
    b_ = Eigen::VectorXd::Zero(14);
    J_ = Eigen::MatrixXd::Zero(7, 7);
    e_ = Eigen::VectorXd::Zero(7);

    Hvb_ = Eigen::MatrixXd::Zero(18, 18);
    bvb_ = Eigen::VectorXd::Zero(18);
    Jvb_ = Eigen::MatrixXd::Zero(9, 9);
    evb_ = Eigen::VectorXd::Zero(9);
  }

  void SetResMapMatchingPose(const ceres::CostFunction *residual, const std::vector<double *> &parameter_blocks) {
    // init:
    ResidualBlockInfo res_map_matching_pose(residual, parameter_blocks);
    Eigen::VectorXd residuals;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    // compute: res_{6x1}, jaco_{6x7}
    Evaluate(res_map_matching_pose, residuals, jacobians);

    const Eigen::MatrixXd &J_m = jacobians.at(0);

    H_.block<7, 7>(0, 0) += J_m.transpose() * J_m;
    b_.block<7, 1>(0, 0) += J_m.transpose() * residuals;
  }

  void SetResRelativePose(const ceres::CostFunction *residual,const std::vector<double *> &parameter_blocks) {
    // init:
    ResidualBlockInfo res_relative_pose(residual, parameter_blocks);
    Eigen::VectorXd residuals;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    // compute: res_{6x1}, jaco1_{6x7}, jaco2_{6x7}
    Evaluate(res_relative_pose, residuals, jacobians);
    const Eigen::MatrixXd &J_m = jacobians.at(0);
    const Eigen::MatrixXd &J_r = jacobians.at(1);

    H_.block<7, 7>(0, 0) += J_m.transpose() * J_m;
    H_.block<7, 7>(0, 7) += J_m.transpose() * J_r;
    H_.block<7, 7>(7, 0) += H_.block<7, 7>(0, 7).transpose();
    H_.block<7, 7>(7, 7) += J_r.transpose() * J_r;

    b_.block<7, 1>(0, 0) += J_m.transpose() * residuals;
    b_.block<7, 1>(7, 0) += J_r.transpose() * residuals;
  }

  void SetResIMUPreIntegration(const ceres::CostFunction *residual, const std::vector<double *> &parameter_blocks) {
    // init:
    ResidualBlockInfo res_imu_pre_integration(residual, parameter_blocks);
    Eigen::VectorXd residuals;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    
    // compute: res_{15x1}, jaco1_{15x7}, jaco2_{15x9}, jaco3_{15x7}, jaco4_{15x9}
    Evaluate(res_imu_pre_integration, residuals, jacobians);

    // pose and ori
    const Eigen::MatrixXd &J1_m = jacobians.at(0);
    const Eigen::MatrixXd &J1_r = jacobians.at(2);
    // speed and bias
    const Eigen::MatrixXd &J2_m = jacobians.at(1);
    const Eigen::MatrixXd &J2_r = jacobians.at(3);
    
    H_.block<7, 7>(0, 0) += J1_m.transpose() * J1_m;
    H_.block<7, 7>(0, 7) += J1_m.transpose() * J1_r;
    H_.block<7, 7>(7, 0) += H_.block<7, 7>(0, 7).transpose();
    H_.block<7, 7>(7, 7) += J1_r.transpose() * J1_r;

    b_.block<7, 1>(0, 0) += J1_m.transpose() * residuals;
    b_.block<7, 1>(7, 0) += J1_r.transpose() * residuals;
    
    Hvb_.block<9, 9>(0, 0) += J2_m.transpose() * J2_m;
    Hvb_.block<9, 9>(0, 9) += J2_m.transpose() * J2_r;
    Hvb_.block<9, 9>(9, 0) += Hvb_.block<9, 9>(0, 9).transpose();
    Hvb_.block<9, 9>(9, 9) += J2_r.transpose() * J2_r;

    bvb_.block<9, 1>(0, 0) += J2_m.transpose() * residuals;
    bvb_.block<9, 1>(9, 0) += J2_r.transpose() * residuals;
  }

  void Marginalize(const double *raw_param_r_0, const double *raw_param_r_1) {

    Eigen::Vector3d p_i(raw_param_r_0[0], raw_param_r_0[1], raw_param_r_0[2]);
    Eigen::Quaterniond q_i(raw_param_r_0[6], raw_param_r_0[3], raw_param_r_0[4], raw_param_r_0[5]);

    Eigen::Vector3d v_i(raw_param_r_1[0], raw_param_r_1[1], raw_param_r_1[2]);
    Eigen::Vector3d ba_i(raw_param_r_1[3], raw_param_r_1[4], raw_param_r_1[5]);
    Eigen::Vector3d bg_i(raw_param_r_1[6], raw_param_r_1[7], raw_param_r_1[8]);

    p_i_  = p_i;
    q_i_  = q_i;
    v_i_  = v_i;
    ba_i_ = ba_i;
    bg_i_ = bg_i;

    Marginalized_process(H_, b_, J_, e_);

    Marginalized_process(Hvb_, bvb_, Jvb_, evb_);
  }

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {	
    // parse parameters:
    Eigen::Vector3d    Pj{parameters[0][0], parameters[0][1], parameters[0][2]};
    Eigen::Quaterniond Qj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Vj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Baj(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Vector3d Bgj(parameters[1][6], parameters[1][7], parameters[1][8]);
    
    Eigen::VectorXd dx(7);
    dx.setZero();
    dx.segment<3>(0) = Pj - p_i_;

    Eigen::Quaterniond dq = q_i_.inverse() * Qj;
    if(dq.w() < 0)
        dx.segment<3>(3) = -2.0 * (dq).normalized().vec();
    else
        dx.segment<3>(3) = 2.0 * (dq).normalized().vec();

    Eigen::VectorXd dvb(9);
    dvb.setZero();
    dvb.segment<3>(0) = Vj  - v_i_;
    dvb.segment<3>(3) = Baj - ba_i_;
    dvb.segment<3>(6) = Bgj - bg_i_;

    // compute residual:
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual.block<6, 1>(0, 0) = (e_ + J_ * dx).block<6, 1>(0, 0);
    residual.block<9, 1>(6, 0) = evb_ + Jvb_ * dvb;

    std::cout << std::endl;
    std::cout << "residual: " << std::endl;
    std::cout << residual(0, 0) << "," << residual(1, 0) << "," << residual(2, 0) << std::endl
              << residual(3, 0) << "," << residual(4, 0) << "," << residual(5, 0) << std::endl
              << residual(6, 0) << "," << residual(7, 0) << "," << residual(8, 0) << std::endl
              << residual(9, 0) << "," << residual(10,0) << "," << residual(11,0) << std::endl
              << residual(12, 0)<< "," << residual(13,0) << "," << residual(14,0) << std::endl;
    std::cout << std::endl;

    // compute jacobian:
    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pr(jacobians[0]);
            jacobian_pr.setZero();
            jacobian_pr.block<7, 7>(0, 0) = J_;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_vag(jacobians[1]);
            jacobian_vag.setZero();
            jacobian_vag.block<9, 9>(6, 0) = Jvb_;
        } 
    }

    return true;
  }

private:
    Eigen::MatrixXd H_;
    Eigen::VectorXd b_;
    Eigen::MatrixXd J_;
    Eigen::VectorXd e_;

    Eigen::Vector3d p_i_;
    Eigen::Quaterniond q_i_;
    Eigen::Vector3d v_i_;
    Eigen::Vector3d ba_i_;
    Eigen::Vector3d bg_i_;

    Eigen::MatrixXd Hvb_;
    Eigen::VectorXd bvb_;
    Eigen::MatrixXd Jvb_;
    Eigen::VectorXd evb_;

    struct ResidualBlockInfo {
      const ceres::CostFunction *residual = nullptr;
      std::vector<double *> parameter_blocks;

      ResidualBlockInfo(void) {}

      ResidualBlockInfo(const ceres::CostFunction *_residual, const std::vector<double *> &_parameter_blocks) : residual(_residual), parameter_blocks(_parameter_blocks) {}
    };

    void Marginalized_process(const Eigen::MatrixXd& Hin, const Eigen::VectorXd& bin, Eigen::MatrixXd& result_J, Eigen::VectorXd& result_e){
      const int m = Hin.rows() / 2;
       
      Eigen::MatrixXd H_mm = 0.5 * (Hin.block(0, 0, m, m) + Hin.block(0, 0, m, m).transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H_mm);
      Eigen::MatrixXd H_mm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > 1.0e-8).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

      Eigen::VectorXd b_m  = bin.segment(0, m);
      Eigen::MatrixXd H_mr = Hin.block(0, m, m, m);
      Eigen::MatrixXd H_rm = Hin.block(m, 0, m, m);
      Eigen::MatrixXd H_rr = Hin.block(m, m, m, m);
      Eigen::VectorXd b_r  = bin.segment(m, m);

      Eigen::MatrixXd H_marginalized = H_rr - H_rm * H_mm_inv * H_mr;
      Eigen::MatrixXd b_marginalized = b_r  - H_rm * H_mm_inv * b_m;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_marginalized);
      Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > 1.0e-8).select(saes2.eigenvalues().array(), 0));
      Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > 1.0e-8).select(saes2.eigenvalues().array().inverse(), 0));

      Eigen::VectorXd S_sqrt = S.cwiseSqrt();
      Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

      result_J = S_sqrt.asDiagonal() * saes.eigenvectors().transpose();
      result_e = S_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose() * b_marginalized;
    }

    static void Evaluate(ResidualBlockInfo &residual_info, Eigen::VectorXd &residuals, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &jacobians) {
        // init residual output:
        const int D = static_cast<int>(residual_info.residual->num_residuals());
        residuals.resize(D);

        // init jacobians output:
        std::vector<int> block_sizes = residual_info.residual->parameter_block_sizes();
        const int N = static_cast<int>(block_sizes.size());

        double **raw_jacobians = new double *[N];
        jacobians.resize(N);

        // create raw pointer adaptor:
        for (int i = 0; i < N; i++) {
            jacobians[i].resize(D, block_sizes[i]);
            raw_jacobians[i] = jacobians[i].data();
        }

        residual_info.residual->Evaluate(residual_info.parameter_blocks.data(), residuals.data(), raw_jacobians);
    }
};

} 