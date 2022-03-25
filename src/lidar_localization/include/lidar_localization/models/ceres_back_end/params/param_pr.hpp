#pragma once

#include <ceres/ceres.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lidar_localization/tools/math_utils.h"

namespace lidar_localization {

using namespace mathutils;

class ParamPR : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> _p(x);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

        Eigen::Map<const Eigen::Vector3d> dp(delta);

        Eigen::Quaterniond dq = DeltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (_q * dq).normalized();

        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    }

    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};

}
