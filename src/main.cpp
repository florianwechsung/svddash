#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <tuple>
#include <math.h>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
// ################################################################################
// ######### Interfacing numpy and xtensor without copying data ###################
// ################################################################################

#define FORCE_IMPORT_ARRAY
//#include "xtensor/xarray.hpp"
#include "xtensor-python/pytensor.hpp"
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyMatrix;
typedef xt::pytensor<double, 4, xt::layout_type::row_major> PyTensor;

//double phi(double t){
    //return 0.5 * t * t;
//}

//double phidash(double t){
    //return t;
//}

//double phidashdash(double t){
    //return 1;
//}

//double phi(double t){
//    return (1./6) * t * t * t;
//}

//double phidash(double t){
//    return (1./2) * t * t;
//}

//double phidashdash(double t){
//    return t;
//}

double phi(double t, double alpha) {
    return 0.5 * pow(std::cosh(alpha*t)-1, 2);
}

double phidash(double t, double alpha) {
    return alpha * (std::cosh(alpha*t)-1) * std::sinh(alpha*t);
}

double phidashdash(double t, double alpha) {
    return alpha * alpha * (std::sinh(alpha*t) * std::sinh(alpha*t) + (std::cosh(alpha*t)-1) * std::cosh(alpha*t));
}

std::tuple<double, PyMatrix, PyTensor> compute(PyMatrix& m, double threshold, double alpha){
    int d = m.shape(0);
    Matrix M(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            M(i, j) = m(i, j);
        }
    }
    Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix U = svd.matrixU();
    Vector S = svd.singularValues();
    Matrix V = svd.matrixV();

    Vector Smax(d);
    for (int i = 0; i < d; ++i) {
        Smax(i) = std::max(S(i)-threshold, 0.);
    }
    double value = 0.;
    for (int i = 0; i < d; ++i) {
        value += phi(Smax(i), alpha);
    }

    Vector PhiDashSmax(d);
    for (int i = 0; i < d; ++i) {
        PhiDashSmax(i) = phidash(Smax(i), alpha);
    }

    Matrix Deriv = U * PhiDashSmax.asDiagonal() * V.transpose();
    PyMatrix PyDeriv = xt::zeros<double>({d, d});
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            PyDeriv(i, j) = Deriv(i, j);
        }
    }
    PyTensor PyDeriv2 = xt::zeros<double>({d, d, d, d});
    for (int m = 0; m < d; ++m) {
        for (int n = 0; n < d; ++n) {
            Matrix Deriv2 = Matrix::Zero(d, d);
            Matrix Direction = Matrix::Zero(d, d);
            Direction(m, n) = 1.;
            Matrix dU = Matrix::Zero(d, d);
            Vector dS = Vector::Zero(d);
            Matrix dV = Matrix::Zero(d, d);
            for (int i = 0; i < d; ++i) {
                Vector ui = U.col(i);
                Vector vi = V.col(i);
                if (S(i) <= threshold)
                    continue;
                double lami = S(i)*S(i);
                Matrix mat1 = Direction.transpose() * M + M.transpose() * Direction;
                Matrix mat2 = Direction * M.transpose() + M * Direction.transpose();
                dS(i) = phidashdash(Smax(i), alpha) * (ui.transpose() * (Direction * V))(i);
                Matrix matvi = (mat1) * vi;
                Matrix matui = (mat2) * ui;
                for (int j = 0; j < d; ++j) {
                    if (i == j)
                        continue;
                    if (std::abs(S(j)-S(i)) < 1e-16)
                        continue;
                    double lamj = S(j)*S(j);
                    Vector uj = U.col(j);
                    Vector vj = V.col(j);
                    dV.col(i) += (vj * vj.transpose()) * matvi/(lami-lamj);
                    dU.col(i) += (uj * uj.transpose()) * matui/(lami-lamj);
                }
            }
            Deriv2 += (
                      dU * PhiDashSmax.asDiagonal() * V.transpose()
                    + U * dS.asDiagonal() * V.transpose()
                    + U * PhiDashSmax.asDiagonal() * dV.transpose()
                    );

            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    PyDeriv2(m, n, i, j) = Deriv2(i, j);
                }
            }
        }
    }
    return std::make_tuple(value, PyDeriv, PyDeriv2);
}

namespace py = pybind11;

PYBIND11_MODULE(svddash, m) {
    xt::import_numpy();

    m.def("compute", &compute);
}
