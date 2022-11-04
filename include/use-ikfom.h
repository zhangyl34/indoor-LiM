#pragma once

/* 采用 3.Remakrs(1) 的方案 https://github.com/hku-mars/IKFoM*/
/* ikfom 第一步，包含头文件*/
#include <IKFoM_toolkit/esekfom/esekfom.hpp>

/* ikfom 第二步，定义基础流形。*/
typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
// 参数 1 在计算 B 的时候发挥作用。将 x 轴转至切空间的法向。
typedef MTK::S2<double, 98090, 10000, 1> S2;

/* ikfom 第三步，定义复合流形。*/
MTK_BUILD_MANIFOLD(state_ikfom,
    ((vect3, pos))
    ((SO3, rot))
    ((SO3, offset_R_L_I))
    ((vect3, offset_T_L_I))
    ((vect3, vel))
    ((vect3, bg))
    ((vect3, ba))
    ((S2, grav)));

MTK_BUILD_MANIFOLD(input_ikfom,
    ((vect3, acc))
    ((vect3, gyro)));

MTK_BUILD_MANIFOLD(process_noise_ikfom,
    ((vect3, ng))
    ((vect3, na))
    ((vect3, nbg))
    ((vect3, nba)));

/* ikfom 第四步，定义正向传播会用到的函数。*/
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in);

/* 初始化协方差矩阵,在 IMU_Processing.hpp 文件中用到。*/
MTK::get_cov<process_noise_ikfom>::type process_noise_cov();




