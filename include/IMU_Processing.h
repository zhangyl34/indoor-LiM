#pragma once

#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <file_logger.h>

#include "common_lib.h"
#include "use-ikfom.h"

#define IMU_MAX_INI_COUNT (10)

// IMU forward propagation and backward undistortion
class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Eigen 在 arm 架构下的 bug

    ImuProcess();
    ~ImuProcess() {};

    /* 更改内参。*/
    void set_extrinsic(const V3D &transl, const M3D &rot) {Lidar_T_wrt_IMU = transl; Lidar_R_wrt_IMU = rot;};
    void set_gyr_cov(const V3D &cov) {cov_gyr = cov;};
    void set_acc_cov(const V3D &cov) {cov_acc = cov;};
    void set_gyr_bias_cov(const V3D &b_g) {cov_bias_gyr = b_g;};
    void set_acc_bias_cov(const V3D &b_a) {cov_bias_acc = b_a;};

    /* 读取内参。*/
    V3D get_mean_acc() const {return mean_acc;};
    M3D get_R_W_G() const {return R_W_G;};

    void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

private:
    void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);
    void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

    int init_iter_num;    // 初始化迭代次数
    bool imu_need_init_;  // 是否需要初始化 IMU

    V3D mean_acc;         // 加速度均值，用于初始化 init_state
    V3D mean_gyr;         // 角速度均值，用于初始化 init_state
    M3D Lidar_R_wrt_IMU;  // （set），lidar 到 IMU 的旋转外参
    V3D Lidar_T_wrt_IMU;  // （set），lidar 到 IMU 的位置外参

    V3D cov_bias_gyr;                 // （set），协方差
    V3D cov_bias_acc;                 // （set），协方差
    V3D cov_acc;                      // （set），协方差
    V3D cov_gyr;                      // （set），协方差
    Eigen::Matrix<double, 12, 12> Q;  // 噪声 w 的协方差矩阵

    sensor_msgs::ImuConstPtr last_imu_;  // 上一包末尾的 IMU
    V3D acc_s_last;                      // 上一帧加速度
    V3D angvel_last;                     // 上一帧角速度
    double last_lidar_end_time_;         // 上一包雷达结束时间戳

    M3D R_W_G;  // 计算 G^R_W，用于储存地图时，地图能够平行于 ground
};




