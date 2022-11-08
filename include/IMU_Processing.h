#pragma once

#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>
#include <Eigen/Eigen>


#include "common_lib.h"
#include "use-ikfom.h"

#define IMU_MAX_INI_COUNT (10)

// IMU forward propagation and backward undistortion
class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Eigen 在 arm 架构下的 bug

    ImuProcess();
    ~ImuProcess() {};

    void Reset();  // 有待统一
    // void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
    void set_extrinsic(const V3D &transl, const M3D &rot) {Lidar_T_wrt_IMU = transl; Lidar_R_wrt_IMU = rot;};
    // void set_extrinsic(const V3D &transl);
    // void set_extrinsic(const MD(4, 4) & T);
    void set_gyr_cov(const V3D &scaler) {cov_gyr_scale = scaler;};
    void set_acc_cov(const V3D &scaler) {cov_acc_scale = scaler;};
    void set_gyr_bias_cov(const V3D &b_g) {cov_bias_gyr = b_g;};
    void set_acc_bias_cov(const V3D &b_a) {cov_bias_acc = b_a;};

    void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

private:
    void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);
    void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);


    sensor_msgs::ImuConstPtr last_imu_;  // 上一帧 IMU
    std::vector<Pose6D> IMUpose;         // IMU 位姿
    M3D Lidar_R_wrt_IMU;                 // （set），lidar 到 IMU 的旋转外参
    V3D Lidar_T_wrt_IMU;                 // （set），lidar 到 IMU 的位置外参
    V3D mean_acc;                        // 加速度均值，用于计算方差
    V3D mean_gyr;                        // 角速度均值，用于计算方差
    V3D angvel_last;                     // 上一帧角速度
    V3D acc_s_last;                      // 上一帧加速度
    double last_lidar_end_time_;         // 上一帧结束时间戳
    int init_iter_num = 1;               // 初始化迭代次数
    bool b_first_frame_ = true;          // 是否是第一帧
    bool imu_need_init_ = true;          // 是否需要初始化 IMU

    // 曾经属于 public
    Eigen::Matrix<double, 12, 12> Q;
    std::ofstream fout_imu;  // IMU 参数输出文件
    V3D cov_acc;             // 由测量结果计算的协方差
    V3D cov_gyr;             // 由测量结果计算的协方差
    V3D cov_acc_scale;       // （set），协方差
    V3D cov_gyr_scale;       // （set），协方差
    V3D cov_bias_gyr;        // （set），协方差
    V3D cov_bias_acc;        // （set），协方差
};




