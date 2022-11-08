#pragma once

#include <string>
#include <deque>
#include <vector>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <lio/Pose6D.h>
#include <sensor_msgs/Imu.h>

#define G_m_s2 (9.81)  // 待统一，IMU_Pkrocessing.cpp 里用到。
#define NUM_MATCH_POINTS (5)  // h_share_model 和拟合平面时使用。

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]

#define DEBUG_FILE_DIR(name)     (std::string(std::string(ROOT_DIR) + "Log/"+ name))

#define common_max(a,b)          ((a) > (b) ? (a) : (b))

typedef lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

#define MD(a,b)  Eigen::Matrix<double,a,b>
#define VD(a)    Eigen::Matrix<double,a,1>
#define MF(a,b)  Eigen::Matrix<float,a,b>
#define VF(a)    Eigen::Matrix<float,a,1>

// 打个括号就报错了。
#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

// Lidar data and imu dates for the curent process
struct MeasureGroup {

    MeasureGroup() {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
};


Pose6D set_pose6d(const double t, const Eigen::Matrix<double, 3, 1> &a, const Eigen::Matrix<double, 3, 1> &g,
    const Eigen::Matrix<double, 3, 1> &v, const Eigen::Matrix<double, 3, 1> &p, const Eigen::Matrix<double, 3, 3> &R);

Eigen::Matrix<double, 3, 3> Exp(const Eigen::Matrix<double, 3, 1> &ang_vel, const double &dt);

bool esti_plane(Eigen::Matrix<float, 4, 1> &pca_result, const PointVector &point, const float &threshold);

float calc_dist(PointType p1, PointType p2);


