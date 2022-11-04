#pragma once

#include <livox_ros_driver/CustomMsg.h>
#include <iostream>

#include "common_lib.h"

class Preprocess
{
public:
    Preprocess();
    ~Preprocess() {};

    void set_blind(const double b) {blind = b;};
    void set_N_SCANS(const int ns) {N_SCANS = ns;};
    void set_point_filter_num(const int pfn) {point_filter_num = pfn;};
    void TEST() const;
    
    // 对 Livox 自定义 Msg 格式的激光雷达数据进行处理
    void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    
private:
    // 对 Livox 激光雷达数据进行处理
    void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    //void pub_func(PointCloudXYZI &pl, const ros::Time &ct);

    // 曾经属于 public
    PointCloudXYZI pl_full;  // 全部点
    PointCloudXYZI pl_surf;  // 平面点
    int point_filter_num;    // 采样间隔，即每隔 point_filter_num 个点取 1 个点
    double blind;            // 最小距离阈值，即过滤掉 0-blind 范围内的点云
    int N_SCANS;             // 扫描线数
    // ros::Publisher pub_full, pub_surf, pub_corn;  // 发布全部点、发布平面点、发布边缘点
};




