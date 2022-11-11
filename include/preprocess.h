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
    
    // 对 Livox 自定义 Msg 格式的激光雷达数据进行处理，降采样后的有效点存入 pl_surf 所指的地址。
    void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr pl_surf);
    
private:

    int point_filter_num;    // 采样间隔，即每隔 point_filter_num 个点取 1 个点
    double blind;            // 最小距离阈值，即过滤掉 0-blind 范围内的点云
    int N_SCANS;             // 扫描线数
};




