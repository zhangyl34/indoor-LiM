#include "preprocess.h"

Preprocess::Preprocess()
    : blind(0.01), point_filter_num(1), N_SCANS(6) {
}


// 输入一帧 LiDAR 数据，输出处理后的点云数据
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out) {
    avia_handler(msg);
    *pcl_out = pl_surf;  // 改值
}

// 拿到 livox 原始数据存入 pl_full，间隔采样后存入 pl_surf
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg) {

    pl_surf.clear();              // 清除之前的平面点云缓存
    pl_full.clear();              // 清除之前的全点云缓存
    int plsize = msg->point_num;  // 一帧中的点云总个数
    pl_surf.reserve(plsize);      // 分配空间
    pl_full.resize(plsize);       // 分配空间

    uint valid_num = 0;  // 有效的点云数
    // 分别对每个点云进行处理
    for (uint i = 1; i < plsize; i++) {
        // 只取线数在 0-N_SCANS 内并且回波次序为 1 或者 0 的点云
        if ((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            valid_num++;  // 有效的点云数

            // 等间隔降采样
            if (valid_num % point_filter_num == 0) {
                pl_full[i].x = msg->points[i].x;                                     // 点云 x 轴坐标
                pl_full[i].y = msg->points[i].y;                                     // 点云 y 轴坐标
                pl_full[i].z = msg->points[i].z;                                     // 点云 z 轴坐标
                pl_full[i].intensity = msg->points[i].reflectivity;                  // 点云反射率
                pl_full[i].curvature = msg->points[i].offset_time / float(1000000);  // 把时间信息存在 curvature 里

                // 只有当当前点和上一点的间距足够大（>1e-7），并且在最小距离阈值之外，才将当前点认为是有用的点，加入到 pl_surf 队列中
                if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) 
                    || (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7) && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y > blind)) {
                    pl_surf.push_back(pl_full[i]);
                }
            }
        }
    }
}

void Preprocess::TEST() const {
    std::cout << "***INFO: preprocess arguments***\n";
    std::cout << "point_filter_num: " <<point_filter_num << "; blind: " << blind << "; N_SCANS: " << N_SCANS << std::endl; 
}


