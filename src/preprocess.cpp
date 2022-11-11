#include "preprocess.h"

Preprocess::Preprocess()
    : blind(0.01), point_filter_num(1), N_SCANS(6) {
}

// 输入一帧 LiDAR 数据，输出处理后的点云数据
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr pl_surf) {

    int plsize = msg->point_num;     // 一帧中的点云总个数
    (*pl_surf).clear();              // 清除之前的平面点云缓存
    (*pl_surf).reserve(plsize);      // 分配空间

    PointType plSurf_last;
    PointType plSurf_current;
    int valid_num = 0;  // 有效的点云数
    // 分别对每个点云进行处理
    for (int i = 0; i < plsize; i++) {
        // 只取线数在 0-N_SCANS 内并且回波次序为 1 或者 0 的点云
        if ((msg->points[i].line < N_SCANS) && 
            ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            
            if (valid_num % point_filter_num == 0) {
                plSurf_current.x = msg->points[i].x;                                     // 点云 x 轴坐标
                plSurf_current.y = msg->points[i].y;                                     // 点云 y 轴坐标
                plSurf_current.z = msg->points[i].z;                                     // 点云 z 轴坐标
                plSurf_current.intensity = msg->points[i].reflectivity;                  // 点云反射率
                plSurf_current.curvature = msg->points[i].offset_time / float(1000000);  // 把时间信息存在 curvature 里

                // 只有当当前点和上一点的间距足够大（>1e-7），并且在最小距离阈值之外，才将当前点认为是有用的点，加入到 pl_surf 队列中
                if ((abs(plSurf_current.x - plSurf_last.x) > 1e-7) || (abs(plSurf_current.y - plSurf_last.y) > 1e-7) 
                    || (abs(plSurf_current.z - plSurf_last.z) > 1e-7) && (plSurf_current.x * plSurf_current.x + plSurf_current.y * plSurf_current.y > blind)) {
                    
                    plSurf_last = plSurf_current;
                    (*pl_surf).push_back(plSurf_current);
                }
            }
            valid_num ++;  // 等间隔采样
        }
    }
}

