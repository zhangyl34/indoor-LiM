#include <signal.h>
#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <mutex>
#include <condition_variable>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ikd_Tree.h>
#include <file_logger.h>

#include "common_lib.h"
#include "IMU_Processing.h"
#include "preprocess.h"
#include "use-ikfom.h"

#define _MOV_THRESHOLD   (1.5f)  // 当前雷达系中心到各个地图边缘的权重
#define _LASER_POINT_COV (0.001)
#define _INIT_TIME       (0.1)

/* 回调函数中使用的全局变量。*/
std::shared_ptr<Preprocess> p_pre(new Preprocess());  // 定义指向激光雷达数据的预处理类 Preprocess 的指针
std::mutex mtx_buffer;                                // 互斥锁
std::condition_variable sig_buffer;                   // 信号容器
double last_timestamp_lidar = 0.0;                    // 上一帧 LiDAR 数据的起始时间戳
double last_timestamp_imu = 0.0;                      // 上一帧 IMU 数据的时间戳
std::deque<double> time_buffer;                       // LiDAR 数据起始时间戳缓存队列
std::deque<PointCloudXYZI::Ptr> lidar_buffer;         // 记录间隔采样后的 LiDAR 数据
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;    // IMU 数据缓存队列

/* 中断函数中使用的全局变量。*/
bool flg_exit = false;

/* h_share_mode 中使用的全局变量。*/
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());   // 校畸变降采样后的单帧点云，lidar 系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());  // 校畸变降采样后的单帧点云，w 系
std::vector<PointVector> Nearest_Points;
KD_TREE<PointType> ikdtree;

// 收到中断信号后，会唤醒所有等待队列中阻塞的线程
// 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

// 把点从 Lidar 系转到 world 系
void pointBodyToWorld(PointType const *const pi, PointType *const po, const state_ikfom &sp)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(sp.rot * (sp.offset_R_L_I * p_body + sp.offset_T_L_I) + sp.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 动态调整地图区域，防止地图过大而内存溢出。
void lasermap_fov_segment(const V3D& pos_LiD, const float& cube_len, const float& det_range) {

    static bool Localmap_Initialized = false;
    static BoxPointType LocalMap_Points;  // ikd-tree 中,局部地图的包围盒角点

    // 初始化局部地图包围盒角点，以为 w 系下 lidar 位置为中心，得到长宽高 1000*1000*1000 的局部地图
    if (!Localmap_Initialized) {
        // 系统起始需要初始化局部地图的大小和位置
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 当前雷达系中心到局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离 <=1.5*450m，标记需要移除，参考论文 Fig3
        if ((dist_to_map_edge[i][0] <= _MOV_THRESHOLD * det_range) ||
            (dist_to_map_edge[i][1] <= _MOV_THRESHOLD * det_range)) {

            need_move = true;
        }
    }
    // 不需要挪动就直接退回了
    if (!need_move) {
        return;
    }

    // 否则计算新的局部地图边界
    std::vector<BoxPointType> cub_needrm;  // ikd-tree 中，地图需要移除的包围盒序列
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = det_range * (_MOV_THRESHOLD - 1);
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        // 与包围盒最小值边界点距离
        if (dist_to_map_edge[i][0] <= _MOV_THRESHOLD * det_range) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);  // need remove.
        }
        else if (dist_to_map_edge[i][1] <= _MOV_THRESHOLD * det_range) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;
    neal::logger(neal::LOG_INFO, "update local map box...");

    // 删除指定盒内的点
    if (cub_needrm.size() > 0) {
        ikdtree.Delete_Point_Boxes(cub_needrm);
    }
}

/* 订阅器 sub_pcl 的回调函数。
接收 Livox 的点云数据，对点云数据进行预处理，并将处理后的数据保存到激光雷达数据队列中*/
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {

    std::unique_lock<std::mutex> locker(mtx_buffer, std::defer_lock);

    locker.lock();
    // 如果当前帧 LiDAR 数据的时间戳比上一帧 LiDAR 数据的时间戳早，需要将激光雷达数据缓存队列清空
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        neal::logger(neal::LOG_ERROR, "lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    //neal::logger(neal::LOG_INFO, "receive pcl data, timestamp: " + std::to_string(last_timestamp_lidar));

    // 如果不需要进行时间同步，而 IMU 时间戳和雷达时间戳相差大于 10s，则输出错误信息
    if (fabs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
        std::string strout;
        strout = "IMU and LiDAR not Synced, IMU time: " + std::to_string(last_timestamp_imu) +
            ", lidar scan end time: " + std::to_string(last_timestamp_lidar) + '.';
        neal::logger(neal::LOG_ERROR, strout);
    }

    // 用 pcl 点云格式保存接收到的激光雷达数据
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    // 对激光雷达数据进行预处理
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    locker.unlock();
    sig_buffer.notify_all();  // 唤醒阻塞的线程
}

/* 订阅器 sub_imu 的回调函数。
接收 IMU 数据，将 IMU 数据保存到 IMU 数据缓存队列中*/
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {

    std::unique_lock<std::mutex> locker(mtx_buffer, std::defer_lock);
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();  // IMU 起始时间戳
    locker.lock();
    // 如果当前 IMU 的时间戳小于上一个时刻 IMU 的时间戳，则 IMU 数据有误，将 IMU 数据缓存队列清空
    if (timestamp < last_timestamp_imu) {
        std::string strout = "imu loop back, clear buffer";
        neal::logger(neal::LOG_ERROR, strout);
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;
    //neal::logger(neal::LOG_INFO, "receive imu data, timestamp: " + std::to_string(last_timestamp_imu));

    // 将当前的 IMU 数据保存到 IMU 数据缓存队列中
    imu_buffer.push_back(msg);
    locker.unlock();
    sig_buffer.notify_all();  // 唤醒阻塞的线程
}

/* 将第一帧 LiDAR 数据，和这段时间内的 IMU 数据从缓存队列中取出，并保存到 meas 中*/
bool sync_packages(MeasureGroup &meas) {
    
    static int scan_num = 0;
    static double lidar_mean_scantime = 0.0;
    static bool lidar_pushed = false;  // LiDAR 数据是否已经存入 meas 中

    // 如果缓存队列中没有数据，则返回 false
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    // 如果 LiDAR 数据尚未存入
    if (!lidar_pushed) {
        // 从 LiDAR 点云缓存队列中取出点云数据，放到 meas 中
        meas.lidar = lidar_buffer.front();
        lidar_buffer.pop_front();  // 将 LiDAR 数据弹出
        // 当前帧 LiDAR 数据起始的时间戳
        meas.lidar_beg_time = time_buffer.front();
        time_buffer.pop_front();   // 将时间戳弹出
        double duration = meas.lidar->points.back().curvature / 1000.0;
        // 如果该数据没有点云
        if (meas.lidar->points.size() <= 1) {
            meas.lidar_end_time = meas.lidar_beg_time + 0.0;
            neal::logger(neal::LOG_WARN, "Too few input point cloud!");
        }
        // 如果扫描用时不正常
        else if (duration < 0.5 * lidar_mean_scantime) {
            meas.lidar_end_time = meas.lidar_beg_time + duration;
            neal::logger(neal::LOG_WARN, "Too short scan time!");
        }
        // 正常情况
        else {
            scan_num ++;
            meas.lidar_end_time = meas.lidar_beg_time + duration;
            lidar_mean_scantime += (duration - lidar_mean_scantime) / scan_num;
            //neal::logger(neal::LOG_INFO, "lidar mean scan time: " + std::to_string(lidar_mean_scantime));
        }
        // 代表 LiDAR 数据已经被放到 meas 中了
        lidar_pushed = true;
    }

    // 如果 IMU 数据还没有收全
    if (last_timestamp_imu < meas.lidar_end_time) {
        return false;
    }

    /* 拿出 lidar_beg_time 到 lidar_end_time 之间的所有 IMU 数据*/
    meas.imu.clear();
    while ((!imu_buffer.empty())) {
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > meas.lidar_end_time) {
            break;
        }
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    lidar_pushed = false;  // 等待放入新的 LiDAR 数据
    std::string strout;
    strout = "receive a message. lidar begin time: " +  std::to_string(meas.lidar_beg_time) +
        "; lidar end time: " + std::to_string(meas.lidar_end_time) +
        "; imu begin time: " + std::to_string(meas.imu.front()->header.stamp.toSec()) + 
        "; imu end time: " + std::to_string(meas.imu.back()->header.stamp.toSec());
    neal::logger(neal::LOG_INFO, strout);
    return true;
}

void map_incremental(const bool flg_EKF_inited, const double filter_size_map_min, const state_ikfom &sp) {

    int feats_down_size = feats_down_body->points.size();

    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++) {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]), sp);
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited) {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++) {
                if (points_near.size() < NUM_MATCH_POINTS) {
                    break;
                }
                if (calc_dist(points_near[readd_i], mid_point) < dist) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) {
                PointToAdd.push_back(feats_down_world->points[i]);
            }
        }
        else {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }
    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
}

void publish_frame_world(const ros::Publisher& pubLaserCloudFull, const bool scan_pub_en, const bool dense_pub_en,
    const bool pcd_save_en, const PointCloudXYZI::Ptr& feats_undistort, const state_ikfom& sp,
    PointCloudXYZI::Ptr& pcl_wait_save, const double& let) {
    
    if (scan_pub_en) {
        // 判断是否发布稠密数据
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        // 获取待转换点云的大小
        int size = laserCloudFullRes->points.size();
        // 转换到世界坐标系的点云
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++) {
            pointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i], sp);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(let);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
    }

    /* save map:
    1. make sure you have enough memories;
    2. pcd save will largely influence the real-time performences.*/
    if (pcd_save_en) {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++) {
            pointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i], sp);
        }
        *pcl_wait_save += *laserCloudWorld;
    }
}

// 在 publish_odometry 和 publish_path 中调用
template <typename T>
void set_posestamp(T &out, const state_ikfom& sp, const geometry_msgs::Quaternion& gq) {

    out.pose.position.x = sp.pos(0);  // 将 esekf 求得的位置传入
    out.pose.position.y = sp.pos(1);
    out.pose.position.z = sp.pos(2);
    out.pose.orientation.x = gq.x;  // 将 esekf 求得的姿态传入
    out.pose.orientation.y = gq.y;
    out.pose.orientation.z = gq.z;
    out.pose.orientation.w = gq.w;
}

// 发布里程计
void publish_odometry(const ros::Publisher &pubOdomAftMapped, const state_ikfom& sp,
    const geometry_msgs::Quaternion& gq, const Eigen::Matrix<double, 23, 23>& P, const double& let) {

    nav_msgs::Odometry odomAftMapped;  // 只包含了一个位姿

    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(let);
    set_posestamp(odomAftMapped.pose, sp, gq);
    pubOdomAftMapped.publish(odomAftMapped);

    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        // 协方差 P 里先是旋转后是位置，这个 POSE 里先是位置后是旋转，所以对应的协方差要对调一下
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

// 每隔 10 个发布一下位姿
void publish_path(const ros::Publisher pubPath, const state_ikfom& sp,
    const geometry_msgs::Quaternion& gq, nav_msgs::Path& path, const double& let) {

    geometry_msgs::PoseStamped msg_body_pose;  // 位姿

    set_posestamp(msg_body_pose, sp, gq);
    msg_body_pose.header.stamp = ros::Time().fromSec(let);
    msg_body_pose.header.frame_id = "camera_init";

    /* if path is too large, the rvis will crash.*/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

// 对应 fast-lio2 公式 12 和 13。fast-lio 公式 14。
void h_share_model(state_ikfom &st, esekfom::dyn_share_datastruct<double> &ekfom_data) {

    int feats_down_size = feats_down_body->points.size();

    // feats_down_body 中的有效点  
    PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
    laserCloudOri->resize(feats_down_size);
    // laserCloudOri 对应的法相量
    PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());
    corr_normvect->resize(feats_down_size);
    // 是否为平面特征点
    std::vector<bool> point_selected_surf;
    point_selected_surf.resize(feats_down_size);
    // 特征点在地图中对应点的，局部平面参数，w 系
    PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
    normvec->resize(feats_down_size);

    /* 最近邻曲面搜索和残差计算*/
    for (int i = 0; i < feats_down_size; i++) {
        /* 将点云坐标转换至世界坐标系下*/
        const PointType &point_body = feats_down_body->points[i];  // 降采样后点云的 LiDAR 坐标
        PointType &point_world = feats_down_world->points[i];      // 降采样后点云的世界坐标
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(st.rot * (st.offset_R_L_I * p_body + st.offset_T_L_I) + st.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        /* 寻找最近邻点*/
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        PointVector &points_near = Nearest_Points[i];  // 点云的最近点序列
        // 在 ikd-Tree 上查找特征点的最近邻
        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

        // 如果最近邻的点数小于 NUM_MATCH_POINTS 或者最近邻的点到特征点的距离大于 5m，则认为该点不是有效点
        point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false :
            (pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5.0 ? false: true);
        if (!point_selected_surf[i]) {  // 如果不是有效点
            continue;
        }

        /* 拟合平面方程 ax+by+cz+d=0 并求解点到平面距离*/
        VF(4) pabcd;                     // 平面点信息
        point_selected_surf[i] = false;  // 先设为无效点
        // common_lib.h 函数，寻找法向量
        if (esti_plane(pabcd, points_near, 0.1f)) {
            // 计算点到平面的距离
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            // 如果 s>0.9，则认为找到平面
            if (s > 0.9) {
                point_selected_surf[i] = true;       // 再次设为有效点
                normvec->points[i].x = pabcd(0);     // 存储法向量
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;  // 存储点到平面的距离
            }
        }
    }

    /* 数据准备*/
    int effct_feat_num = 0;  // 有效特征点数
    for (int i = 0; i < feats_down_size; i++) {
        // 如果是有效点
        if (point_selected_surf[i]) {
            // 将点云的 LiDAR 坐标存到 laserCloudOri 中
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            // 将拟合平面法向量存到 corr_normvect 中
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num ++;  // 有效特征点数 ++
        }
    }

    if (effct_feat_num < 1) {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        neal::logger(neal::LOG_WARN, "No Effective Points!");
        return;
    }

    /* 求解观测雅可比矩阵 H 和观测向量 h*/
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);  // 观测雅可比矩阵 H
    ekfom_data.h.resize(effct_feat_num);                         // 观测向量 h
    for (int i = 0; i < effct_feat_num; i++) {
        // 拿到有效点云的 LiDAR 坐标
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;  // LiDAR 中，点云的 ^ 矩阵
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        // 转换到 IMU 坐标系下
        V3D point_this = st.offset_R_L_I * point_this_be + st.offset_T_L_I;
        M3D point_crossmat;  // IMU 中，点云的 ^ 矩阵
        point_crossmat << SKEW_SYM_MATRX(point_this);
        // 拿到拟合平面的法向量，Global 系下。
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        // 更新观测雅可比矩阵 H
        V3D C(st.rot.conjugate() * norm_vec);                        // (G^R_I)^T * norm_vec，IMU 系下的法向量
        V3D A(point_crossmat * C);                                   // IMU 系下，点叉乘法向量
        V3D B(point_be_crossmat * st.offset_R_L_I.conjugate() * C);  // LiDAR 系下，点叉乘法向量
        ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        // 更新观测向量 h = -z
        ekfom_data.h(i) = -norm_p.intensity;
    }
}




int main(int argc, char **argv) {

    neal::logger(neal::LOG_INFO, "Test start.");

    /* 变量申明与定义*/
    // Debug
    //int spin_num = 0;
    // 地图的最小分辨率
    double filter_size_map_min = 0.0;
    // I^T_L 和 I^R_L
    std::vector<double> extrinT(3, 0.0);
    std::vector<double> extrinR(9, 0.0);
    // IMU 的角速度协方差，加速度协方差，角速度偏置协方差，加速度偏置协方差
    double gyr_cov = 0.0, acc_cov = 0.0, b_gyr_cov = 0.0, b_acc_cov = 0.0;
    // KF 最大迭代次数
    const int NUM_MAX_ITERATIONS = 4;
    // LiDAR topic 和 IMU topic
    std::string lid_topic, imu_topic;

    // 发布当前正在扫描的点云数据，将点云地图保存到 PCD 文件
    const bool scan_pub_en = true, pcd_save_en = true;
    // 发布经过运动畸变校正注册到 IMU 坐标系的点云数据，需要两个变量同时为 true 才发布
    const bool dense_pub_en = true, scan_body_pub_en = true;
    // 立方体长度，当前雷达系中心到各个地图边缘的距离
    float cube_len = 0.0;
    float det_range = 0.0;
    // preprocess 参数
    double p_blind = 0.0;
    int p_N_SCANS = 0;

    // 初始化 ROS 节点，节点名为 laserMapping
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    // 从文件读取参数
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<std::string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<float>("cube_side_length", cube_len, 1000.0);
    nh.param<float>("mapping/det_range", det_range, 450.0);
    nh.param<double>("preprocess/blind", p_blind, 4.0);
    nh.param<int>("preprocess/scan_line", p_N_SCANS, 6);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);

    p_pre->set_blind(p_blind);
    p_pre->set_N_SCANS(p_N_SCANS);
    p_pre->set_point_filter_num(3);

    // 初始化输出路径
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";
    // VoxelGrid 用来执行降采样操作，PointType = pcl::PointXYZINormal
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    downSizeFilterSurf.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    // 设置 IMU 的参数，对 p_imu 进行初始化
    V3D Lidar_T_wrt_IMU(V3D(0.0,0.0,0.0));
    M3D Lidar_R_wrt_IMU(M3D::Identity());
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);           // I^T_L
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);           // I^R_L
    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());  // 定义指向 IMU 数据预处理类 ImuProcess 的指针
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    /* ikfom 第六步，初始化。*/
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;  // 状态，噪声维度，输入
    /* ikfom 第七步，发布 kf。*/
    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);  // 迭代收敛条件。
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /* ROS 订阅器和发布器的定义和初始化*/
    // 雷达点云的订阅器 sub_pcl，订阅点云的 topic
    ros::Subscriber sub_pcl = nh.subscribe(lid_topic, 200000, livox_pcl_cbk);  // 100ms 一帧
    // IMU 的订阅器 sub_imu，订阅 IMU 的 topic
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);  // 5ms 一帧
    // 发布当前正在扫描的点云，topic 名字为 cloud_registered
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    // 发布当前里程计信息，topic 名字为 Odometry
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    // 发布里程计总的路径，topic 名字为 path
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    
    // 中断处理函数，第一个参数 SIGINT 代表中断（interrupt）
    // 如果有中断信号（比如 Ctrl+C），则执行第二个参数里面的 SigHandle 函数
    signal(SIGINT, SigHandle);

    /* 主循环变量申明*/
    ros::Rate rate(5000);  // 设置主循环每次运行的时间至少为 0.0002 秒（5000Hz）
    bool status = ros::ok();
    // LiDAR 初次扫描时间
    bool flg_first_scan = true;
    double first_lidar_time = 0.0;
    // sync_packages 中使用的变量
    MeasureGroup measures;
    // publish_frame_world 中使用的变量
    PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());  // 等待保存的点云
    // debug parameters.
    int dbg_run_times = 0;
    while (status) {
        if (flg_exit) {  // 有中断产生
            break;
        }
        // 订阅器的回调函数处理一次
        // spin_num ++;
        // neal::logger(neal::LOG_INFO, "spin once. spin times: " + std::to_string(spin_num));
        ros::spinOnce();
        // 将第一帧 LiDAR 数据，和这段时间内的 IMU 数据从缓存队列中取出，并保存到 meas 中
        if (!sync_packages(measures)) {
            status = ros::ok();
            rate.sleep();
            continue;
        }
        // 第一次 while 循环，进行初始化
        if (flg_first_scan) {
            first_lidar_time = measures.lidar_beg_time;
            flg_first_scan = false;
            continue;
        }
        ROS_INFO("number of run times: %d \n", dbg_run_times++);
        // 对 IMU 数据进行预处理，包含了前向传播和反向传播
        PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());  // 去畸变的特征
        p_imu->Process(measures, kf, feats_undistort);
        // 如果点云数据为空，代表激光雷达没有完成去畸变，此时还不能初始化成功
        if (feats_undistort->empty() || (feats_undistort == NULL)) {
            ROS_WARN("No point, skip this scan!(1)\n");
            continue;
        }
        // 获取 kf 预测的全局状态
        state_ikfom state_point = kf.get_x();
        // 世界系下雷达坐标系的位置，W^p_L = W^p_I + W^R_I * I^t_L
        vect3 pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        // 动态调整局部地图
        lasermap_fov_segment(pos_lid, cube_len, det_range);

        // 对一次 scan 内的特征点云降采样
        downSizeFilterSurf.setInputCloud(feats_undistort);     // 输入去畸变后的点云数据
        downSizeFilterSurf.filter(*feats_down_body);           // 输出降采样后的点云数据
        int feats_down_size = feats_down_body->points.size();  // 降采样后的点云数量
        neal::logger(neal::LOG_INFO, "size before down sample: " + std::to_string(feats_undistort->points.size())
            + "; size after down sample: " + std::to_string(feats_down_size));
        if (feats_down_size <= 5) {
            ROS_WARN("No point, skip this scan!(2)\n");
            continue;
        }
        
        // 构建 ikd-Tree
        if (ikdtree.Root_Node == nullptr) {
            // 设置 ikd-Tree 的降采样参数
            ikdtree.set_downsample_param(filter_size_map_min);
            // 世界坐标系下，降采样的点云数据
            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++) {
                // 将降采样得到的点云数据，转换到世界坐标系下
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]), state_point);
            }
            // 构建 ikd-Tree
            ikdtree.Build(feats_down_world->points);
            ROS_INFO("ikd-Tree initialized!");

            continue;
        }

        /* 迭代卡尔曼滤波更新地图信息*/
        feats_down_world->resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        /* ikfom 第九步，更新*/
        kf.update_iterated_dyn_share_modified(_LASER_POINT_COV);

        /* 发布里程计*/
        state_point = kf.get_x();
        geometry_msgs::Quaternion geoQuat;  // 四元数
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];
        publish_odometry(pubOdomAftMapped, state_point, geoQuat, kf.get_P(), measures.lidar_end_time);

        /* 向 ikd-Tree 添加特征点*/
        bool flg_EKF_inited = (measures.lidar_beg_time - first_lidar_time) < _INIT_TIME ? false : true;
        map_incremental(flg_EKF_inited, filter_size_map_min, state_point);

        /* 发布轨迹和点*/
        publish_path(pubPath, state_point, geoQuat, path, measures.lidar_end_time);
        if (scan_pub_en || pcd_save_en) {
            publish_frame_world(pubLaserCloudFull, scan_pub_en, dense_pub_en, pcd_save_en,
                feats_undistort, state_point, pcl_wait_save, measures.lidar_end_time);
        }

        status = ros::ok();
        rate.sleep();
    }

    /* save map
    1. make sure you have enough memories;
    2. pcd save will largely influence the real-time performences*/
    if (pcl_wait_save->size() > 0 && pcd_save_en) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        std::cout << "current scan saved to /PCD/" << file_name << std::endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    return 0;
}