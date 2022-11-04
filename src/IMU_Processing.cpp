#include "IMU_Processing.h"


ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true) {

    init_iter_num = 1;
    Q = process_noise_cov();
    cov_acc         = V3D(0.1, 0.1, 0.1);
    cov_gyr         = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr    = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc    = V3D(0.0001, 0.0001, 0.0001);
    mean_acc        = V3D(0.0, 0.0, -1.0);
    mean_gyr        = V3D(0.0, 0.0, 0.0);
    angvel_last     = V3D(0.0 ,0.0 ,0.0);
    Lidar_T_wrt_IMU = V3D(0.0 ,0.0 ,0.0);
    Lidar_R_wrt_IMU = M3D::Identity();
    last_imu_.reset(new sensor_msgs::Imu());
}

void ImuProcess::Reset() {
    mean_acc = V3D(0.0, 0.0, -1.0);
    mean_gyr = V3D(0.0, 0.0, 0.0);
    angvel_last = V3D(0.0 ,0.0 ,0.0);
    imu_need_init_ = true;                    // 是否需要初始化 IMU
    init_iter_num = 1;                        // 初始化迭代次数
    IMUpose.clear();                          // IMU 位姿清空
    last_imu_.reset(new sensor_msgs::Imu());  // 上一帧 IMU 初始化
}


void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state) {

    V3D cur_acc, cur_gyr;

    // 如果是第一帧
    if (b_first_frame_) {
        Reset();  // 重置参数
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;  // IMU 初始时刻的加速度
        const auto &gyr_acc = meas.imu.front()->angular_velocity;     // IMU 初始时刻的角速度
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;                  // 作为加速度均值
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;                  // 作为角速度均值
    }

    // 计算方差
    for (const auto &imu : meas.imu) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        // 均值更新
        mean_acc += (cur_acc - mean_acc) / init_iter_num;
        mean_gyr += (cur_gyr - mean_gyr) / init_iter_num;
        // 方差更新。https://blog.csdn.net/weixin_44479136/article/details/90510374 
        // cwiseProduct() 对应系数相乘
        // cov_acc = cov_acc * (init_iter_num - 1.0) / init_iter_num + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) / (init_iter_num - 1.0);
        // cov_gyr = cov_gyr * (init_iter_num - 1.0) / init_iter_num + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) / (init_iter_num - 1.0);

        init_iter_num ++;
    }
    
    state_ikfom init_state = kf_state.get_x();  // 在 esekfom.hpp 获得 x_ 的状态
    // 从 common_lib.h 中拿到重力值（9.81），以加速度均值作为重力方向
    init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);
    init_state.bg = mean_gyr;                   // 角速度均值作为陀螺仪偏差
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;  // 将 lidar 和 imu 外参位移量传入
    init_state.offset_R_L_I = Lidar_R_wrt_IMU;  // 将 lidar 和 imu 外参旋转量传入
    kf_state.change_x(init_state);              // 更新 x_ 的状态

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();  // 在 esekfom.hpp 获得协方差矩阵 P_
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);    // 更新协方差矩阵 P_
    last_imu_ = meas.imu.back();  // 将最后一帧的 imu 数据传入 last_imu_ 中
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, 
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out) {

    std::deque<sensor_msgs::Imu::ConstPtr> v_imu = meas.imu;           // 拿到当前的 IMU 数据
    v_imu.push_front(last_imu_);                                       // 将上一帧最后尾部的 IMU 添加到当前帧头部
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();  // 拿到当前帧头部的 IMU 的时间
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();   // 拿到当前帧尾部的 IMU 的时间
    const double &pcl_beg_time = meas.lidar_beg_time;                  // pcl 开始的时间戳

    // 把点云数据赋值给 pcl_out
    pcl_out = *(meas.lidar);
    // 根据点云中每个点的时间戳对点云进行重排序
    auto time_list = [](PointType &x, PointType &y) {return (x.curvature < y.curvature);};
    std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
    // pcl 结束时间戳 = pcl 开始时间戳 + 最后一帧的 offset_time
    const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
    // 获取上一次 KF 估计的后验状态作为本次 IMU 预测的初始状态
    state_ikfom imu_state = kf_state.get_x();
    // pose6d 包含：LiDAR offset_time，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
    // 平均角速度，平均加速度，IMU 加速度，IMU 速度，IMU 位置
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;       // IMU 旋转矩阵
    double dt = 0;   // 时间间隔
    input_ikfom in;  // 系统输入

    /* 前向传播：遍历本次估计的所有 IMU 测量并且进行积分*/
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);      // 拿到当前帧的 IMU 数据
        auto &&tail = *(it_imu + 1);  // 拿到下一帧的 IMU 数据
        // 判断时间先后顺序，不符合直接 continue
        if (tail->header.stamp.toSec() < last_lidar_end_time_) {
            continue;
        }

        // 离散中值积分
        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        // 通过重力数值对加速度进行一下微调
        acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

        // 如果 IMU 开始时刻早于上次雷达最晚时刻
        if (head->header.stamp.toSec() < last_lidar_end_time_) {
            // 从上次雷达最晚时刻开始传播
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        }
        else {
            // 从 IMU 开始时刻开始传播
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        // 原始测量的中值作为系统输入
        in.acc = acc_avr;
        in.gyro = angvel_avr;
        // 配置噪声 w 的协方差矩阵
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

        /* ikfom 第八步，IMU 前向传播。*/
        kf_state.predict(dt, Q, in);

        // 保存 IMU 预测过程的状态
        imu_state = kf_state.get_x();
        angvel_last = angvel_avr - imu_state.bg;                // 计算出来的角速度均值，IMU 坐标系下
        acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);  // 计算出来的加速度均值，世界坐标系下
        for (int i = 0; i < 3; i++) {
            acc_s_last[i] += imu_state.grav[i];                 // 加上重力得到真正的加速度，世界坐标系下
        }
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;  // 后一个 IMU 时刻距离此次雷达开始的时间间隔
        // 保存 IMU 预测过程的状态
        IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
    }

    // 把最后一帧 IMU 测量也补上
    // 判断雷达结束时间是否晚于 IMU，最后一个 IMU 时刻可能早于雷达末尾，也可能晚于雷达末尾
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);  // dt 不应该恒正？
    kf_state.predict(dt, Q, in);
    imu_state = kf_state.get_x();         // 点云结束时刻的状态向量
    last_imu_ = meas.imu.back();          // 保存最后一帧 IMU 数据
    last_lidar_end_time_ = pcl_end_time;  // 保存雷达测量的结束时间

    /* 反向传播，去畸变*/
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);       // 拿到前一帧的 IMU 旋转矩阵
        vel_imu << VEC_FROM_ARRAY(head->vel);     // 拿到前一帧的 IMU 速度
        pos_imu << VEC_FROM_ARRAY(head->pos);     // 拿到前一帧的 IMU 位置
        acc_imu << VEC_FROM_ARRAY(tail->acc);     // 拿到后一帧的 IMU 加速度，应该拿前一帧？
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);  // 拿到后一帧的 IMU 角速度，应该拿前一帧？

        // 点云时刻迟于前一帧 IMU 时刻，早于后一帧 IMU 时刻
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time;                   // 点云时刻到前一帧 IMU 时刻的时间间隔
            M3D R_i(R_imu * Exp(angvel_avr, dt));                                        // 点云时刻的 IMU 姿态，即 W^R_I
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);                                    // 点云时刻的点云位置，即 L^P
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);  // 点云时刻的 IMU 位置 - 点云结束时刻的 IMU 位置
            // conjugate() 取旋转矩阵的转置，（可能作者重新写了这个函数 eigen 官方库里这个函数好像没有转置这个操作，实际 cout 矩阵确实输出了转置）
            // imu_state.offset_R_L_I 是惯性系下雷达坐标系的姿态，简单记为 I^R_L
            // 这里倒推一下去畸变补偿的公式
            // e 代表 end 时刻
            // P_compensate 是点在末尾时刻（即补偿后）在雷达系的坐标，简记为 L^P_e
            // 将右侧矩阵乘过来并加上右侧平移
            // 左边变为 I^R_L * L^P_e + I^t_L = I^P_e，也就是 end 时刻点在 IMU 系下的坐标
            // 右边剩下 imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei)
            // imu_state.rot.conjugate() 是结束时刻 IMU 到世界坐标系的旋转矩阵的转置，也就是 (W^R_i_e)^T
            // T_ei 是：点所在时刻 IMU 在世界坐标系下的位置 - end 时刻 IMU 在世界坐标系下的位置，也就是 (W^t_I-W^t_I_e)
            // 现在等式两边变为 I^P_e =  (W^R_i_e)^T * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + W^t_I - W^t_I_e)
            // (W^R_i_e) * I^P_e + W^t_I_e = R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + W^t_I
            // 世界坐标系无所谓时刻，因为只有一个世界坐标系
            // W^P = R_i * I^P + W^t_I
            // W^P = W^P
            V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

            // 保存去畸变结果
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin())
                break;
        }
    }
}

void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_) {

    if (meas.imu.empty()) {  // 当前帧的 IMU 测量为空，则直接返回
        return;
    }
    ROS_ASSERT(meas.lidar != nullptr);

    // 前 MAX_INI_COUNT 帧 LiDAR 数据，用于初始化
    if (imu_need_init_) {
        // 前 MAX_INI_COUNT 帧 LiDAR 数据，用于计算加速度/角速度的均值
        IMU_init(meas, kf_state);
        last_imu_ = meas.imu.back();
        // 需要更新 MAX_INI_COUNT 次
        if (init_iter_num > IMU_MAX_INI_COUNT) {
            imu_need_init_ = false;
            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
            ROS_INFO("IMU Initial Done");
            fout_imu.open(DEBUG_FILE_DIR("imu.txt"), std::ios::out);
        }
        return;
    }

    // 正向传播，反向传播
    UndistortPcl(meas, kf_state, *cur_pcl_un_);
}



