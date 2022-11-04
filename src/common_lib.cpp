#include "common_lib.h"

Pose6D set_pose6d(const double t, const Eigen::Matrix<double, 3, 1> &a, const Eigen::Matrix<double, 3, 1> &g,
    const Eigen::Matrix<double, 3, 1> &v, const Eigen::Matrix<double, 3, 1> &p, const Eigen::Matrix<double, 3, 3> &R) {

    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++) {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return std::move(rot_kp);
}

Eigen::Matrix<double, 3, 3> Exp(const Eigen::Matrix<double, 3, 1> &ang_vel, const double &dt) {

    double ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<double, 3, 3> Eye3 = Eigen::Matrix<double, 3, 3>::Identity();

    if (ang_vel_norm > 0.0000001) {
        Eigen::Matrix<double, 3, 1> r_axis = ang_vel / ang_vel_norm;
        Eigen::Matrix<double, 3, 3> K;

        K << SKEW_SYM_MATRX(r_axis);

        double r_ang = ang_vel_norm * dt;

        /// Roderigous Tranformation
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

bool esti_plane(Eigen::Matrix<float, 4, 1> &pca_result, const PointVector &point, const float &threshold) {

    Eigen::Matrix<float, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<float, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Eigen::Matrix<float, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    float n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + 
            pca_result(2) * point[j].z + pca_result(3)) > threshold) {
            return false;
        }
    }
    return true;
}

float calc_dist(PointType p1, PointType p2) {

    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}


