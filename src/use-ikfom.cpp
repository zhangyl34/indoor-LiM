#include "use-ikfom.h"

Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in) {

    // 24 对应：G^p_I，G^R_I，I^R_L，I^p_L，G^v_I，b_w，b_a，G^g
    Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
    
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    vect3 a_inertial = s.rot * (in.acc - s.ba);
    for (int i = 0; i < 3; i++) {
        res(i) = s.vel[i];                        // G^v_I
        res(i + 3) = omega[i];                    // omega_m - b_omega
        res(i + 12) = a_inertial[i] + s.grav[i];  // G^R_I * (a_m - b_a) + G^g
    }
    return res;
}

Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in) {

    // 23 = pos, rot, offset_R_L_I, offset_T_L_I, vel, bg, ba, grav(2).
    Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();

    vect3 acc_;
    in.acc.boxminus(acc_, s.ba);
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
    Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
    s.S2_Mx(grav_matrix, vec, 21);

    cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();                 // I
    cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc_);  // -G^R_I * [a_m - b_a]^
    cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();                  // -G^R_I
    cov.template block<3, 2>(12, 21) = grav_matrix;                                // -[G^g]^ * B(G^g)
    cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();                // -I
    return cov;
}

Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in) {

    Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
    cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();    // -G^R_I
    cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();  // -I
    cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();  // I
    cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();  // I
    return cov;
}

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}




