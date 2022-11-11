/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"

//#define USE_sparse


namespace esekfom {

using namespace Eigen;

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as a manifold.
template<typename S, typename M, int measurement_noise_dof = M::DOF>
struct share_datastruct
{
	bool valid;
	bool converge;
	M z;
	Eigen::Matrix<typename S::scalar, M::DOF, measurement_noise_dof> h_v;
	Eigen::Matrix<typename S::scalar, M::DOF, S::DOF> h_x;
	Eigen::Matrix<typename S::scalar, measurement_noise_dof, measurement_noise_dof> R;
};

//used for iterated error state EKF update
//for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
//applied for measurement as an Eigen matrix whose dimension is changing
template<typename T>
struct dyn_share_datastruct
{
	bool valid;
	bool converge;
	Eigen::Matrix<T, Eigen::Dynamic, 1> z;
	Eigen::Matrix<T, Eigen::Dynamic, 1> h;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
};


template<typename state, int process_noise_dof, typename input = state, typename measurement=state, int measurement_noise_dof=0>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF, m = state::DIM, l = measurement::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;
	typedef measurement measurementModel(state &, bool &);
	typedef measurement measurementModel_share(state &, share_datastruct<state, measurement, measurement_noise_dof> &);
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> measurementModel_dyn(state &, bool &);
	//typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> measurementModel_dyn_share(state &,  dyn_share_datastruct<scalar_type> &);
	typedef void measurementModel_dyn_share(state &,  dyn_share_datastruct<scalar_type> &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &, bool&);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){
	#ifdef USE_sparse
		SparseMatrix<scalar_type> ref(n, n);
		ref.setIdentity();
		l_ = ref;
		f_x_2 = ref;
		f_x_1 = ref;
	#endif
	};

	//receive system-specific models and their differentions
	//for measurement as an Eigen matrix whose dimension is changing.
	//calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function (h_dyn_share_in).
	void init_dyn_share(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel_dyn_share h_dyn_share_in, int maximum_iteration, scalar_type limit_vector[n])
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;
		h_dyn_share = h_dyn_share_in;

		maximum_iter = maximum_iteration;
		for(int i=0; i<n; i++)
		{
			limit[i] = limit_vector[i];
		}

		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
	}

    // 正向传播
	void predict(double &dt, processnoisecovariance &Q, const input &i_in) {
		flatted_state f_ = f(x_, i_in);  // 24x1
		cov_ f_x_ = f_x(x_, i_in);       // 24x23
		cov f_x_final;                   // 23x23，管理 G_f
		F_x1 = cov::Identity();          // 23x23，管理 G_x

		Matrix<scalar_type, m, process_noise_dof> f_w_ = f_w(x_, i_in);  // 24x12
		Matrix<scalar_type, n, process_noise_dof> f_w_final;             // 23x12
		state x_before = x_;
		x_.oplus(f_, dt);

        // 用 f_x_ 和 f_w_ 给 f_x_final 和 f_w_final 赋值，先赋 vect 部分。
		for (std::vector<std::pair<std::pair<int, int>, int>>::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;  // 都是 3。
			for(int i = 0; i < n; i++) {  // 23
				for(int j = 0; j < dof; j++) {
                    f_x_final(idx+j, i) = f_x_(dim+j, i);
                }
			}
			for(int i = 0; i < process_noise_dof; i++) {  // 12
				for(int j = 0; j < dof; j++) {
                    f_w_final(idx+j, i) = f_w_(dim+j, i);
                }
			}
		}

        // 再赋 SO3 部分。
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_(dim + i) * dt;
			}
			MTK::SO3<scalar_type> res;
			// w = cos(theta/2), vec = sin(theta/2)[x,y,z]
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
		#ifdef USE_sparse
			// res_temp_SO3 = res.toRotationMatrix();
			// for(int i = 0; i < 3; i++){
			// 	for(int j = 0; j < 3; j++){
			// 		f_x_1.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
			// 	}
			// }
		#else
			// res = (cos(theta/2),sin(theta/2)[x,y,z]).
			// toRotationMatrix 绕 [x,y,z] 轴旋转 theta 角。
			F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();  // G_x
		#endif			
			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < n; i++) {  // 23
			    // G_f * f_x_
				f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < process_noise_dof; i++) {  // 12
			    // G_f * f_w_
				f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
			}
		}
		
		// 最后赋 S2 部分。
		Matrix<scalar_type, 2, 3> res_temp_S2;
		Matrix<scalar_type, 2, 2> res_temp_S2_;
		MTK::vect<3, scalar_type> seg_S2;
		for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++) {
				seg_S2(i) = f_(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			// w = cos(theta/2), vec = sin(theta/2)[x,y,z]
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);          // Nx = 1 / r / r * B^T * (x_)^
			x_before.S2_Mx(Mx, vec, idx);  // Mx = -Exp(B*vec) * (x_before)^ * A(B*vec)^T * B
		#ifdef USE_sparse
			// res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;
			// for(int i = 0; i < 2; i++){
			// 	for(int j = 0; j < 2; j++){
			// 		f_x_1.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
			// 	}
			// }
		#else
			F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;  // G_x
		#endif
			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat * MTK::A_matrix(seg_S2).transpose();
			for(int i = 0; i < n; i++) {
				// G_f * f_x_
				f_x_final.template block<2, 1>(idx, i) = res_temp_S2 * (f_x_.template block<3, 1>(dim, i));
			}
			for(int i = 0; i < process_noise_dof; i++) {
				// G_f * f_w_
				f_w_final.template block<2, 1>(idx, i) = res_temp_S2 * (f_w_.template block<3, 1>(dim, i));
			}
		}
	
	#ifdef USE_sparse
		// f_x_1.makeCompressed();
		// spMt f_x2 = f_x_final.sparseView();
		// spMt f_w1 = f_w_final.sparseView();
		// spMt xp = f_x_1 + f_x2 * dt;
		// P_ = xp * P_ * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	#else
		F_x1 += f_x_final * dt;  // G_x + G_f * dt * f_x_
		P_ = (F_x1) * P_ * (F_x1).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();  // 23x23
	#endif
	}
	
	
	/* iterated esekf.
	点云的协方差，函数耗时。*/
	void update_iterated_dyn_share_modified(const double& R) {
		  
		dyn_share_datastruct<scalar_type> dyn_share;
		dyn_share.valid = true;
		dyn_share.converge = true;
		// 获取最后一次的状态和协方差矩阵
		state x_propagated = x_;        // hat(x_k)
		cov P_propagated = P_;          // hat(P_k)
		Matrix<scalar_type, n, 1> K_h;  // 23x1
		Matrix<scalar_type, n, n> K_x;  // 23x23
		int dof_Measurement;
		
		vectorized_state dx_new = vectorized_state::Zero();  // 23x1
		// 最多进行 maximum_iter 次迭代优化
		for(int i = 0; i < maximum_iter; i ++) {
			dyn_share.valid = true;
			// 计算观测模型的 h 和 h_x
			h_dyn_share(x_, dyn_share);

			if(!dyn_share.valid) {  // 观测点数量不足
				continue; 
			}

			// 获取观测模型的 h_x
			Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ = dyn_share.h_x;

			dof_Measurement = h_x_.rows();  // 观测方程个数
			vectorized_state dx;            // 误差状态 23x1
			x_.boxminus(dx, x_propagated);  // dx = hat(x_k^k) - hat(x_k)
			dx_new = dx;                    // 
			P_ = P_propagated;  // 预测得到的误差状态协方差矩阵，23x23
			
			/* P_ = J_2^k * hat(P_k) * (J_2^k)^T.
			dx_new = J_2^k * dx.*/
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin();
				it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				// int dim = (*it).second;
				for(int j = 0; j < 3; j++) {
					seg_SO3(j) = dx(idx+j);
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();  // J_2^k = A^T
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int j = 0; j < n; j++) {  // n = 23，左乘 J_2^k
					P_. template block<3, 1>(idx, j) = res_temp_SO3 * (P_. template block<3, 1>(idx, j));	
				}
				for(int j = 0; j < n; j++) {  // 右乘 (J_2^k)^T
					P_. template block<1, 3>(j, idx) =(P_. template block<1, 3>(j, idx)) * 
						res_temp_SO3.transpose();
				}
			}
			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin();
				it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				for(int j = 0; j < 2; j++) {
					seg_S2(j) = dx(idx + j);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);                 // Nx = 1 / r / r * B^T * (x_)^
				x_propagated.S2_Mx(Mx, seg_S2, idx);  // Mx = -Exp(B*dx) * (x_propagated)^ * A(B*dx)^T * B
				res_temp_S2 = Nx * Mx;                // J_2^k = Nx * Mx
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int j = 0; j < n; j++) {  // 左乘 J_2^k
					P_. template block<2, 1>(idx, j) = res_temp_S2 * (P_. template block<2, 1>(idx, j));
				}
				for(int j = 0; j < n; j++) {  // 右乘 (J_2^k)^T
					P_. template block<1, 2>(j, idx) = (P_. template block<1, 2>(j, idx)) * 
						res_temp_S2.transpose();
				}
			}

            /* 计算 K_x = K * H; K_h = K * z。*/
			if(n > dof_Measurement) {  // n = 23，如果状态维度大于观测方程，不满秩
				/* 下列计算等价于：K = P * H^T * (H * P * H^T + R)^-1。*/
				Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = 
					Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
				h_x_cur.topLeftCorner(dof_Measurement, 12) = h_x_;  // h_x_cur = H
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose()/R + 
					Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse()/R;
				K_h = K_ * dyn_share.h;  // K * -z
				K_x = K_ * h_x_cur;      // K * H
			}
			else {
				/* 下列计算等价于：K = (H^T * R^-1 * H + P^-1)^-1 * H^T * R^-1。*/
				cov P_temp = (P_/R).inverse();
				Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_.transpose() * h_x_;
				P_temp. template block<12, 12>(0, 0) += HTH;
				cov P_inv = P_temp.inverse();
				K_h = P_inv. template block<n, 12>(0, 0) * h_x_.transpose() * dyn_share.h;    // K_h = K * -z
				K_x.setZero();
				K_x. template block<n, 12>(0, 0) = P_inv. template block<n, 12>(0, 0) * HTH;  // K_x = K * H
			}

			// 误差状态增量 dx_ = -Kz + (KH - I) * J_2^k * dx
			Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
			// 更新 hat(x_k^k)
			x_.boxplus(dx_);
			// 判断迭代是否收敛
			dyn_share.converge = true;
			for(int j = 0; j < n ; j++) {
				if(std::fabs(dx_[j]) > limit[j]) {
					dyn_share.converge = false;
					break;
				}
			}

            /* 迭代完成后，更新协方差矩阵：bar(P_k) = (I - KH)P。*/
			if(dyn_share.converge || i == (maximum_iter - 1)) {
				L_ = P_;  // P
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int j = 0; j < 3; j++) {
						seg_SO3(j) = dx_(j + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int j = 0; j < n; j++) {
						L_. template block<3, 1>(idx, j) = res_temp_SO3 * (P_. template block<3, 1>(idx, j)); 
					}
					for(int j = 0; j < 12; j++) {
						K_x. template block<3, 1>(idx, j) = res_temp_SO3 * (K_x. template block<3, 1>(idx, j));
					}
					for(int j = 0; j < n; j++) {
						L_. template block<1, 3>(j, idx) = (L_. template block<1, 3>(j, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(j, idx) = (P_. template block<1, 3>(j, idx)) * res_temp_SO3.transpose();
					}
				}
				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
					for(int j = 0; j < 2; j++) {
						seg_S2(j) = dx_(j + idx);
					}
					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
					for(int j = 0; j < n; j++) {
						L_. template block<2, 1>(idx, j) = res_temp_S2 * (P_. template block<2, 1>(idx, j)); 
					}
					for(int j = 0; j < 12; j++) {
						K_x. template block<2, 1>(idx, j) = res_temp_S2 * (K_x. template block<2, 1>(idx, j));
					}
					for(int j = 0; j < n; j++) {
						L_. template block<1, 2>(j, idx) = (L_. template block<1, 2>(j, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(j, idx) = (P_. template block<1, 2>(j, idx)) * res_temp_S2.transpose();
					}
				}
				P_ = L_ - K_x.template block<n, 12>(0, 0) * P_.template block<12, n>(0, 0);
				//P_ = L_ - K_x * L_;  // 更有可能会中途崩溃。
				return;
			}
		}
	}

	void change_x(state &input_state)
	{
		x_ = input_state;
		if((!x_.vect_state.size())&&(!x_.SO3_state.size())&&(!x_.S2_state.size()))
		{
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
		}
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
private:
	state x_;
	measurement m_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	measurementModel *h;
	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	measurementModel_dyn *h_dyn;
	measurementMatrix1_dyn *h_x_dyn;
	measurementMatrix2_dyn *h_v_dyn;

	measurementModel_share *h_share;
	measurementModel_dyn_share *h_dyn_share;

	int maximum_iter = 0;
	scalar_type limit[n];
	
	template <typename T>
    T check_safe_update( T _temp_vec )
    {
        T temp_vec = _temp_vec;
        if ( std::isnan( temp_vec(0, 0) ) )
        {
            temp_vec.setZero();
            return temp_vec;
        }
        double angular_dis = temp_vec.block( 0, 0, 3, 1 ).norm() * 57.3;
        double pos_dis = temp_vec.block( 3, 0, 3, 1 ).norm();
        if ( angular_dis >= 20 || pos_dis > 1 )
        {
            printf( "Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis );
            temp_vec.setZero();
        }
        return temp_vec;
    }
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
