/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;






class EFResidual
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
		data(org), point(point_), host(host_), target(target_)
	{
		isLinearized=false;
		isActiveAndIsGoodNEW=false;
		J = new RawResidualJacobian();
		assert(((long)this)%16==0);
		assert(((long)J)%16==0);
	}
	inline ~EFResidual()
	{
		delete J;
	}


	void takeDataF();


	void fixLinearizationF(EnergyFunctional* ef);


	// structural pointers
	PointFrameResidual* data;
	int hostIDX, targetIDX;  		//!< 残差对应的 host 和 Target ID号
	EFPoint* point;					//!< 残差点
	EFFrame* host;					//!< 主
	EFFrame* target;				//!< 目标
	int idxInAll;					//!< 所有残差中的id

	RawResidualJacobian* J;			//!< 用来计算jacob, res值

	VecNRf res_toZeroF;				//!< 更新delta后的线性残差
	Vec8f JpJdF;					//!< 点对逆深度求导


	// status.
	bool isLinearized;				//!< 计算完成res_toZeroF

	// if residual is not OOB & not OUTLIER & should be used during accumulations
	bool isActiveAndIsGoodNEW;
	inline const bool &isActive() const {return isActiveAndIsGoodNEW;}
};


enum EFPointStatus {PS_GOOD=0, PS_MARGINALIZE, PS_DROP};

class EFPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFPoint(PointHessian* d, EFFrame* host_) : data(d),host(host_)
	{
		takeData();
		stateFlag=EFPointStatus::PS_GOOD;
	}
	void takeData();

	PointHessian* data;	//!< PointHessian数据



	float priorF;		//!< 逆深度先验信息
	float deltaF;		//!< 当前逆深度和线性化处的差


	// constant info (never changes in-between).
	int idxInPoints;	//!< 当前点在EFFrame中id
	EFFrame* host;

	// contains all residuals.
	std::vector<EFResidual*> residualsAll;	//!< 该点的所有残差

	float bdSumF;
	float HdiF;				//!< 
	float Hdd_accLF;		//!< 逆深度的hessian
	VecCf Hcd_accLF;		//!< 逆深度和内参的hessian
	float bd_accLF;			//!< J逆深度*残差
	float Hdd_accAF;
	VecCf Hcd_accAF;
	float bd_accAF;


	EFPointStatus stateFlag;
};



class EFFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame(FrameHessian* d) : data(d)
	{
		takeData();
	}
	void takeData();

	//! 位姿 0-5, 光度ab 6-7
	//TODO 和fh的state有啥区别??
	Vec8 prior;					//!<  prior hessian (diagonal)
	Vec8 delta_prior;			//!<    // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
	Vec8 delta;					//!<   // state - state_zero.



	std::vector<EFPoint*> points;	//!< 帧上所有点
	FrameHessian* data;				//!< 对应FrameHessian数据
	//? 和FrameHessian中的idx有啥不同
	int idx;			//!< 在能量函数中帧id // idx in frames.

	int frameID;  		//!< 所有历史帧ID 
};

}

