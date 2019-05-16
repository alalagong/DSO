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
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>




namespace dso
{
struct CalibHessian;
struct FrameHessian;


struct Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// index in jacobian. never changes (actually, there is no reason why).
	float u,v;

	// idepth / isgood / energy during optimization.
	float idepth;				//!< 该点对应参考帧的逆深度
	bool isGood;				//!< 怎么判断?
	Vec2f energy;				//!<   // (UenergyPhotometric, energyRegularizer)	
	bool isGood_new;
	float idepth_new;			//!< 该点在新的一帧(当前帧)上的逆深度
	Vec2f energy_new;			//!< 

	float iR;					//!< 
	float iRSumNum;

	float lastHessian;			//!< 
	float lastHessian_new;

	// max stepsize for idepth (corresponding to max. movement in pixel-space).
	float maxstep;				//!< 

	// idx (x+y*w) of closest point one pyramid level above.
	int parent;		  			//!< 上一层中该点的父节点 (距离最近的)的id
	float parentDist;			//!< 上一层中与父节点的距离

	// idx (x+y*w) of up to 10 nearest points in pixel space.
	int neighbours[10];			//!< 图像中离该点最近的10个点
	float neighboursDist[10];   //!< 最近10个点的距离

	float my_type; 				//!< 第0层提取是1, 2, 4, 对应d, 2d, 4d, 其它层是1
	float outlierTH; 			//!< 外点阈值
};

class CoarseInitializer {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	CoarseInitializer(int w, int h);
	~CoarseInitializer();


	void setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian);
	bool trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps);
	void calcTGrads(FrameHessian* newFrameHessian);

	int frameID;					//!< 
	bool fixAffine;
	bool printDebug;

	Pnt* points[PYR_LEVELS]; 		//!< 每一层上的点类, 是第一帧提取出来的
	int numPoints[PYR_LEVELS];  	//!< 每一层的点数目
	AffLight thisToNext_aff;		//!< 参考帧与当前帧之间光度系数
	SE3 thisToNext;					//!< 


	FrameHessian* firstFrame;		//!< 第一帧
	FrameHessian* newFrame;			//!< track中新加入的帧
private:
	Mat33 K[PYR_LEVELS];			//!< camera参数
	Mat33 Ki[PYR_LEVELS];
	double fx[PYR_LEVELS];
	double fy[PYR_LEVELS];
	double fxi[PYR_LEVELS];
	double fyi[PYR_LEVELS];
	double cx[PYR_LEVELS];
	double cy[PYR_LEVELS];
	double cxi[PYR_LEVELS];
	double cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];
	void makeK(CalibHessian* HCalib);

	bool snapped;					//!< 
	int snappedAt;

	// pyramid images & levels on all levels
	Eigen::Vector3f* dINew[PYR_LEVELS];
	Eigen::Vector3f* dIFist[PYR_LEVELS];

	Eigen::DiagonalMatrix<float, 8> wM;

	// temporary buffers for H and b.
	Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
	Vec10f* JbBuffer_new;		//!< 

	//* 9维向量, 乘积获得9*9矩阵, 并做的累加器
	Accumulator9 acc9;				
	Accumulator9 acc9SC;


	Vec3f dGrads[PYR_LEVELS];		//!< 

	float alphaK;					//!< 
	float alphaW;					//!< 
	float regWeight;				//!< 对逆深度的加权值
	float couplingWeight;			//!< 

	Vec3f calcResAndGS(
			int lvl,
			Mat88f &H_out, Vec8f &b_out,
			Mat88f &H_out_sc, Vec8f &b_out_sc,
			const SE3 &refToNew, AffLight refToNew_aff,
			bool plot);
	Vec3f calcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
	void optReg(int lvl);

	void propagateUp(int srcLvl);
	void propagateDown(int srcLvl);
	float rescale();

	void resetPoints(int lvl);
	void doStep(int lvl, float lambda, Vec8f inc);
	void applyStep(int lvl);

	void makeGradients(Eigen::Vector3f** data);

    void debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps);
	void makeNN();
};



//* 作为 KDTreeSingleIndexAdaptor 类的第二个模板参数必须给出, 包括下面的接口
struct FLANNPointcloud
{
    inline FLANNPointcloud() {num=0; points=0;}
    inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
	int num;
	Pnt* points;
	// 返回数据点的数目
	inline size_t kdtree_get_point_count() const { return num; }
	// 使用L2度量时使用, 返回向量p1, 到第idx_p2个数据点的欧氏距离
	inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
	{
		const float d0=p1[0]-points[idx_p2].u;
		const float d1=p1[1]-points[idx_p2].v;
		return d0*d0+d1*d1;
	}
	// 返回第idx个点的第dim维数据
	inline float kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0) return points[idx].u;
		else return points[idx].v;
	}
	// 可选计算bounding box
	// false 表示默认
	// true 本函数应该返回bb
	template <class BBOX>
		bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

}


