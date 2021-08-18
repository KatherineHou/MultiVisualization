#pragma once


#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include  "glew.h"
#include <math.h>
#include "glut.h"
#include<iostream>
#include <math_constants.h>
#include <cuda_gl_interop.h>
#include<curand_kernel.h>
#include <omp.h>
#include <time.h>
#include<windows.h>
#include <tchar.h>
#include <random>


#define TOLERANCE 0.0001
#define MAXRECURSE 20
#define SIGMA 0.3
#define PI 3.1416
#define LAMBDAMAX 7

#pragma pack(16) 

struct PointProperty
{
	PointProperty() {}; 
	float Value;
	float x;
	float y;
	float z;
	float Nx;
	float Ny;
	float Nz;
	float Kmax;
	float Kmin;
	float KK;
	float KH;

};

struct Grid_Param
{
	Grid_Param(int w, int h, int d, int s, int i) :width(w), height(h), depth(d), cubeSize(s), isoValue(i){};
	Grid_Param() {};
	int width;
	int height;
	int depth;
	int cubeSize;
	float isoValue;

};

struct CellSample
{
	//this struct records how many sample points in each cell, and the index of the first point in this cell
	//It can be used in samples redistribution, to search the neighbor points within sigma around one point
	CellSample(int s, int n) : Serial(s), Number(n) {};
	int Serial;
	int Number;
};

struct EnergyProperty
{
	EnergyProperty(float e, float l, int f) :Energy(e), Lambda(l), flag(f) {};
	float Energy;
	float Lambda;
	int flag;
};





#pragma pack()
