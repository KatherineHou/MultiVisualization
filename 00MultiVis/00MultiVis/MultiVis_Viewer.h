#pragma once

#include <glew.h>
#include<glut.h>
#include<math.h>
#include"math3d.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include"GLframe.h"
#include<assert.h>
#include"type_define.h"
#include<iostream>
#include<fstream>
#include<sstream>
//#include<tchar.h>
//#include <random>

using namespace std;

#define MAX_SHADER_LENGTH 81920
#define GLT_ATTRIBUTE_VERTEX    0
#define GLT_ATTRIBUTE_TEXCOORD1 1

#define CUBESIZE 8
#define DOMAINSIZE 4
#define GRIDWIDTH 32
#define GRIDHIGHT 32
#define GRIDDEPTH 32

#define SPHERESEGMENTS 50
#define TORUSRADIUS 2.0

#define NU 10
#define NN 8
//#define TOLERANCE 0.0001
//#define MAXRECURSE 20

class MULTIVIS_RENDER
{
public:
	MULTIVIS_RENDER(int w, int h, int d);
	~MULTIVIS_RENDER();
	//void SampleDisplay();
	void MultiVisRendering();
	void cleanup();

	int ObjectSelection=1; // "0" for sphere, "1" for torus, "2" for plane, "3" for cuda surface attraction 
	int Parameters = 0;
	float IsoValue = 1;
	//float MajorRadius = 2;
	//float MinorRadius = 1;
	int SampleNumber = 1000;
	
	inline void setTransformMatrix(float Tx, float Ty, float Tz, float Rx, float Ry)
	{
		M3DMatrix44f translationMatrix, transM1;
		m3dLoadIdentity44(translationMatrix);
		m3dLoadIdentity44(transM1);
		m3dLoadIdentity44(transMatrix);
		m3dTranslationMatrix44(translationMatrix, Tx, Ty, -Tz);
		M3DMatrix44f rotationMatrix;
		m3dLoadIdentity44(rotationMatrix);
		m3dRotationMatrix44(rotationMatrix, (Rx), 1.0f, 0.0f, 0.0f);
		m3dMatrixMultiply44(transM1, rotationMatrix, translationMatrix);
		m3dLoadIdentity44(rotationMatrix);
		m3dRotationMatrix44(rotationMatrix, (Ry), 0.0f, 1.0f, 0.0f);
		m3dMatrixMultiply44(transMatrix, rotationMatrix, transM1);
	}

	void setEyePosition(const M3DVector3f vLocation)
	{
		memcpy(m_eyePos, vLocation, sizeof(float) * 3);
	}

	inline void setModelViewMatrix(const M3DMatrix44f & modViewMatrix)
	{
		memcpy(this->modelViewMatrix, modViewMatrix, sizeof(float) * 16);
	}

	inline void setProjectMatrix(const M3DMatrix44f & mvpMatrix)
	{
		memcpy(this->mvpMatrix, mvpMatrix, sizeof(float) * 16);
	}

	inline void loadShaderSrc(const char * szShaderSrc, GLuint shader)
	{
		GLchar * szString[1];
		szString[0] = (GLchar *)szShaderSrc;
		glShaderSource(shader, 1, (const GLchar **)szString, NULL);
	}

	inline bool loadShaderFile(const char *szFile, GLuint shader)
	{
		int shaderLength = 0;
		FILE *fp;
		fp = fopen(szFile, "r");
		if (fp != NULL) {
			while (fgetc(fp) != EOF)shaderLength++;
			assert(shaderLength<MAX_SHADER_LENGTH);
			rewind(fp);
			if (shaderText != NULL) fread(shaderText, 1, shaderLength, fp);
			shaderText[shaderLength] = '\0';
			fclose(fp);

		}
		else {
			fclose(fp);
			return false;
		}
		loadShaderSrc((const char *)shaderText, shader);
		return true;
	}

	inline float urand()
	{
		return rand() / (float)RAND_MAX;
	}

	inline float gauss()
	{
		float u1 = urand();
		float u2 = urand();

		if (u1 < 1e-6f)
		{
			u1 = 1e-6f;
		}
		return (cosf(2 * 3.1415926 * u2) + 1) * 0.5;
	}

protected:

	int width;
	int height;
	int depth;

	
	GLubyte shaderText[MAX_SHADER_LENGTH];

	//M3DVector3f m_lightPos;
	M3DVector3f  m_eyePos;

	GLCameraFrame *G_LIGHTFRAME;

	M3DMatrix44f modelViewMatrix;
	M3DMatrix44f mvpMatrix;
	M3DMatrix44f invertModeViewMatrix;
	M3DMatrix44f transMatrix;

	float BSplineMatrix[16] =
	{
		-1.0, 3.0, -3.0, 1.0,
		3.0,-6.0,  3.0, 0.0,
		-3.0, 0.0,  3.0, 0.0,
		1.0, 4.0,  1.0, 0.0
	};

	float* GridValue;
	float* d_GridValue;

	//PointProperty* SamplePointProperty;
	//PointProperty* d_SamplePointProperty;

	CellSample* cellSampleInfo;

	float4* SamplePoints;
	float4* d_SamplePoints;
	float4* SampleNormals;
	float4* SphereVertices;
	
	int PointsNumber;
	int* d_PointsNumber;
	
	GLuint SampleRenderProg;
	GLuint SampleDiskRenderProg;
	GLuint SphereRenderProg;
	GLuint SampleVertexBuffer;
	GLuint SampleNormalBuffer;
	GLuint SamplePosBuffer;
	GLuint SpherePosBuffer;
	GLuint indexBuffer;

	GLuint locMVP;
	
	cudaGraphicsResource *cuda_sampleVertex_resource;
	
	virtual void startUp();
	void createVBO(GLuint *vbo, int size);
	GLuint loadShaderWithAttributes(const char * szVertexProg, const char * szGeometryProg, const char * szFragProg, ...);
	GLuint loadShaderPairWithAttributes(const char * szVertexProg, const char * szFragProg, ...);
	void TorusGridGenerator(float* Grid);
	void SphereGridGenerator(float* Grid);
	void Sphere(float4* vertices, GLuint* id);
	void Torus(float4* vertices, GLuint* id);
	void RandomSamplesOnSurface(float4* Points, CellSample* SampleInfo, int* SampleNum);
	double Determinant(int n, float* A);
	double Cofactor(float *A, int m, int n, int k);
	void InverseMatrix(float* A, float* InverseA, int n);
	void ComputeProperty(PointProperty* Point, int n);
	float BSplineValue(float x, float y, float z, float* gridValue);
	float4 NormalComutation(float x, float y, float z, float* gridValue);
	float XX_2orderDerivative(float x, float y, float z, float* gridValue);
	float YY_2orderDerivative(float x, float y, float z, float* gridValue);
	float ZZ_2orderDerivative(float x, float y, float z, float* gridValue);
	float XY_2orderDerivative(float x, float y, float z, float* gridValue);
	float XZ_2orderDerivative(float x, float y, float z, float* gridValue);
	float YZ_2orderDerivative(float x, float y, float z, float* gridValue);
	void MatrixMultiplication(float* A, float* B, float* C, int n);
	void ScalarMatrixMultiplication(float* A, float* B, float R, int n);
	float2 CurvatureComputation(float x, float y, float z, float* gridValue);
	float2 CurvatureAssessment(float x, float y, float z, float* gridValue);	
	void SurfaceAttraction(float4* Points, float* gridValue, int n);
	float4 PointAttraction(float x, float y, float z, float* gridValue);
	float4 BiSearch(float x0, float y0, float z0, float Nx, float Ny, float Nz, float t, int RecursiveLayer);
	float4 BiSearchNoRecursion(float x0, float y0, float z0, float x1, float y1, float z1, float Nx, float Ny, float Nz, float* gridValue);
	void BsplinePropertyComputation(PointProperty* Point, float* gridValue, int n);	
	float PointsDistance(float x0, float y0, float z0, float x1, float y1, float z1);
	float Eij(float Rij);
	float Ei(float x, float y, float z, CellSample* SampleInfo, float4* Points, int count);
	float DerEij(float Rij);
	float D2Eij(float Rij);
	void OuterProduct(float* V, float* M, int num);
	void MatrixAddition(float* Ma, float* Mb, float* Mc, int num);
	void MatVecProduct(float* M, float* V, float* Vresult, int num);
	void Redistribution(CellSample* SampleInfo, float4* samplePoints, int pointsNumber, float* gridValue);	
	void Di(float x0, float y0, float z0, CellSample* SampleInfo, float4* Points, float* result);
	void Hi(float x0, float y0, float z0, CellSample* SampleInfo, float4* Points, float* result);
	void PlaneGridGenerator(float* Grid);	
	void InitPlane();
	void InitSphere();
	void InitTorus();
	void RenderSphereDisks();
	void RenderPlaneSamples();
	void RenderTorusDisks();
	void Cuda_SampleDisplay();
	void Test();

};