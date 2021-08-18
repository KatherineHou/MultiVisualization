#include "type_define.h"



#ifndef _MULTIVIS_COMPUTE_CU_
#define _MULTIVIS_COMPUTE_CU_


#define NUM_THREADS_PER_BLOCK   4


cudaArray *d_scalarFieldArray = 0;

texture<float, 3, cudaReadModeElementType>scalarTex;

dim3 grids;
dim3 blocks;

Grid_Param  *Grid_Para;
__constant__ Grid_Param d_Grid_Para;



extern "C" void initCuda(float *h_ScalarField, int width, int height, int depth, int cubesize, float iso)
{
	Grid_Para = new Grid_Param(width, height, depth, cubesize, iso);
	cudaMemcpyToSymbol(d_Grid_Para, Grid_Para, sizeof(Grid_Param));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	cudaExtent volumeDataSize = make_cudaExtent(height, width, depth);

	cudaMalloc3DArray(&d_scalarFieldArray, &channelDesc, volumeDataSize);

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_ScalarField, height * sizeof(float), height, width);
	copyParams.dstArray = d_scalarFieldArray;
	copyParams.extent = volumeDataSize;
	copyParams.kind = cudaMemcpyHostToDevice;

	cudaMemcpy3D(&copyParams);

	scalarTex.normalized = TRUE;
	scalarTex.filterMode = cudaFilterModeLinear;
	scalarTex.addressMode[0] = cudaAddressModeWrap;
	scalarTex.addressMode[1] = cudaAddressModeWrap;
	scalarTex.addressMode[2] = cudaAddressModeWrap;
	
	cudaBindTextureToArray(scalarTex, d_scalarFieldArray, channelDesc);

}
extern "C" void updateScalarTexture(void *d_scalarField)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	int h = Grid_Para->height;
	int w = Grid_Para->width;
	int d = Grid_Para->depth;

	cudaExtent volumeDataSize = make_cudaExtent(h, w, d);

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(d_scalarField, h * sizeof(float4), h, w);
	copyParams.dstArray = d_scalarFieldArray;
	copyParams.extent = volumeDataSize;
	copyParams.kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy3D(&copyParams);

}


__device__ bool isInBoundary(int x, int y, int z, int bx, int by, int bz, int pointNumber)
{
	int offset = z * bx * by + y * bx + x;
	if (offset <= pointNumber) return true;

	//if (x < bx  && y < by   && z < bz)return true;
	return false;
}

__device__ float BSplineValue(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex, cubeSize, domainSize;
	float  x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	int width = d_Grid_Para.width;
	int height = d_Grid_Para.height;
	int depth = d_Grid_Para.depth;

	cubeSize = d_Grid_Para.cubeSize;
	domainSize = cubeSize / 2;

	float CellSize;
	CellSize = float(cubeSize) / float(width);

	float BSplineMatrix[16] =
	{
		-1.0, 3.0, -3.0, 1.0,
		3.0,-6.0,  3.0, 0.0,
		-3.0, 0.0,  3.0, 0.0,
		1.0, 4.0,  1.0, 0.0
	};
	
	//printf("%f %f %f\n", x, y, z);

	//int CellSize;
	//CellSize = CUBESIZE / GRIDWIDTH;

	if (x >= domainSize || x <= ((-1)*domainSize) || y >= domainSize || y <= ((-1)*domainSize) || z >= domainSize || z <= ((-1)*domainSize))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;




	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	//Ux = x - x0;
	//Uy = y - y0;
	//Uz = z - z0;

	//printf("P: %f %f %f    P0: %f %f %f     U: %f %f %f\n", x, y, z, x0, y0, z0, Ux, Uy, Uz);
	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + width / 2 - 1;
				jIndex = y0 / CellSize + j + height / 2 - 1;
				kIndex = depth / 2 - z0 / CellSize - k + 1;

				//Value = tex3D(scalarTex, iIndex, jIndex, kIndex);

				//printf("%d %d %d   %f\n", iIndex, jIndex, kIndex, Value);

				offset = kIndex * width * height + jIndex * width + iIndex;
				Value = gridValue[offset];
				//printf("%d %d %d   %f\n", iIndex, jIndex, kIndex, Value);

				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;
				//printf("ijk and value: %d %d %d  %f   Sx: %f\n", iIndex, jIndex, kIndex, Value, Sx);

			}

			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;
			//printf("\nSx: %f   Sy: %f \n", Sx, Sy);
		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;
		//printf("\nSy: %f   Sz: %f \n", Sy, Sz);
	}
	//printf("the result is: %f %f %f     %f\n\n\n\n\n", x, y, z, Sz);

	return (Sz);

}

__device__ float4 NormalComutation(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex, cubeSize, domainSize;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz, NormalMagnitude;
	float Nx, Ny, Nz;
	float4 NormalDirection;

	int width = d_Grid_Para.width;
	int height = d_Grid_Para.height;
	int depth = d_Grid_Para.depth;

	cubeSize = d_Grid_Para.cubeSize;
	domainSize = cubeSize / 2;

	float CellSize;
	CellSize = float(cubeSize) / float(width);

	float BSplineMatrix[16] =
	{
		-1.0, 3.0, -3.0, 1.0,
		3.0,-6.0,  3.0, 0.0,
		-3.0, 0.0,  3.0, 0.0,
		1.0, 4.0,  1.0, 0.0
	};

	if (x >= domainSize || x <= ((-1)*domainSize) || y >= domainSize || y <= ((-1)*domainSize) || z >= domainSize || z <= ((-1)*domainSize))
		return (make_float4(0, 0, 0, 0));
	//int CellSize;
	//CellSize = CUBESIZE / GRIDWIDTH;

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;


	//Ux = x - x0;
	//Uy = y - y0;
	//Uz = z - z0;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + width / 2 - 1;
				jIndex = y0 / CellSize + j + height / 2 - 1;
				kIndex = depth / 2 - z0 / CellSize - k + 1;

				//Value = tex3D(scalarTex, iIndex, jIndex, kIndex);

				offset = kIndex * width * height + jIndex * width + iIndex;
				Value = gridValue[offset];

				Sx += Value*(3.0*Ux*Ux*BSplineMatrix[i] + 2.0*Ux*BSplineMatrix[4 + i] + BSplineMatrix[8 + i]) / 6.0 / CellSize;

			}

			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;


		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;


	}
	Nx = Sz;


	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + width / 2 - 1;
				jIndex = y0 / CellSize + j + height / 2 - 1;
				kIndex = depth / 2 - z0 / CellSize - k + 1;

				//Value = tex3D(scalarTex, iIndex, jIndex, kIndex);

				offset = kIndex * width * height + jIndex * width + iIndex;
        		Value = gridValue[offset];

				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;

			}

			Sy += Sx*(3.0*Uy*Uy*BSplineMatrix[j] + 2.0*Uy*BSplineMatrix[4 + j] + BSplineMatrix[8 + j]) / 6.0 / CellSize;

		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;

	}
	Ny = Sz;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + width / 2 - 1;
				jIndex = y0 / CellSize + j + height / 2 - 1;
				kIndex = depth / 2 - z0 / CellSize - k + 1;
				
				//Value = tex3D(scalarTex, iIndex, jIndex, kIndex);

				offset = kIndex * width * height + jIndex * width + iIndex;
				Value = gridValue[offset];
				
				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;

			}

			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;

		}

		Sz += Sy*(3.0*Uz*Uz*BSplineMatrix[k] + 2.0*Uz*BSplineMatrix[4 + k] + BSplineMatrix[8 + k]) / 6.0 / CellSize;

	}
	Nz = Sz;

	NormalMagnitude = sqrt(Nx*Nx + Ny*Ny + Nz*Nz);

	Nx = Nx / NormalMagnitude;
	Ny = Ny / NormalMagnitude;
	Nz = Nz / NormalMagnitude;
	NormalDirection = make_float4(Nx, Ny, Nz, NormalMagnitude);
	//printf("%f %f %f\n", Nx, Ny, Nz);

	return (NormalDirection);

}

__device__ float4 BiSearch(float x0, float y0, float z0, float x1, float y1, float z1, float Nx, float Ny, float Nz, float* gridValue)
{
	float xm, ym, zm;
	float Value, Errors, CellSize, MajorRadius, IsoValue;
	float4 result;
	int i = 0;

	CellSize = float(d_Grid_Para.cubeSize) / float(d_Grid_Para.width);
	//MajorRadius = d_Grid_Para.majorRadius;
	IsoValue = d_Grid_Para.isoValue;
	
	//x1 = x0 + t * CellSize * Nx;
	//y1 = y0 + t * CellSize * Ny;
	//z1 = z0 + t * CellSize * Nz;

	xm = (x0 + x1) / 2.0;
	ym = (y0 + y1) / 2.0;
	zm = (z0 + z1) / 2.0;


	//MajorDistance = sqrt(xm*xm + zm*zm) - MajorRadius;
	//Value = sqrt(MajorDistance*MajorDistance + ym*ym);

	Value = BSplineValue(xm, ym, zm, gridValue);

	Errors = Value - IsoValue;
	if (Errors < 0)
		Errors = (-1.0) * Errors;
	

	while (Errors > TOLERANCE && i < MAXRECURSE)
	{
		
		if (Value > IsoValue)
		{
			x1 = xm;
			y1 = ym;
			z1 = zm;

		}
		else
		{
			x0 = xm;
			y0 = ym;
			z0 = zm;
		}
		xm = (x0 + x1) / 2.0;
		ym = (y0 + y1) / 2.0;
		zm = (z0 + z1) / 2.0;


		//MajorDistance = sqrt(xm*xm + zm*zm) - MajorRadius;
		//Value = sqrt(MajorDistance*MajorDistance + ym*ym);

		Value = BSplineValue(xm, ym, zm, gridValue);
		

		Errors = Value - IsoValue;
		if (Errors < 0)
			Errors = (-1.0) * Errors;
		i++;
	}

	result = make_float4(xm, ym, zm, 1);
	return (result);
}

__global__ void SurfaceAttraction_kernel(float4* Points, float* gridValue, int* n)
{
	float x0, y0, z0, dx, dy, dz, x1, y1, z1, Value, v0, v1, CellSize, IsoValue;
	float x, y, z;
	float4 normal, SurfacePoint;

	CellSize = float(d_Grid_Para.cubeSize) / float(d_Grid_Para.width);
	IsoValue = d_Grid_Para.isoValue;

	int width = d_Grid_Para.width;
	int height = d_Grid_Para.height;
	int depth = d_Grid_Para.depth;
	
	int NewDepth = *n / (height * width) + ((*n % (height * width)) ? 1 : 0);
	
	int block_size_t = NUM_THREADS_PER_BLOCK * NUM_THREADS_PER_BLOCK * NUM_THREADS_PER_BLOCK * sizeof(float);
	
	int gridx = blockDim.x * blockIdx.x + threadIdx.x;
	int gridy = blockDim.y * blockIdx.y + threadIdx.y;
	int gridz = blockDim.z * blockIdx.z + threadIdx.z;
		
	int offset = gridz * width * height + gridy * width + gridx;
	

	if (isInBoundary(gridx, gridy, gridz, width, height, NewDepth, *n)) 
	{
		
		x = Points[offset].x;
		y = Points[offset].y;
		z = Points[offset].z;
		//printf("%d  %f %f %f\n", offset, Points[offset].x, y, z);
		
		Value = BSplineValue(x, y, z, gridValue);

		//printf("%d   %f\n", offset, Value);

		normal = NormalComutation(x, y, z, gridValue);
		dx = normal.x;
		dy = normal.y;
		dz = normal.z;

		//dx = Property[offset].Nx;
		//dy = Property[offset].Ny;
		//dz = Property[offset].Nz;

		if (Value < IsoValue)
		{
			//x0 = Property[offset].x;
			//y0 = Property[offset].y;
			//z0 = Property[offset].z;

			x0 = x;
			y0 = y;
			z0 = z;

			x1 = x0 + 1.8 * CellSize * dx;
			y1 = y0 + 1.8 * CellSize * dy;
			z1 = z0 + 1.8 * CellSize * dz;

			v0 = Value;
			v1 = BSplineValue(x1, y1, z1, gridValue);

			if (v0 <= IsoValue && v1 <= IsoValue || v0 >= IsoValue && v1 >= IsoValue)
			{
				x1 = x1 + 1.8 * CellSize * dx;
				y1 = y1 + 1.8 * CellSize * dy;
				z1 = z1 + 1.8 * CellSize * dz;
			}

		}
		else
		{
			//x0 = Property[offset].x - 1.8 * CellSize * dx;
			//y0 = Property[offset].y - 1.8 * CellSize * dy;
			//z0 = Property[offset].z - 1.8 * CellSize * dz;

			x1 = x;
			y1 = y;
			z1 = z;

			x0 = x1 - 1.8 * CellSize * dx;
			y0 = y1 - 1.8 * CellSize * dy;
			z0 = z1 - 1.8 * CellSize * dz;

			v1 = Value;
			v0 = BSplineValue(x0, y0, z0, gridValue);

			if (v0 <= IsoValue && v1 <= IsoValue || v0 >= IsoValue && v1 >= IsoValue)
			{
				x0 = x0 - 1.8 * CellSize * dx;
				y0 = y0 - 1.8 * CellSize * dy;
				z0 = z0 - 1.8 * CellSize * dz;
			}

		}

		SurfacePoint = BiSearch(x0, y0, z0, x1, y1, z1, dx, dy, dz, gridValue);

		Points[offset] = SurfacePoint;
		//printf("%f  %f  %f\n", SurfacePoint.x, SurfacePoint.x, SurfacePoint.x);

		//Property[offset].x = SurfacePoint.x;
		//Property[offset].y = SurfacePoint.y;
		//Property[offset].z = SurfacePoint.z;
		
	}

}

extern "C" void Cuda_SurfaceAttraction(float4* Points, float* gridValue, int* n)
{
	int width = Grid_Para->width;
	int height = Grid_Para->height;
	int depth = Grid_Para->depth;


	int PointsNumber;
	cudaMemcpy(&PointsNumber, n, sizeof(int), cudaMemcpyDeviceToHost);

	

	int NewDepth = PointsNumber / (height * width) + ((PointsNumber % (height * width)) ? 1 : 0);

	grids.x = width / NUM_THREADS_PER_BLOCK + ((width % NUM_THREADS_PER_BLOCK) ? 1 : 0);
	grids.y = height / NUM_THREADS_PER_BLOCK + ((height % NUM_THREADS_PER_BLOCK) ? 1 : 0);
	grids.z = NewDepth / NUM_THREADS_PER_BLOCK + ((NewDepth % NUM_THREADS_PER_BLOCK) ? 1 : 0);

	blocks.x = NUM_THREADS_PER_BLOCK;
	blocks.y = NUM_THREADS_PER_BLOCK;
	blocks.z = NUM_THREADS_PER_BLOCK;
	
	

	SurfaceAttraction_kernel << <grids, blocks >> > (Points, gridValue, n);
		
}


#endif