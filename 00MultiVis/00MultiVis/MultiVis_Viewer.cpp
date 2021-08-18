#include"MultiVis_Viewer.h"

using namespace std;

extern "C" void initCuda(float *h_ScalarField, int width, int height, int depth, int cubesize, float iso);
extern "C" void Cuda_SurfaceAttraction(float4* Points, float* gridValue, int* n);

MULTIVIS_RENDER::~MULTIVIS_RENDER()
{
	glDeleteBuffers(1, &SampleVertexBuffer);
	glDeleteBuffers(1, &SampleNormalBuffer);
	glDeleteBuffers(1, &SpherePosBuffer);
	glDeleteBuffers(1, &SamplePosBuffer);
	glDeleteBuffers(1, &indexBuffer);
		
	glDeleteProgram(SampleRenderProg);
	glDeleteProgram(SampleDiskRenderProg);
	glDeleteProgram(SphereRenderProg);
	
	delete(G_LIGHTFRAME);

	free(GridValue);
	free(SamplePoints);
	free(SampleNormals);
	//free(SamplePointProperty);
	free(cellSampleInfo);

}
MULTIVIS_RENDER::MULTIVIS_RENDER(int w, int h, int d)
{
	this->width = w;
	this->height = h;
	this->depth = d;

	G_LIGHTFRAME = new GLCameraFrame();

	startUp();
}
GLuint MULTIVIS_RENDER::loadShaderWithAttributes(const char * szVertexProg, const char * szGeometryProg, const char * szFragProg, ...)
{
	GLint verTexShader, fragMentShader, geometryShader;
	GLint hReturn = 0;
	verTexShader = glCreateShader(GL_VERTEX_SHADER);
	fragMentShader = glCreateShader(GL_FRAGMENT_SHADER);
	geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
	GLint testVal = 0;
	if (loadShaderFile(szVertexProg, verTexShader) == false)
	{
		glDeleteShader(verTexShader);
		glDeleteShader(geometryShader);
		glDeleteShader(fragMentShader);
		fprintf(stderr, "The shader at %s could not be found !", szVertexProg);
		return (GLuint)NULL;
	}
	if (loadShaderFile(szGeometryProg, geometryShader) == false)
	{
		glDeleteShader(verTexShader);
		glDeleteShader(geometryShader);
		glDeleteShader(fragMentShader);
		fprintf(stderr, "The shader at %s could not be found !", szGeometryProg);
		return (GLuint)NULL;
	}
	if (loadShaderFile(szFragProg, fragMentShader) == false)
	{
		glDeleteShader(verTexShader);
		glDeleteShader(geometryShader);
		glDeleteShader(fragMentShader);
		fprintf(stderr, "The shader at %s could not be found !", szFragProg);
		return (GLuint)NULL;
	}

	glCompileShader(verTexShader);
	glCompileShader(fragMentShader);
	glCompileShader(geometryShader);

	glGetShaderiv(verTexShader, GL_COMPILE_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char InfoLog[1024];
		glGetShaderInfoLog(verTexShader, 1024, NULL, InfoLog);
		fprintf(stderr, "The shader at file %s failed to compile with the following error :\n%s\n", szVertexProg, InfoLog);
		glDeleteShader(verTexShader);
		glDeleteShader(geometryShader);
		glDeleteShader(fragMentShader);
		return (GLuint)NULL;
	}
	glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char InfoLog[1024];
		glGetShaderInfoLog(geometryShader, 1024, NULL, InfoLog);
		fprintf(stderr, "The shader at file %s failed to compile with the following error:\n%s\n", szGeometryProg, InfoLog);
		glDeleteShader(verTexShader);
		glDeleteShader(geometryShader);
		glDeleteShader(fragMentShader);
		return (GLuint)NULL;
	}
	glGetShaderiv(fragMentShader, GL_COMPILE_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char InfoLog[1024];
		glGetShaderInfoLog(fragMentShader, 1024, NULL, InfoLog);
		fprintf(stderr, "The shader at file %s failed to compile with the following error:\n%s\n", szFragProg, InfoLog);
		glDeleteShader(verTexShader);
		glDeleteShader(geometryShader);
		glDeleteShader(fragMentShader);
		return (GLuint)NULL;
	}
	hReturn = glCreateProgram();
	glAttachShader(hReturn, verTexShader);
	glAttachShader(hReturn, geometryShader);
	glAttachShader(hReturn, fragMentShader);

	va_list attributeList;
	va_start(attributeList, szFragProg);



	char *szNextArg;
	int argCount = va_arg(attributeList, int);
	for (int i = 0; i < argCount; i++) {
		int index = va_arg(attributeList, int);
		szNextArg = va_arg(attributeList, char *);
		glBindAttribLocation(hReturn, index, szNextArg);
	}
	va_end(attributeList);

	glLinkProgram(hReturn);

	glDeleteShader(verTexShader);
	glDeleteShader(geometryShader);
	glDeleteShader(fragMentShader);

	glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char infoLog[1024];
		glGetProgramInfoLog(hReturn, 1024, NULL, infoLog);
		fprintf(stderr, "The program %s , %s and %s failed to link with following errors :\n%s\n", szVertexProg, szGeometryProg, szFragProg, infoLog);
		glDeleteProgram(hReturn);
		return (GLuint)NULL;
	}
	return hReturn;
}
GLuint MULTIVIS_RENDER::loadShaderPairWithAttributes(const char * szVertexProg, const char * szFragProg, ...)
{
	GLint verTexShader, fragMentShader;
	GLint hReturn = 0;
	verTexShader = glCreateShader(GL_VERTEX_SHADER);
	fragMentShader = glCreateShader(GL_FRAGMENT_SHADER);
	GLint testVal = 0;
	if (loadShaderFile(szVertexProg, verTexShader) == false)
	{
		glDeleteShader(verTexShader);
		glDeleteShader(fragMentShader);
		fprintf(stderr, "The shader at %s could not be found !", szVertexProg);
		return (GLuint)NULL;
	}
	if (loadShaderFile(szFragProg, fragMentShader) == false)
	{
		glDeleteShader(verTexShader);
		glDeleteShader(fragMentShader);
		fprintf(stderr, "The shader at %s could not be found !", szFragProg);
		return (GLuint)NULL;
	}

	glCompileShader(verTexShader);
	glCompileShader(fragMentShader);

	glGetShaderiv(verTexShader, GL_COMPILE_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char InfoLog[1024];
		glGetShaderInfoLog(verTexShader, 1024, NULL, InfoLog);
		fprintf(stderr, "The shader at file %s failed to compile with the following error :\n%s\n", szVertexProg, InfoLog);
		glDeleteShader(verTexShader);
		glDeleteShader(fragMentShader);
		return (GLuint)NULL;
	}
	glGetShaderiv(fragMentShader, GL_COMPILE_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char InfoLog[1024];
		glGetShaderInfoLog(fragMentShader, 1024, NULL, InfoLog);
		fprintf(stderr, "The shader at file %s failed to compile with the following error:\n%s\n", szFragProg, InfoLog);
		glDeleteShader(verTexShader);
		glDeleteShader(fragMentShader);
		return (GLuint)NULL;
	}
	hReturn = glCreateProgram();
	glAttachShader(hReturn, verTexShader);
	glAttachShader(hReturn, fragMentShader);

	va_list attributeList;
	va_start(attributeList, szFragProg);



	char *szNextArg;
	int argCount = va_arg(attributeList, int);
	for (int i = 0; i < argCount; i++) {
		int index = va_arg(attributeList, int);
		szNextArg = va_arg(attributeList, char *);
		glBindAttribLocation(hReturn, index, szNextArg);
	}
	va_end(attributeList);

	glLinkProgram(hReturn);

	glDeleteShader(verTexShader);
	glDeleteShader(fragMentShader);


	glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
	if (testVal == GL_FALSE) {
		char infoLog[1024];
		glGetProgramInfoLog(hReturn, 1024, NULL, infoLog);
		fprintf(stderr, "The program %s and %s failed to link with following errors :\n%s\n", szVertexProg, szFragProg, infoLog);
		glDeleteProgram(hReturn);
		return (GLuint)NULL;
	}
	return hReturn;

}
void MULTIVIS_RENDER::createVBO(GLuint *vbo, int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void MULTIVIS_RENDER::startUp()
{
	GridValue = (float*)malloc(sizeof(float)*GRIDWIDTH*GRIDHIGHT*GRIDDEPTH);
	SamplePoints = (float4*)malloc(sizeof(float4)*GRIDWIDTH*GRIDHIGHT*GRIDDEPTH);
	SampleNormals = (float4*)malloc(sizeof(float4)*GRIDWIDTH*GRIDHIGHT*GRIDDEPTH);
    //SamplePointProperty = (PointProperty*)malloc(sizeof(PointProperty)*GRIDWIDTH*GRIDHIGHT*GRIDDEPTH);
	cellSampleInfo = (CellSample*)malloc(sizeof(CellSample)*GRIDWIDTH*GRIDHIGHT*GRIDDEPTH);
	SphereVertices = (float4*)malloc(sizeof(float4)*(SPHERESEGMENTS+1)*(SPHERESEGMENTS+1));

	int vertexSize;
	vertexSize = sizeof(float4) * GRIDWIDTH*GRIDHIGHT*GRIDDEPTH;

	//initCuda(void *h_ScalarField, int width, int height, int depth, int cubesize, float major, float minor, float iso);

	SampleRenderProg = loadShaderPairWithAttributes("shader/SamplePoints.vert", "shader/SamplePoints.frag", 1, GLT_ATTRIBUTE_VERTEX, "vVertex");
	SampleDiskRenderProg = loadShaderWithAttributes("shader/SampleDisks.vert", "shader/SampleDisks.geom", "shader/SampleDisks.frag", 2, GLT_ATTRIBUTE_VERTEX, "vVertex", GLT_ATTRIBUTE_TEXCOORD1, "nNormal");
	SphereRenderProg = loadShaderPairWithAttributes("shader/Sphere.vert", "shader/Sphere.frag", 1, GLT_ATTRIBUTE_VERTEX, "vVertex");

	glGenBuffers(1, &SampleVertexBuffer);
	glBindBuffer(GL_TEXTURE_BUFFER, SampleVertexBuffer);
	glBufferData(GL_TEXTURE_BUFFER, GRIDWIDTH*GRIDHIGHT*GRIDDEPTH, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	glGenBuffers(1, &SampleNormalBuffer);
	glBindBuffer(GL_TEXTURE_BUFFER, SampleNormalBuffer);
	glBufferData(GL_TEXTURE_BUFFER, GRIDWIDTH*GRIDHIGHT*GRIDDEPTH, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	glGenBuffers(1, &SpherePosBuffer);
	glBindBuffer(GL_TEXTURE_BUFFER, SpherePosBuffer);
	glBufferData(GL_TEXTURE_BUFFER, 50*50, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	createVBO(&SamplePosBuffer, vertexSize);
	cudaGraphicsGLRegisterBuffer(&cuda_sampleVertex_resource, SamplePosBuffer, cudaGraphicsMapFlagsNone);

	if (ObjectSelection == 0)
	    InitSphere();
    if (ObjectSelection == 1)
		InitTorus();
	if (ObjectSelection == 2)
	    InitPlane();
	
}

void MULTIVIS_RENDER::TorusGridGenerator(float* Grid)
{
	int i, j, k, offset;
	float x, y, z, MinorDistance;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	for (k = 0; k < GRIDDEPTH; k++)
		for (j = 0; j<GRIDHIGHT; j++)
			for (i = 0; i<GRIDWIDTH; i++)
			{
				offset = k * GRIDWIDTH * GRIDHIGHT + j * GRIDWIDTH + i;
				x = i * CellSize - GRIDWIDTH * CellSize / 2;
				y = j * CellSize - GRIDHIGHT * CellSize / 2;
				z = k * CellSize * (-1) + GRIDDEPTH * CellSize / 2;


				MinorDistance = sqrt(x*x + z*z) - TORUSRADIUS;
				Grid[offset] = sqrt(MinorDistance*MinorDistance + y*y);
				//Grid[offset] = MinorDistance*MinorDistance + y*y;

			}
}

void MULTIVIS_RENDER::SphereGridGenerator(float* Grid)
{
	int i, j, k, offset;
	float x, y, z;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	for (k = 0; k < GRIDDEPTH; k++)
		for (j = 0; j<GRIDHIGHT; j++)
			for (i = 0; i<GRIDWIDTH; i++)
			{
				offset = k * GRIDWIDTH * GRIDHIGHT + j * GRIDWIDTH + i;
				x = i * CellSize - GRIDWIDTH * CellSize / 2;
				y = j * CellSize - GRIDHIGHT * CellSize / 2;
				z = k * CellSize * (-1) + GRIDDEPTH * CellSize / 2;

				Grid[offset] = x*x + y*y + z*z;

			}

}

void MULTIVIS_RENDER::Sphere(float4* vertices, GLuint* id)
{
	//Generate a sphere, which can be used as the background of sphere samples
	int i, j;
	float SphereRadius;
	SphereRadius = IsoValue - 0.02;

	for (int y = 0; y <= SPHERESEGMENTS; y++)
	{
		for (int x = 0; x <= SPHERESEGMENTS; x++)
		{
			float xSegment = (float)x / (float)SPHERESEGMENTS;
			float ySegment = (float)y / (float)SPHERESEGMENTS;
			float xPos = SphereRadius *cos(xSegment * 2.0f * PI) * sin(ySegment * PI);
			float yPos = SphereRadius *cos(ySegment * PI);
			float zPos = SphereRadius *sin(xSegment * 2.0f * PI) * sin(ySegment * PI);

			vertices[y*SPHERESEGMENTS + x] = make_float4(xPos, yPos, zPos, 1);
			
		}
	}

	//Generate the sphere render index
	int size = (SPHERESEGMENTS)*(SPHERESEGMENTS) * sizeof(GLuint) * 6;
	glGenBuffersARB(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);
	GLuint *indices = (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_WRITE);
	if (!indices)return;

	for (i = 0; i < SPHERESEGMENTS; i++)
	{
		for (j = 0; j < SPHERESEGMENTS; j++) {
			//first face 
			*indices++ = i*SPHERESEGMENTS + j;
			*indices++ = (i + 1)*SPHERESEGMENTS + j;
			*indices++ = (i + 1)*SPHERESEGMENTS + j + 1;

			//second face
			*indices++ = i*SPHERESEGMENTS + j + 1;
			*indices++ = i*SPHERESEGMENTS + j;
			*indices++ = (i + 1)*SPHERESEGMENTS + j + 1;
		}
	}
		
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}

void MULTIVIS_RENDER::Torus(float4* vertices, GLuint* id)
{
	//Generate a Torus, which can be used as the background of sphere samples
	int i, j;
	//float R = 2.0;
	float r;

	r = IsoValue - 0.02;


	for (int i = 0; i <= SPHERESEGMENTS; i++)
	{
		for (int j = 0; j <= SPHERESEGMENTS; j++)
		{
			float Phi = (float)i / (float)SPHERESEGMENTS;
			float Theta = (float)j / (float)SPHERESEGMENTS;
			float xPos = (TORUSRADIUS + r *cos(Theta * 2.0f * PI)) * cos(Phi * 2.0 * PI);
			float yPos = r *sin(Theta * 2.0f * PI);
			float zPos = (TORUSRADIUS + r *cos(Theta * 2.0f * PI)) * sin(Phi * 2.0 * PI);

			vertices[i*SPHERESEGMENTS + j] = make_float4(xPos, yPos, zPos, 1);

		}
	}

	//Generate the torus render index
	int size = (SPHERESEGMENTS)*(SPHERESEGMENTS) * sizeof(GLuint) * 6;
	glGenBuffersARB(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);
	GLuint *indices = (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_WRITE);
	if (!indices)return;

	for (i = 0; i < SPHERESEGMENTS; i++)
	{
		for (j = 0; j < SPHERESEGMENTS; j++) {
			//first face 
			*indices++ = i*SPHERESEGMENTS + j;
			*indices++ = (i + 1)*SPHERESEGMENTS + j;
			*indices++ = (i + 1)*SPHERESEGMENTS + j + 1;

			//second face
			*indices++ = i*SPHERESEGMENTS + j + 1;
			*indices++ = i*SPHERESEGMENTS + j;
			*indices++ = (i + 1)*SPHERESEGMENTS + j + 1;
		}
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}

void MULTIVIS_RENDER::PlaneGridGenerator(float* Grid)
{
	int i, j, k, offset;
	float x, y, z;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	for (k = 0; k < GRIDDEPTH; k++)
		for (j = 0; j<GRIDHIGHT; j++)
			for (i = 0; i<GRIDWIDTH; i++)
			{
				offset = k * GRIDWIDTH * GRIDHIGHT + j * GRIDWIDTH + i;
				x = i * CellSize - GRIDWIDTH * CellSize / 2;
				y = j * CellSize - GRIDHIGHT * CellSize / 2;
				z = k * CellSize * (-1) + GRIDDEPTH * CellSize / 2;

				Grid[offset] = z;

			}

}

void MULTIVIS_RENDER::RandomSamplesOnSurface(float4* Points, CellSample* SampleInfo, int* SampleNum)
{
	int i, j, k, offset, SampleCellNumber, SampleCount, ii, jj, kk, CellOffset;
	int dx, dy, dz, TempOffset, flag[7], tt, CellPointNumber;
	float x, y, z, PoissionMean;

	float4* SampleCellIndex;
	SampleCellIndex = (float4*)malloc(sizeof(float4)*GRIDWIDTH*GRIDHIGHT*GRIDDEPTH);

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	//SampleInfo array records how many sample points in each cell, and the index of the first point in this cell
	//SampleInfo can be used in samples redistribution, to search the neighbor points within sigma around one point
	//It's initialized with 0
	for (CellOffset = 0; CellOffset < GRIDWIDTH*GRIDHIGHT*GRIDDEPTH; CellOffset++)
	{
		SampleInfo[CellOffset].Serial = 0;
		SampleInfo[CellOffset].Number = 0;
	}

	//SampleCellNumber records how many cells contain a piece of iso-surface, and it's initialized with 0
	//SampleCellIndex records the position of each cell which contains iso-surface
	SampleCellNumber = 0;

	for (k = 0; k < (GRIDDEPTH - 1); k++)
		for (j = 0; j<(GRIDHIGHT - 1); j++)
			for (i = 0; i < (GRIDWIDTH - 1); i++)
			{
				offset = k * GRIDWIDTH * GRIDHIGHT + j * GRIDWIDTH + i;
				x = i * CellSize - GRIDWIDTH * CellSize / 2;
				y = j * CellSize - GRIDHIGHT * CellSize / 2;
				z = k * CellSize * (-1) + GRIDDEPTH * CellSize / 2;

				//test if this cell include the iso-value
				tt = 0;
				for (dz = 0; dz<2; dz++)
					for (dy = 0; dy<2; dy++)
						for (dx = 0; dx < 2; dx++)
						{
							//if (MinorRadius < MajorRadius)
							//{
								TempOffset = (k + dz) * GRIDWIDTH * GRIDHIGHT + (j + dy) * GRIDWIDTH + i + dx;

								if (TempOffset != offset)
								{
									//test the value of each vertex of a cell, to see if there exists the situation that some values are greater than the iso-value
									//and others are less than the iso-value
									if ((GridValue[offset] < IsoValue) && (GridValue[TempOffset] < IsoValue) || (GridValue[offset] > IsoValue) && (GridValue[TempOffset] > IsoValue))
										flag[tt] = 0;
									else
									{
										flag[tt] = 1;
									}
									
									tt++;
									
								}
							//}
							
						}
			
				if (flag[0] || flag[1] || flag[2] || flag[3] || flag[4] || flag[5] || flag[6])
				{
					SampleCellIndex[SampleCellNumber] = make_float4(x, y, z, 1);
					SampleCellNumber++;
				}
					
			}

	//Generate sample points in each cell that contains iso-surface

	PoissionMean = (float)SampleNumber / (float)SampleCellNumber;

	default_random_engine generator;
	poisson_distribution<int> distribution(PoissionMean);

	SampleCount = 0;

	for (offset = 0; offset < SampleCellNumber; offset++)
	{

		CellPointNumber = distribution(generator);

		ii = SampleCellIndex[offset].x / CellSize + GRIDWIDTH / 2;
		jj = SampleCellIndex[offset].y / CellSize + GRIDHIGHT / 2;
		kk = GRIDDEPTH / 2 - SampleCellIndex[offset].z / CellSize;
		CellOffset = kk * GRIDWIDTH * GRIDHIGHT + jj * GRIDWIDTH + ii;

		SampleInfo[CellOffset].Serial = SampleCount;
		SampleInfo[CellOffset].Number = CellPointNumber;

		for (i = 0; i < CellPointNumber; i++)
		{
			float xr = urand();
			float yr = urand();
			float zr = urand();
			x = SampleCellIndex[offset].x + xr*CellSize;
			y = SampleCellIndex[offset].y + yr*CellSize;
			z = SampleCellIndex[offset].z - zr*CellSize;
			
			Points[SampleCount] = make_float4(x, y, z, 1);

			SampleCount++;
		}
	}

	*SampleNum = SampleCount;
	free(SampleCellIndex);
}

double MULTIVIS_RENDER::Determinant(int n, float* A)
{
	float* MinorB;
	double sum=0;
	int row, column,sign, TempRow, flag=0, IndexB, IndexA;

	MinorB = (float*)malloc(sizeof(float)*(n-1)*(n-1));

	if (n == 1)
		return A[0];

	for (row = 0; row < n; row++)
	{
		for (TempRow = 0; TempRow < n - 1; TempRow++)
		{
			for (column = 0; column < n - 1; column++)
			{
				if (TempRow < row) { 
					flag = 0;
				}
				else {
					flag = 1;
				}
				IndexB = TempRow*(n-1) + column;
				IndexA = (TempRow + flag)*n + column + 1;
				MinorB[IndexB] = A[IndexA];
				
			}
		}
		if (row % 2 == 0) 
		{ 
			sign = 1;
		}
		else 
		{ 
			sign = (-1);
		}
		
		sum += A[row*n] * Determinant(n - 1, MinorB) * sign;

	}

	free(MinorB);
	return sum;
	
}

double MULTIVIS_RENDER::Cofactor(float *A, int n, int i, int j)
{
	int row, column, IndexA=0, IndexB=0, sign;
	double CofactorValue = 0;
	float *MinorB;

	MinorB = (float*)malloc(sizeof(float)*(n-1)*(n-1));

	for (row = 0; row < n; row++)
	{
		for (column = 0; column < n; column++)
		{
			if (row != i && column != j)
			{
				MinorB[IndexB] = A[IndexA];
				IndexB++;
			}
			IndexA++;
		}
	}

	sign = (i + j) % 2 == 0 ? 1 : -1;
	CofactorValue = (float)sign*Determinant(n-1, MinorB);
	free(MinorB);
	return (CofactorValue);
}

void MULTIVIS_RENDER::InverseMatrix(float* A, float* InverseA, int n)
{
	int row, column;
	double Det, Co;

	Det = Determinant(n, A);

	if (Det == 0)
		Det = 0.00001;
	if (Det == -0)
		Det = -0.00001;

	for (row = 0; row < n; row++)
	{
		for (column = 0; column < n; column++)
		{
			Co = Cofactor(A, n, column, row);

			InverseA[row*n + column] = Co / Det;
	
		}

	}

}

void MULTIVIS_RENDER::ComputeProperty(PointProperty* Point, int n)
{
	float4 Delta[NN];
	float Phi[NN];
	float PMatrix[NN*NU];
	float Alpha[NU];
	float B[NU];
	float AMatrix[NU*NU];
	float AInverse[NU*NU];

	float x0, y0, z0, t1, t2, sum, magnitude;
	int offset, dx, dy, dz, i, j, k, row, index;

	int count;

	int CellSize;
	CellSize = CUBESIZE / GRIDWIDTH;

	for (count = 0; count < n; count++)
	{
		x0 = floor(Point[count].x);
		y0 = floor(Point[count].y);
		z0 = floor(Point[count].z);

		offset = 0;

		while (offset < NN)
		{
			for (dz = 0; dz<2; dz++)
				for (dy = 0; dy<2; dy++)
					for (dx = 0; dx < 2; dx++)
					{
						Delta[offset].x = Point[count].x - (x0 + dx);
						Delta[offset].y = Point[count].y - (y0 + dy);
						Delta[offset].z = Point[count].z - (z0 + dz);

						t1 = Delta[offset].x * Delta[offset].x + Delta[offset].y * Delta[offset].y + Delta[offset].z * Delta[offset].z;
						t2 = (CUBESIZE / GRIDWIDTH)*(CUBESIZE / GRIDWIDTH);
                        Delta[offset].w = exp((-1)*t1 / t2);

						i = x0 / CellSize + dx + GRIDWIDTH / 2;
						j = y0 / CellSize + dy + GRIDHIGHT / 2;
						k = GRIDDEPTH / 2 - z0 / CellSize -dz;
						index = k * GRIDWIDTH * GRIDHIGHT + j * GRIDWIDTH + i;
						Phi[offset] = GridValue[index];

						offset++;
					}
		}
	
	

	for (i = 0; i < NN; i++)
	{
		row = i*NU;
		PMatrix[row] = 1;
		PMatrix[row + 1] = Delta[i].x;
		PMatrix[row + 2] = Delta[i].y;
		PMatrix[row + 3] = Delta[i].z;
		PMatrix[row + 4] = Delta[i].x * Delta[i].x;
		PMatrix[row + 5] = Delta[i].y * Delta[i].y;
		PMatrix[row + 6] = Delta[i].z * Delta[i].z;
		PMatrix[row + 7] = Delta[i].x * Delta[i].y;
		PMatrix[row + 8] = Delta[i].x * Delta[i].z;
		PMatrix[row + 9] = Delta[i].y * Delta[i].z;

	}

	for (k = 0; k < NU; k++)
	{
		for (j = 0; j < NU; j++)
		{
			sum = 0;
			for (i = 0; i < NN; i++)
			{
				sum += Delta[i].w * PMatrix[i*NU + k] * PMatrix[i*NU + j];

			}
			AMatrix[k*NU + j] = sum;	
		}

	}


	for (k = 0; k < NU; k++)
	{
		sum = 0;
		for (i = 0; i < NN; i++)
		{
			sum += Delta[i].w * PMatrix[i*NU + k] * Phi[i];

		}
		B[k] = sum;
	}

	InverseMatrix(AMatrix, AInverse, NU);

	for (i = 0; i < NU; i++)
	{
		sum = 0;
		for (j = 0; j < NU; j++)
		{
			sum += AInverse[i*NU + j] * B[j];
		}
		Alpha[i] = sum;

	}
	Point[count].Value = Alpha[0];

	magnitude = sqrt(Alpha[1] * Alpha[1] + Alpha[2] * Alpha[2] + Alpha[3] * Alpha[3]);
	Point[count].Nx = Alpha[1] / magnitude;
	Point[count].Ny = Alpha[2] / magnitude;
	Point[count].Nz = Alpha[3] / magnitude;
	
	}

}

float MULTIVIS_RENDER::BSplineValue(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float  x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;
	
	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);
	
	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;
	
    Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;
		
				Value = gridValue[offset];
				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;
								
			}

			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;

		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;

	}

	return (Sz);
	
}

float4 MULTIVIS_RENDER::NormalComutation(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz, NormalMagnitude;
	float Nx, Ny, Nz;
	float4 NormalDirection;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (make_float4(0, 0, 0, 0));

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

    Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 -1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;
					
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
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;

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
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;

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

	return (NormalDirection);

}

float MULTIVIS_RENDER::XX_2orderDerivative(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;
	

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;
				
				Value = gridValue[offset];
				if (Value == -1)
					Value = 0;
				Sx += Value*(6.0*Ux*BSplineMatrix[i] + 2.0*BSplineMatrix[4 + i]) / 6.0 / (CellSize * CellSize);
								
			}
		
			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;

		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;

	}

	return (Sz);

}

float MULTIVIS_RENDER::YY_2orderDerivative(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;
	
	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;
				
				Value = gridValue[offset];
				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;
				
			}

			Sy += Sx*(6.0*Uy*BSplineMatrix[j] + 2.0*BSplineMatrix[4 + j]) / 6.0 / (CellSize * CellSize);

		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;

	}

	return (Sz);

}

float MULTIVIS_RENDER::ZZ_2orderDerivative(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;

				Value = gridValue[offset];
				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;
				

			}

			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;

		}

		Sz += Sy*(6.0*Uz*BSplineMatrix[k] + 2.0*BSplineMatrix[4 + k]) / 6.0 / (CellSize * CellSize);

	}

	return (Sz);

}

float MULTIVIS_RENDER::XY_2orderDerivative(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;

				Value = gridValue[offset];
				Sx += Value*(3.0*Ux*Ux*BSplineMatrix[i] + 2.0*Ux*BSplineMatrix[4 + i] + BSplineMatrix[8 + i]) / 6.0 / CellSize;

			}

			Sy += Sx*(3.0*Uy*Uy*BSplineMatrix[j] + 2.0*Uy*BSplineMatrix[4 + j] + BSplineMatrix[8 + j]) / 6.0 / CellSize;

		}

		Sz += Sy*(Uz*Uz*Uz*BSplineMatrix[k] + Uz*Uz*BSplineMatrix[4 + k] + Uz*BSplineMatrix[8 + k] + BSplineMatrix[12 + k]) / 6.0;

	}

	return (Sz);

}

float MULTIVIS_RENDER::XZ_2orderDerivative(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;

				Value = gridValue[offset];
				Sx += Value*(3.0*Ux*Ux*BSplineMatrix[i] + 2.0*Ux*BSplineMatrix[4 + i] + BSplineMatrix[8 + i]) / 6.0 / CellSize;

			}

			Sy += Sx*(Uy*Uy*Uy*BSplineMatrix[j] + Uy*Uy*BSplineMatrix[4 + j] + Uy*BSplineMatrix[8 + j] + BSplineMatrix[12 + j]) / 6.0;

		}

		Sz += Sy*(3.0*Uz*Uz*BSplineMatrix[k] + 2.0*Uz*BSplineMatrix[4 + k] + BSplineMatrix[8 + k]) / 6.0 / CellSize;

	}

	return (Sz);

}

float MULTIVIS_RENDER::YZ_2orderDerivative(float x, float y, float z, float* gridValue)
{
	int i, j, k, offset;
	int iIndex, jIndex, kIndex;
	float x0, y0, z0;
	double Sx, Sy, Sz, Value;
	double Ux, Uy, Uz;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	x0 = floor(x / CellSize) * CellSize;
	y0 = floor(y / CellSize) * CellSize;
	z0 = floor(z / CellSize) * CellSize;

	Ux = (x - x0) / CellSize;
	Uy = (y - y0) / CellSize;
	Uz = (z - z0) / CellSize;

	Sz = 0;
	for (k = 0; k < 4; k++)
	{
		Sy = 0;
		for (j = 0; j < 4; j++)
		{
			Sx = 0;
			for (i = 0; i < 4; i++)
			{
				iIndex = x0 / CellSize + i + GRIDWIDTH / 2 - 1;
				jIndex = y0 / CellSize + j + GRIDHIGHT / 2 - 1;
				kIndex = GRIDDEPTH / 2 - z0 / CellSize - k + 1;
				offset = kIndex * GRIDWIDTH * GRIDHIGHT + jIndex * GRIDWIDTH + iIndex;

				Value = gridValue[offset];
				Sx += Value*(Ux*Ux*Ux*BSplineMatrix[i] + Ux*Ux*BSplineMatrix[4 + i] + Ux*BSplineMatrix[8 + i] + BSplineMatrix[12 + i]) / 6.0;
				

			}

			Sy += Sx*(3.0*Uy*Uy*BSplineMatrix[j] + 2.0*Uy*BSplineMatrix[4 + j] + BSplineMatrix[8 + j]) / 6.0 / CellSize;

		}

		Sz += Sy*(3.0*Uz*Uz*BSplineMatrix[k] + 2.0*Uz*BSplineMatrix[4 + k] + BSplineMatrix[8 + k]) / 6.0 / CellSize;

	}

	return (Sz);
	
}

void MULTIVIS_RENDER::MatrixMultiplication(float* A, float* B, float* C, int n)
{
	int i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			C[i*n + j] = 0;
			for (k = 0; k < n; k++)
			{
				C[i*n + j] += A[i*n + k] * B[k*n + j];
			}
			
		}
	}
}

void MULTIVIS_RENDER::ScalarMatrixMultiplication(float* A, float* B, float R, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			B[i*n + j] = R*A[i*n + j];
			
		}
	}
}

float2 MULTIVIS_RENDER::CurvatureComputation(float x, float y, float z, float* gridValue)
{
	float PMatrix[9];
	float HessianMatrix[9];
	float GMatrix[9];
	float T1Matrix[9];
	float T2Matrix[9];
	float4 NormalVector;
	float Nx, Ny, Nz, Nmagnitude;
	float T, F, k1, k2;
	float2 result;

	NormalVector = NormalComutation(x, y, z, gridValue);

	Nx = NormalVector.x;
	Ny = NormalVector.y;
	Nz = NormalVector.z;
	Nmagnitude = NormalVector.w;

	PMatrix[0] = 1 - Nx*Nx;
	PMatrix[1] = (-1.0)*Nx*Ny;
	PMatrix[2] = (-1.0)*Nx*Nz;
	PMatrix[3] = (-1.0)*Ny*Nx;
	PMatrix[4] = 1 - Ny*Ny;
	PMatrix[5] = (-1.0)*Ny*Nz;
	PMatrix[6] = (-1.0)*Nz*Nx;
	PMatrix[7] = (-1.0)*Nz*Ny;
	PMatrix[8] = 1 - Nz*Nz;

	HessianMatrix[0] = XX_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[1] = XY_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[2] = XZ_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[3] = XY_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[4] = YY_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[5] = YZ_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[6] = XZ_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[7] = YZ_2orderDerivative(x, y, z, gridValue);
	HessianMatrix[8] = ZZ_2orderDerivative(x, y, z, gridValue);

	MatrixMultiplication(PMatrix, HessianMatrix, T1Matrix, 3);
	MatrixMultiplication(T1Matrix, PMatrix, T2Matrix, 3);
	ScalarMatrixMultiplication(T2Matrix, GMatrix, (1.0) / Nmagnitude, 3);

	T = GMatrix[0] + GMatrix[4] + GMatrix[8];
	F = sqrt(GMatrix[0]* GMatrix[0] + GMatrix[4]* GMatrix[4] + GMatrix[8]* GMatrix[8]);

	double Discriminant;

	Discriminant = 2 * F*F - T*T;

	if (Discriminant < 0)
		Discriminant *= -1.0;

	k1 = 0.5*(T + sqrt(Discriminant));
	k2 = 0.5*(T - sqrt(Discriminant));

	result = make_float2(k1, k2);

	return(result);

}

float2 MULTIVIS_RENDER::CurvatureAssessment(float x, float y, float z, float* gridValue)
{

	float4 NormalVector;
	float Nx, Ny, Nz, Nmagnitude;
	float Fx, Fy, Fz, Fxx, Fyy, Fzz, Fxy, Fxz, Fyz;
	float Kmax, Kmin, KH, KH1, KH2, KK, KK1, KK2;
	float2 result;

	NormalVector = NormalComutation(x, y, z, gridValue);

	Nx = NormalVector.x;
	Ny = NormalVector.y;
	Nz = NormalVector.z;
	Nmagnitude = NormalVector.w;

	Fx = Nx*Nmagnitude;
	Fy = Ny*Nmagnitude;
	Fz = Nz*Nmagnitude;

	Fxx = XX_2orderDerivative(x, y, z, gridValue);
	Fyy = YY_2orderDerivative(x, y, z, gridValue);
	Fzz = ZZ_2orderDerivative(x, y, z, gridValue);
	Fxy = XY_2orderDerivative(x, y, z, gridValue);
	Fxz = XZ_2orderDerivative(x, y, z, gridValue);
	Fyz = YZ_2orderDerivative(x, y, z, gridValue);

	KH1 = (Fx*Fx*(Fyy + Fzz) + Fy*Fy*(Fxx + Fzz) + Fz*Fz*(Fxx + Fyy)) / (2.0*Nmagnitude*Nmagnitude*Nmagnitude);
	KH2 = (Fx*Fy*Fxy + Fx*Fz*Fxz + Fy*Fz*Fyz) / (Nmagnitude*Nmagnitude*Nmagnitude);
	KH = KH1 - KH2;

	KK1 = 2 * (Fx*Fy*(Fxz*Fyz - Fxy*Fzz) + Fx*Fz*(Fxy*Fyz - Fxz*Fyy) + Fy*Fz*(Fxy*Fxz - Fyz*Fxx)) / (Nmagnitude*Nmagnitude);
	KK2 = (Fx*Fx*(Fyy*Fzz - Fyz*Fyz) + Fy*Fy*(Fxx*Fzz - Fxz*Fxz) + Fz*Fz*(Fxx*Fyy - Fxy*Fxy)) / (Nmagnitude*Nmagnitude);
	KK = KK1 + KK2;

	double D;
	D = KH * KH - KK;
	if (D < 0)
		D *= -1.0;
	Kmin = KH - sqrt(D);
	Kmax = KH + sqrt(D);

	result = make_float2(Kmax, Kmin);
	return(result);


}

void MULTIVIS_RENDER::Test()
{
	int i, j, k, h, offset, count, Temp, ExampleNumber = 5;

	float XX, YY, ZZ, XY, XZ, YZ, xx, yy, zz, xy, xz, yz, x, y, z, tt, MajorDistance;
	float VALUE, value, XZdistance, NormalX, NormalY, NormalZ, NormalMagnitude;
	float Kmax, Kmin, KK, KH, kmax, kmin, kk, kh, CosTheta, Kmax1, Kmin1, KK1, KH1;
	float4 NORMAL, normal;
	float2 PrincipalCurvature, P1;

	TorusGridGenerator(GridValue);
	RandomSamplesOnSurface(SamplePoints, cellSampleInfo, &PointsNumber);

	for (i = 0; i < ExampleNumber; i++)
	{
		x = SamplePoints[i].x;
		y = SamplePoints[i].y;
		z = SamplePoints[i].z;
		VALUE = BSplineValue(x, y, z, GridValue);
		MajorDistance = sqrt(x*x + z*z) - TORUSRADIUS;
		//Value = MajorDistance*MajorDistance + y*y;
		value = MajorDistance*MajorDistance + y*y;
		NORMAL = NormalComutation(x, y, z, GridValue);
		XZdistance = sqrt(x*x + z*z);

		NormalX = (XZdistance - TORUSRADIUS)*x / XZdistance / value;
		NormalY = y / value;
		NormalZ = (XZdistance - TORUSRADIUS)*z / XZdistance / value;

		NormalMagnitude = sqrt(NormalX*NormalX + NormalY*NormalY + NormalZ*NormalZ);

		normal.x = NormalX / NormalMagnitude;
		normal.y = NormalY / NormalMagnitude;
		normal.z = NormalZ / NormalMagnitude;
		
		printf("Position: x: %f, y: %f, z: %f\n\n", x, y, z);
		printf("Scalar Value and Normal:\n");
		printf("Analytical computation:     ");
		printf("Scalar Value: %f, NormalX: %f, NormalY: %f NormalZ: %f\n", value, normal.x, normal.y, normal.z);
		printf("Numerical computation:      ");
		printf("Scalar Value: %f, NormalX: %f, NormalY: %f NormalZ: %f\n\n", VALUE, NORMAL.x, NORMAL.y, NORMAL.z);
		printf("Second Order Derivatives:\n");
		printf("Analytical computation:     ");

		XX = XX_2orderDerivative(x, y, z, GridValue);
		YY = YY_2orderDerivative(x, y, z, GridValue);
		ZZ = ZZ_2orderDerivative(x, y, z, GridValue);
		XY = XY_2orderDerivative(x, y, z, GridValue);
		XZ = XZ_2orderDerivative(x, y, z, GridValue);
		YZ = YZ_2orderDerivative(x, y, z, GridValue);
		tt = sqrt(x*x + z*z);
		xx = 2 - 2 * TORUSRADIUS*z*z / (tt*tt*tt);
		yy = 2;
		zz = 2 - 2 * TORUSRADIUS*x*x / (tt*tt*tt);
		xy = 0;
		yz = 0;
		xz = 2 * x*z*TORUSRADIUS / (tt*tt*tt);
		printf("XX: %f, YY: %f, ZZ: %f, XY: %f, XZ: %f, YZ: %f\n", XX, YY, ZZ, XY, XZ, YZ);
		printf("Numerical computation:      ");
		printf("XX: %f, YY: %f, ZZ: %f, XY: %f, XZ: %f, YZ: %f\n\n", xx, yy, zz, xy, xz, yz);

		PrincipalCurvature = CurvatureAssessment(x, y, z, GridValue);
		Kmax = PrincipalCurvature.x;
		Kmin = PrincipalCurvature.y;
		KK = PrincipalCurvature.x * PrincipalCurvature.y;
		KH = (PrincipalCurvature.x + PrincipalCurvature.y) / 2.0;

		P1 = CurvatureComputation(x, y, z, GridValue);
		Kmax1 = P1.x;
		Kmin1 = P1.y;
		KK1 = P1.x * P1.y;
		KH1 = (P1.x + P1.y) / 2.0;
		
		CosTheta = (sqrt(x*x + z*z) - TORUSRADIUS) / IsoValue;
		kk = CosTheta / (value*(TORUSRADIUS + value*CosTheta));
		kh = (1.0)*(TORUSRADIUS + 2 * IsoValue*CosTheta) / (2 * IsoValue*(TORUSRADIUS + IsoValue*CosTheta));

		double D;
		D = kh * kh - kk;
		if (D < 0)
			D *= -1.0;
		kmin = kh - sqrt(D);
		kmax = kh + sqrt(D);
		printf("\n Curvature:\n");
		printf("Analytical computation:     ");
		printf("Gussian curvature: %f, Mean curvature: %f, Kmax: %f Kmin: %f\n", kk, kh, kmax, kmin);
		printf("Numerical computation (1):  ");
		printf("Gussian curvature: %f, Mean curvature: %f, Kmax: %f Kmin: %f\n", KK, KH, Kmax, Kmin);
		printf("Numerical computation (2):  ");
		printf("Gussian curvature: %f, Mean curvature: %f, Kmax: %f Kmin: %f\n\n\n\n\n", KK1, KH1, Kmax1, Kmin1);

	}
}

void MULTIVIS_RENDER::BsplinePropertyComputation(PointProperty* Property, float* gridValue, int n)
{
	int i;
	float x, y, z;
	
	float4 NormalVector;
	float2 PrincipalCurvature;

	for (i = 0; i < n; i++)
	{
		x = Property[i].x;
		y = Property[i].y;
		z = Property[i].z;

		Property[i].Value = BSplineValue(x, y, z, gridValue);

		NormalVector = NormalComutation(x, y, z, gridValue);
		Property[i].Nx = NormalVector.x;
		Property[i].Ny = NormalVector.y;
		Property[i].Nz = NormalVector.z;

		//PrincipalCurvature = CurvatureComputation(x, y, z, gridValue);
		PrincipalCurvature = CurvatureAssessment(x, y, z, gridValue);
		Property[i].Kmax = PrincipalCurvature.x;
		Property[i].Kmin = PrincipalCurvature.y;
		Property[i].KK = PrincipalCurvature.x * PrincipalCurvature.y;
		Property[i].KH = (PrincipalCurvature.x + PrincipalCurvature.y) / 2.0;


	}
}

float4 MULTIVIS_RENDER::BiSearch(float x0, float y0, float z0, float Nx, float Ny, float Nz, float t, int RecursiveLayer)
{
	float x1, y1, z1, xm, ym, zm; 
	float MajorDistance, Value, Errors;
	float4 result;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	x1 = x0 + t * CellSize * Nx;
	y1 = y0 + t * CellSize * Ny;
	z1 = z0 + t * CellSize * Nz;

	xm = (x0 + x1) / 2.0;
	ym = (y0 + y1) / 2.0;
	zm = (z0 + z1) / 2.0;
	
	Value = BSplineValue(xm, ym, zm, GridValue);

	//MajorDistance = sqrt(xm*xm + zm*zm) - MajorRadius;
	//Value = sqrt(MajorDistance*MajorDistance + ym*ym);

	Errors = Value - IsoValue;
	if (Errors < 0)
		Errors = (-1.0) * Errors;

	if (Errors < TOLERANCE || RecursiveLayer > MAXRECURSE)
	{
		result = make_float4(xm, ym, zm, 1);

		return (result);

	}
	else if (Value > IsoValue)
	{
		return BiSearch(x0, y0, z0, Nx, Ny, Nz, t / 2, RecursiveLayer + 1);

	}
	else
	{
		return BiSearch(xm, ym, zm, Nx, Ny, Nz, t / 2, RecursiveLayer + 1);
	}
}

float4 MULTIVIS_RENDER::BiSearchNoRecursion(float x0, float y0, float z0, float x1, float y1, float z1, float Nx, float Ny, float Nz, float* gridValue)
{
	float xm, ym, zm;
	float Value, Errors;
	float4 result;
	int i=0;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	xm = (x0 + x1) / 2.0;
	ym = (y0 + y1) / 2.0;
	zm = (z0 + z1) / 2.0;

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

		Value = BSplineValue(xm, ym, zm, gridValue);

		Errors = Value - IsoValue;
		if (Errors < 0)
			Errors = (-1.0) * Errors;
		i++;
	}

	result = make_float4(xm, ym, zm, 1);
	return (result);

	
}

float MULTIVIS_RENDER::PointsDistance(float x0, float y0, float z0, float x1, float y1, float z1)
{
	float x, y, z;
	x = x1 - x0;
	y = y1 - y0;
	z = z1 - z0;
	return(sqrt(x*x + y*y + z*z));
}

float MULTIVIS_RENDER::Eij(float Rij)
{
	float m;
	
	if (Rij > 0 && Rij <= SIGMA)
	{
		m = (Rij / SIGMA) * PI / 2.0;

		return(1.0 / tan(m) + m - PI / 2.0);
	
	}
	else return (0);
}

float MULTIVIS_RENDER::DerEij(float Rij)
{

	float m, result;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (Rij > 0 && Rij <= SIGMA)
	{
		m = (Rij / SIGMA) * PI / 2.0;
		result = ((PI / (2.0*SIGMA))*(1 - 1.0 / (sin(m)*sin(m)))) * CellSize;

		return(result);

	}
	else return (0);
}

float MULTIVIS_RENDER::D2Eij(float Rij)
{
	float m, result;

	if (Rij > 0 && Rij <= SIGMA)
	{
		float CellSize;
		CellSize = float(CUBESIZE) / float(GRIDWIDTH);
		
		m = (Rij / SIGMA) * PI / 2.0;
		result = (PI*PI / (2.0*SIGMA*SIGMA) * cos(m) / (sin(m)*sin(m)*sin(m))) * CellSize * CellSize;

		return(result);

	}
	else return (0);
}

float MULTIVIS_RENDER::Ei(float x, float y, float z, CellSample* SampleInfo, float4* Points, int count)
{
	int i, j, k, Di, Dj, Dk, NeighborRange, offset, cellIndex, sampleIndex, CellPointsNumber;
	float x0, y0, z0, Rij, ijEnergy, result=0;
	
	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (0);

	NeighborRange = ceil(SIGMA / CellSize);
	
	i = x / CellSize + GRIDWIDTH / 2;
	j = y / CellSize + GRIDHIGHT / 2;
	k = GRIDDEPTH / 2 - (ceil(z / CellSize));


	for (Dk = (-1)*(NeighborRange-1); Dk<NeighborRange; Dk++)
		for (Dj = (-1)*(NeighborRange-1); Dj<NeighborRange; Dj++)
			for (Di = (-1)*(NeighborRange-1); Di < NeighborRange; Di++)
			{
				cellIndex = (k-Dk) * GRIDWIDTH * GRIDHIGHT + (j+Dj) * GRIDWIDTH + i+Di;

				if ((i + Di) < 0 || (i + Di) >= GRIDWIDTH || (j + Dj) < 0 || (j + Dj) >= GRIDHIGHT || (k - Dk) < 0 || (k - Dk) >= GRIDDEPTH)
					continue;
				

				if (SampleInfo[cellIndex].Number != 0)
				{
					sampleIndex = SampleInfo[cellIndex].Serial;
					for (offset = 0; offset < SampleInfo[cellIndex].Number; offset++)
					{
						if ((sampleIndex + offset) != count)
						{
							x0 = Points[sampleIndex + offset].x;
							y0 = Points[sampleIndex + offset].y;
							z0 = Points[sampleIndex + offset].z;

						
							Rij = PointsDistance(x, y, z, x0, y0, z0);
					
							if (Rij > 0 && Rij <= SIGMA)
							{
								ijEnergy = Eij(Rij);
								result += ijEnergy;
								
							}
						}
						
					}
				}
			}
	
	result /= 2.0;
	return (result);
}

void MULTIVIS_RENDER::Di(float x0, float y0, float z0, CellSample* SampleInfo, float4* Points, float* result)
{
	int i, j, k, Di, Dj, Dk, NeighborRange, cellIndex, sampleIndex, offset;
	float x, y, z, distance, derEij;
	float Nij[3];
	//float DiVector[3];

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);


	memset(result, 0.0, sizeof(float) * 3);

	if (x0 >= DOMAINSIZE || x0 <= ((-1)*DOMAINSIZE) || y0 >= DOMAINSIZE || y0 <= ((-1)*DOMAINSIZE) || z0 >= DOMAINSIZE || z0 <= ((-1)*DOMAINSIZE))
		return;

	NeighborRange = ceil(SIGMA / CellSize);

	i = x0 / CellSize + GRIDWIDTH / 2;
	j = y0 / CellSize + GRIDHIGHT / 2;
	k = GRIDDEPTH / 2 - ceil(z0 / CellSize);

	for (Dk = (-1)*(NeighborRange - 1); Dk<NeighborRange; Dk++)
		for (Dj = (-1)*(NeighborRange - 1); Dj<NeighborRange; Dj++)
			for (Di = (-1)*(NeighborRange - 1); Di < NeighborRange; Di++)
			{
		
				if ((i + Di) < 0 || (i + Di) >= GRIDWIDTH || (j + Dj) < 0 || (j + Dj) >= GRIDHIGHT || (k - Dk) < 0 || (k - Dk) >= GRIDDEPTH)
					continue;
				
				
				cellIndex = (k - Dk) * GRIDWIDTH * GRIDHIGHT + (j + Dj) * GRIDWIDTH + i + Di;

	
				if (SampleInfo[cellIndex].Number != 0)
				{
					sampleIndex = SampleInfo[cellIndex].Serial;
					for (offset = 0; offset < SampleInfo[cellIndex].Number; offset++)
					{
						x = Points[sampleIndex + offset].x;
						y = Points[sampleIndex + offset].y;
						z = Points[sampleIndex + offset].z;

						distance = PointsDistance(x, y, z, x0, y0, z0);


							if (distance > 0 && distance <= SIGMA)
							{

								derEij = DerEij(distance);

								Nij[0] = (x0 - x) / distance;
								Nij[1] = (y0 - y) / distance;
								Nij[2] = (z0 - z) / distance;

								result[0] += derEij*Nij[0];
								result[1] += derEij*Nij[1];
								result[2] += derEij*Nij[2];


							}						
						
					}
				}
			}

}

void MULTIVIS_RENDER::Hi(float x0, float y0, float z0, CellSample* SampleInfo, float4* Points, float* result)
{
	int i, j, k, row, column, Di, Dj, Dk, NeighborRange, cellIndex, sampleIndex, offset;
	float x, y, z, distance, d2Eij;
	float Nij[3];
	float MatrixNij[9];

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);
	
	memset(result, 0.0, sizeof(float) * 9);

	if (x0 >= DOMAINSIZE || x0 <= ((-1)*DOMAINSIZE) || y0 >= DOMAINSIZE || y0 <= ((-1)*DOMAINSIZE) || z0 >= DOMAINSIZE || z0 <= ((-1)*DOMAINSIZE))
		return;

	NeighborRange = ceil(SIGMA / CellSize);

	i = x0 / CellSize + GRIDWIDTH / 2;
	j = y0 / CellSize + GRIDHIGHT / 2;
	k = GRIDDEPTH / 2 - ceil(z0 / CellSize);


	for (Dk = (-1)*(NeighborRange - 1); Dk<NeighborRange; Dk++)
		for (Dj = (-1)*(NeighborRange - 1); Dj<NeighborRange; Dj++)
			for (Di = (-1)*(NeighborRange - 1); Di < NeighborRange; Di++)
			{
				if ((i + Di) < 0 || (i + Di) >= GRIDWIDTH || (j + Dj) < 0 || (j + Dj) >= GRIDHIGHT || (k - Dk) < 0 || (k - Dk) >= GRIDDEPTH)
					continue;
				
				cellIndex = (k - Dk) * GRIDWIDTH * GRIDHIGHT + (j + Dj) * GRIDWIDTH + i + Di;
		
				if (SampleInfo[cellIndex].Number != 0)
				{
					sampleIndex = SampleInfo[cellIndex].Serial;
					for (offset = 0; offset < SampleInfo[cellIndex].Number; offset++)
					{
						x = Points[sampleIndex + offset].x;
						y = Points[sampleIndex + offset].y;
						z = Points[sampleIndex + offset].z;

						distance = PointsDistance(x, y, z, x0, y0, z0);

						if (distance > 0 && distance <= SIGMA)
						{

							d2Eij = D2Eij(distance);

							Nij[0] = (x0 - x) / distance;
							Nij[1] = (y0 - y) / distance;
							Nij[2] = (z0 - z) / distance;

							OuterProduct(Nij, MatrixNij, 3);

							for (row = 0; row < 3; row++)
							{
								for (column = 0; column < 3; column++)
								{
									result[row * 3 + column] += d2Eij*MatrixNij[row * 3 + column];
									
								}
							}
							

						}
						
					}
				}
			}

}

void MULTIVIS_RENDER::OuterProduct(float* V, float* M, int num)
{

	int i, j;

	for (i=0;i<num;i++)
		for (j = 0; j < num; j++)
		{
			M[i*num + j] = V[i] * V[j];

		}
}

void MULTIVIS_RENDER::MatrixAddition(float* Ma, float* Mb, float* Mc, int num)
{
	int i, j;

	for (i = 0; i<num; i++)
		for (j = 0; j < num; j++)
		{
			Mc[i*num + j] = Ma[i*num + j] + Mb[i*num + j];

		}
}

void MULTIVIS_RENDER::MatVecProduct(float* M, float* V, float* Vresult, int num)
{
	int i, j;

	for (i = 0; i < num; i++)
	{
		Vresult[i] = 0;
		for (j = 0; j < num; j++)
		{
			Vresult[i] += M[i*num + j] * V[j];
			
		}
	
	}
}

void MULTIVIS_RENDER::SurfaceAttraction(float4* Points, float* gridValue, int n)
{
	int count;

	float x0, y0, z0, dx, dy, dz, x1, y1, z1, Value, v0, v1;
	float x, y, z;
	float4 normal, SurfacePoint;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);


	for (count = 0; count < n; count++)
	{
		x = Points[count].x;
		y = Points[count].y;
		z = Points[count].z;
		Value = BSplineValue(x, y, z, gridValue);

		normal = NormalComutation(x, y, z, gridValue);
		dx = normal.x;
		dy = normal.y;
		dz = normal.z;
		
		if (Value < IsoValue)
		{
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
		
		SurfacePoint = BiSearchNoRecursion(x0, y0, z0, x1, y1, z1, dx, dy, dz, gridValue);

		float vv = BSplineValue(SurfacePoint.x, SurfacePoint.y, SurfacePoint.z, gridValue);

		SamplePoints[count] = SurfacePoint;

	}
}

float4 MULTIVIS_RENDER::PointAttraction(float x, float y, float z, float* gridValue)
{
	
	float x0, y0, z0, dx, dy, dz, x1, y1, z1, scalarValue, v0, v1;
	float4 SurfacePoint, Normal;

	float CellSize;
	CellSize = float(CUBESIZE) / float(GRIDWIDTH);

	if (x >= DOMAINSIZE || x <= ((-1)*DOMAINSIZE) || y >= DOMAINSIZE || y <= ((-1)*DOMAINSIZE) || z >= DOMAINSIZE || z <= ((-1)*DOMAINSIZE))
		return (make_float4(x, y, z, 1));

	scalarValue = BSplineValue(x, y, z, gridValue);
	Normal = NormalComutation(x, y, z, gridValue);

		dx = Normal.x;
		dy = Normal.y;
		dz = Normal.z;
	
		if (scalarValue < IsoValue)
		{
			x0 = x;
			y0 = y;
			z0 = z;

			x1 = x0 + 1.8 * CellSize * dx;
			y1 = y0 + 1.8 * CellSize * dy;
			z1 = z0 + 1.8 * CellSize * dz;

			v0 = scalarValue;
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
			x1 = x;
			y1 = y;
			z1 = z;

			x0 = x1 - 1.8 * CellSize * dx;
			y0 = y1 - 1.8 * CellSize * dy;
			z0 = z1 - 1.8 * CellSize * dz;

			v1 = scalarValue;
			v0 = BSplineValue(x0, y0, z0, gridValue);

			if (v0 <= IsoValue && v1 <= IsoValue || v0 >= IsoValue && v1 >= IsoValue)
			{
				x0 = x0 - 1.8 * CellSize * dx;
				y0 = y0 - 1.8 * CellSize * dy;
				z0 = z0 - 1.8 * CellSize * dz;
			}
			
		}


		SurfacePoint = BiSearchNoRecursion(x0, y0, z0, x1, y1, z1, dx, dy, dz, gridValue);

		return (SurfacePoint);
	
}

void MULTIVIS_RENDER::Redistribution(CellSample* SampleInfo, float4* samplePoints, int pointsNumber, float* gridValue)
{
	int i, j, k, sampleCount, VelocityValid = 1;
	float x0, y0, z0, x1, y1, z1;
	float SumLambda=0, AverageLambda, EnergyNew;
	float4 SurfacePoint;
	float DiVector[3];
	float Nij[3];
	float H[9];
	float Hnew[9];
	float InverseH[9];
	float Velocity[3];

	float V[3];
	float Gradient[3];
	float4 normal;
	float OP_Gradient[9];
	float CoefMatrix[9];
	float IMatrix[9];
	float innerProduct;
	
	EnergyProperty* EnergyArray;
	EnergyArray = (EnergyProperty*)malloc(sizeof(EnergyProperty)*pointsNumber);

	//Initialize EnergyArray, which contains the information of energy for each point, including energy, lambda, and a flag to indicate
	// if it's the first time that the new energy is greater than the old one through the loop
	for (i = 0; i < pointsNumber; i++)
	{
			
		EnergyArray[i].Energy = Ei(samplePoints[i].x, samplePoints[i].y, samplePoints[i].z, SampleInfo, samplePoints, i);
		EnergyArray[i].Lambda = 1.0;
		EnergyArray[i].flag = 0;
		
	}

	AverageLambda = 0;

	while (AverageLambda < LAMBDAMAX && VelocityValid)
	{
		//update each sample point
		for (sampleCount = 0; sampleCount < pointsNumber; sampleCount++)
		{
			x0 = samplePoints[sampleCount].x;
			y0 = samplePoints[sampleCount].y;
			z0 = samplePoints[sampleCount].z;
			
			//compute Di and Hi
			Di(x0, y0, z0, cellSampleInfo, samplePoints, DiVector);
			Hi(x0, y0, z0, cellSampleInfo, samplePoints, H);
			EnergyArray[sampleCount].Energy = Ei(x0, y0, z0, SampleInfo, samplePoints, sampleCount);

			VelocityValid = 1;

			do
			{
				//multiply the diagonal elements of Hi with 1+lambda
				for (i=0;i<3;i++)
					for (j = 0; j < 3; j++)
					{
						if (i != j)
						{
							Hnew[i * 3 + j] = H[i * 3 + j];
						}
						else
						{
							Hnew[i * 3 + j] = H[i * 3 + j] * (1 + EnergyArray[sampleCount].Lambda);
						}
					}

				//compute inverse matrix of new Hi
				InverseMatrix(Hnew, InverseH, 3);
			
				for (i = 0; i < 9; i++)
					InverseH[i] *= (-1.0);
				//multiply the inverse matrix of Hi with the original velocity DiVector, to get a new velocity
				MatVecProduct(InverseH, DiVector, V, 3);

				//compute the gradient of each point, normal is a normalized vector, and it's 'w' item is the original magnitude of the normal
				normal = NormalComutation(x0, y0, z0, gridValue);

				Gradient[0] = normal.x * normal.w;
				Gradient[1] = normal.y * normal.w;
				Gradient[2] = normal.z * normal.w;

				//compute the ourter product of the gradient
				OuterProduct(Gradient, OP_Gradient, 3);

				//compute the inner product of gradient
				innerProduct = Gradient[0] * Gradient[0] + Gradient[1] * Gradient[1] + Gradient[2] * Gradient[2];

				if (innerProduct == 0)
					break;

				//build a identity matrix
				for (i = 0; i<3; i++)
					for (j = 0; j < 3; j++)
					{
						if (i != j)
						{
							IMatrix[i * 3 + j] = 0;
						}
						else
						{
							IMatrix[i * 3 + j] = 1;
						}
					}
				//the identity matrix IMatrix substract ourterproduct divided by innerproduct, to get a new coefficient of the velocity
				for (i = 0; i<3; i++)
					for (j = 0; j < 3; j++)
					{
						
						CoefMatrix[i * 3 + j] = IMatrix[i * 3 + j] - OP_Gradient[i * 3 + j] / innerProduct;
						
					}
				//multiply the coefficient matrix with the last velocity, to get the correct velocity
				MatVecProduct(CoefMatrix, V, Velocity, 3);


				if (Velocity[0] == 0 && Velocity[1] == 0 && Velocity[2] == 0)
				{
					VelocityValid = 0;
					break;
				}
				// add the old position and the velocity, to get a new position
				x1 = x0 + Velocity[0];
				y1 = y0 + Velocity[1];
				z1 = z0 + Velocity[2];

				if (x1 >= DOMAINSIZE || x1 <= ((-1)*DOMAINSIZE) || y1 >= DOMAINSIZE || y1 <= ((-1)*DOMAINSIZE) || z1 >= DOMAINSIZE || z1 <= ((-1)*DOMAINSIZE))
				{
					VelocityValid = 0;
					break;

				}

				// attract the new position to the surface
				SurfacePoint = PointAttraction(x1, y1, z1, gridValue);

				x1 = SurfacePoint.x;
				y1 = SurfacePoint.y;
				z1 = SurfacePoint.z;

				//compute the new energy, to see if it's greater than the old energy
				EnergyNew = Ei(x1, y1, z1, SampleInfo, samplePoints, sampleCount);

				if (EnergyNew>EnergyArray[sampleCount].Energy)
					EnergyArray[sampleCount].Lambda *= 10;

			} while (EnergyNew >= EnergyArray[sampleCount].Energy  && VelocityValid && EnergyArray[sampleCount].Lambda < 1000000);

			if (VelocityValid)

			{
				samplePoints[sampleCount].x = x1;
				samplePoints[sampleCount].y = y1;
				samplePoints[sampleCount].z = z1;	

				EnergyArray[sampleCount].Energy = EnergyNew;

				if (EnergyArray[sampleCount].flag == 0)
				{
					EnergyArray[sampleCount].Lambda /= 10;
					EnergyArray[sampleCount].flag = 1;
					
				
				}
			}

		}
		for (i = 0; i < pointsNumber; i++)
		{
			SumLambda += log10(EnergyArray[i].Lambda);
		}
		
		AverageLambda = SumLambda / pointsNumber;

	}
	free(EnergyArray);
}

void MULTIVIS_RENDER::InitPlane()
{
	long time1;
	time1 = clock();

	PlaneGridGenerator(GridValue);

	RandomSamplesOnSurface(SamplePoints, cellSampleInfo, &PointsNumber);

	SurfaceAttraction(SamplePoints, GridValue, PointsNumber);

	Redistribution(cellSampleInfo, SamplePoints, PointsNumber, GridValue);

	printf("Computation time using CPU: %d\n", clock() - time1);

}

void MULTIVIS_RENDER::RenderPlaneSamples()
{
	glPointSize(3.0f);

	glBindBuffer(GL_ARRAY_BUFFER, SampleVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * PointsNumber, SamplePoints, GL_STATIC_DRAW);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glUseProgram(SampleRenderProg);

	locMVP = glGetUniformLocation(SampleRenderProg, "mvpMatrix");
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvpMatrix);

	glDrawArrays(GL_POINTS, 0, PointsNumber);

	glUseProgram(0);
}

void MULTIVIS_RENDER::InitSphere()
{
	int i;
	float4 PointNormal;

	long time1;
	time1 = clock();

	SphereGridGenerator(GridValue);

	Sphere(SphereVertices, &indexBuffer);

	RandomSamplesOnSurface(SamplePoints, cellSampleInfo, &PointsNumber);
	
	SurfaceAttraction(SamplePoints, GridValue, PointsNumber);

	Redistribution(cellSampleInfo, SamplePoints, PointsNumber, GridValue);
	

	for (i = 0; i < PointsNumber; i++)
	{
		PointNormal = NormalComutation(SamplePoints[i].x, SamplePoints[i].y, SamplePoints[i].z, GridValue);
		SampleNormals[i] = PointNormal;
	}
	
	printf("Computation time using CPU: %d\n", clock() - time1);
	
}

void MULTIVIS_RENDER::RenderSphereDisks()
{
	glEnable(GL_DEPTH_TEST);

	glBindBuffer(GL_ARRAY_BUFFER, SpherePosBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * (SPHERESEGMENTS + 1)*(SPHERESEGMENTS + 1), SphereVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glUseProgram(SphereRenderProg);

	locMVP = glGetUniformLocation(SphereRenderProg, "mvpMatrix");
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvpMatrix);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glDrawElements(GL_TRIANGLES, SPHERESEGMENTS * SPHERESEGMENTS * 6, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT, GL_FILL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		
	glUseProgram(0);
			
	glBindBuffer(GL_ARRAY_BUFFER, SampleVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * PointsNumber, SamplePoints, GL_STATIC_DRAW);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, SampleNormalBuffer);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_TEXCOORD1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * PointsNumber, SampleNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(GLT_ATTRIBUTE_TEXCOORD1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		
	glUseProgram(SampleDiskRenderProg);

	locMVP = glGetUniformLocation(SampleDiskRenderProg, "mvpMatrix");
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvpMatrix);

	glDrawArrays(GL_POINTS, 0, PointsNumber);

	glUseProgram(0);

}

void MULTIVIS_RENDER::InitTorus()
{
	int i;
	float4 PointNormal;

	long time1;
	time1 = clock();

	TorusGridGenerator(GridValue);

	Torus(SphereVertices, &indexBuffer);

	RandomSamplesOnSurface(SamplePoints, cellSampleInfo, &PointsNumber);

	SurfaceAttraction(SamplePoints, GridValue, PointsNumber);

	Redistribution(cellSampleInfo, SamplePoints, PointsNumber, GridValue);
	
	for (i = 0; i < PointsNumber; i++)
	{
		PointNormal = NormalComutation(SamplePoints[i].x, SamplePoints[i].y, SamplePoints[i].z, GridValue);
		SampleNormals[i] = PointNormal;
	}

	printf("Computation time using CPU: %d\n", clock() - time1);

}

void MULTIVIS_RENDER::RenderTorusDisks()
{
	glEnable(GL_DEPTH_TEST);

	glBindBuffer(GL_ARRAY_BUFFER, SpherePosBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * (SPHERESEGMENTS + 1)*(SPHERESEGMENTS + 1), SphereVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glUseProgram(SphereRenderProg);

	locMVP = glGetUniformLocation(SphereRenderProg, "mvpMatrix");
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvpMatrix);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glDrawElements(GL_TRIANGLES, SPHERESEGMENTS * SPHERESEGMENTS * 6, GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT, GL_FILL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glUseProgram(0);

	glBindBuffer(GL_ARRAY_BUFFER, SampleVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * PointsNumber, SamplePoints, GL_STATIC_DRAW);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, SampleNormalBuffer);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_TEXCOORD1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * PointsNumber, SampleNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(GLT_ATTRIBUTE_TEXCOORD1, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glUseProgram(SampleDiskRenderProg);

	locMVP = glGetUniformLocation(SampleDiskRenderProg, "mvpMatrix");
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvpMatrix);

	glDrawArrays(GL_POINTS, 0, PointsNumber);

	glUseProgram(0);

}

void MULTIVIS_RENDER::Cuda_SampleDisplay()
{
	SphereGridGenerator(GridValue);
	
	initCuda(GridValue, GRIDWIDTH, GRIDHIGHT, GRIDDEPTH, CUBESIZE, IsoValue);

	RandomSamplesOnSurface(SamplePoints, cellSampleInfo, &PointsNumber);

	size_t numbytes[2];
	cudaGraphicsMapResources(1, &cuda_sampleVertex_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_SamplePoints, &numbytes[0], cuda_sampleVertex_resource);

	/*cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	//cudaMalloc((void**)&d_SamplePointProperty, PointsNumber * sizeof(PointProperty));
	//cudaMemcpy(d_SamplePointProperty, SamplePointProperty, PointsNumber * sizeof(PointProperty), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_GridValue, GRIDWIDTH*GRIDHIGHT*GRIDDEPTH * sizeof(float));
	cudaMemcpy(d_GridValue, GridValue, GRIDWIDTH*GRIDHIGHT*GRIDDEPTH * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&d_SamplePoints, PointsNumber * sizeof(float4));
	cudaMemcpy(d_SamplePoints, SamplePoints, PointsNumber * sizeof(float4), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&d_PointsNumber, sizeof(int));
	cudaMemcpy(d_PointsNumber, &PointsNumber, sizeof(int), cudaMemcpyHostToDevice);

	Cuda_SurfaceAttraction(d_SamplePoints, d_GridValue, d_PointsNumber);

	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	float time = elapsedTime;
	printf("Computation time using CUDA: %f\n", time);*/

	cudaGraphicsUnmapResources(1, &cuda_sampleVertex_resource, 0);
	
	glBindBuffer(GL_ARRAY_BUFFER, SamplePosBuffer);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glUseProgram(SampleRenderProg);

	locMVP = glGetUniformLocation(SampleRenderProg, "mvpMatrix");
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvpMatrix);

	glPointSize(2.0f);

	glDrawArrays(GL_POINTS, 0, PointsNumber);
	glUseProgram(0);

	cudaFree(d_SamplePoints);
	cudaFree(d_GridValue);
	
}

void MULTIVIS_RENDER::MultiVisRendering()
{
	if (ObjectSelection == 0)
		RenderSphereDisks();
	if (ObjectSelection == 1)
		RenderTorusDisks();
	if (ObjectSelection == 2)
		RenderPlaneSamples();
	if (ObjectSelection == 3)
		Cuda_SampleDisplay();

}

void MULTIVIS_RENDER::cleanup()
{
	
	cudaGraphicsUnregisterResource(cuda_sampleVertex_resource);


}