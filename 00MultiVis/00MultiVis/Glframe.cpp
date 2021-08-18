#include "GLframe.h"

GLActorFrame::GLActorFrame()
{
	m3dLoadIdentity44(m_viewMatrix);
	m3dLoadIdentity44(m_projectionMatrix);
	m_vLocation[0] = m_vLocation[1] = m_vLocation[2] = 0.0f;
	m_vUp[0] = 0.0f;
	m_vUp[1] = 1.0f;
	m_vUp[2] = 0.0f;
	m_vForward[0] = 0.0f;
	m_vForward[1] = 0.0f;
	m_vForward[2] = 1.0f;
	UpdateViewMatrix();
}
GLActorFrame::~GLActorFrame() {}
void GLActorFrame::UpdateViewMatrix()
{

	M3DVector3f yAxis, zAxis, xAxis;
	memcpy(zAxis, m_vForward, sizeof(float) * 3);
	m3dNormalizeVector3(zAxis);
	m3dCrossProduct3(xAxis, m_vUp, m_vForward);
	m3dNormalizeVector3(xAxis);
	m3dCrossProduct3(yAxis, zAxis, xAxis);
	m3dNormalizeVector3(yAxis);
	/*for(int i=0;i<3;i++)m_viewMatrix[4*i]=xAxis[i];
	m_viewMatrix[12]=m3dDotProduct3(m_vLocation,xAxis);
	for(int i=0;i<3;i++)m_viewMatrix[4*i+1]=yAxis[i];
	m_viewMatrix[13]=m3dDotProduct3(m_vLocation,yAxis);
	for(int i=0;i<3;i++)m_viewMatrix[4*i+2]=zAxis[i];
	m_viewMatrix[14]=m3dDotProduct3(m_vLocation,zAxis);
	m_viewMatrix[15]=1.0f;*/
#define M(row,col) m_viewMatrix[col*4+row]
	M(0, 0) = xAxis[0];
	M(0, 1) = xAxis[1];
	M(0, 2) = xAxis[2];
	M(0, 3) = m3dDotProduct3(m_vLocation, xAxis);
	M(1, 0) = yAxis[0];
	M(1, 1) = yAxis[1];
	M(1, 2) = yAxis[2];
	M(1, 3) = m3dDotProduct3(m_vLocation, yAxis);
	M(2, 0) = zAxis[0];
	M(2, 1) = zAxis[1];
	M(2, 2) = zAxis[2];
	M(2, 3) = m3dDotProduct3(m_vLocation, zAxis);
	M(3, 0) = 0.0;
	M(3, 1) = 0.0;
	M(3, 2) = 0.0;
	M(3, 3) = 1.0;
#undef  M

}
void GLActorFrame::MoveForward(GLfloat dis)
{
	M3DMatrix44f translationMatrix;
	m3dLoadIdentity44(translationMatrix);
	m3dTranslationMatrix44(translationMatrix, 0.0f, 0.0f, -dis);
	//m3dMatrixMultiply44(m_viewMatrix,m_viewMatrix,translationMatrix);

	glMultMatrixf(translationMatrix);
}
void GLActorFrame::MoveLR(GLfloat dis)
{
	M3DMatrix44f translationMatrix;
	m3dLoadIdentity44(translationMatrix);
	m3dTranslationMatrix44(translationMatrix, dis, 0.0f, 0.0f);
	//	m3dMatrixMultiply44(m_viewMatrix,m_viewMatrix,translationMatrix);
	glMultMatrixf(translationMatrix);
}
void GLActorFrame::MoveUD(GLfloat dis)
{
	M3DMatrix44f translationMatrix;
	m3dLoadIdentity44(translationMatrix);
	m3dTranslationMatrix44(translationMatrix, 0.0f, dis, 0.0f);
	//	m3dMatrixMultiply44(m_viewMatrix,m_viewMatrix,translationMatrix);
	glMultMatrixf(translationMatrix);
}
void GLActorFrame::RotateLocalX(GLfloat angle)
{
	M3DMatrix44f rotationMatrix, transformMatrix;
	m3dLoadIdentity44(rotationMatrix);
	m3dRotationMatrix44(rotationMatrix, (angle), 1.0f, 0.0f, 0.0f);
	glMultMatrixf(rotationMatrix);
}
void GLActorFrame::RotateLocalY(GLfloat angle)
{
	M3DMatrix44f rotationMatrix;
	m3dLoadIdentity44(rotationMatrix);
	m3dRotationMatrix44(rotationMatrix, (angle), 0.0f, 1.0f, 0.0f);
	glMultMatrixf(rotationMatrix);
}
void GLActorFrame::RotateLocalZ(GLfloat angle)
{
	M3DMatrix44f rotationMatrix;
	m3dLoadIdentity44(rotationMatrix);
	m3dRotationMatrix44(rotationMatrix, m3dRadToDeg(angle), 0.0f, 0.0f, 1.0f);
	glMultMatrixf(rotationMatrix);
}
void GLActorFrame::ResetModel()
{
	m_vLocation[0] = m_vLocation[1] = m_vLocation[2] = 0.0f;
	m_vUp[0] = 0.0f;
	m_vUp[1] = 1.0f;
	m_vUp[2] = 0.0f;
	m_vForward[0] = 0.0f;
	m_vForward[1] = 0.0f;
	m_vForward[2] = 1.0f;
	UpdateViewMatrix();
}
void GLActorFrame::ApplyActorTransform()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(m_viewMatrix);

}

void GLActorFrame::SetOrigin(M3DVector3f vPosition)
{
	memcpy(m_vLocation, vPosition, sizeof(float) * 3);
	m_viewMatrix[12] = -m3dDotProduct3(m_vLocation, m_vRight);
	m_viewMatrix[13] = -m3dDotProduct3(m_vLocation, m_vUp);
	m_viewMatrix[14] = -m3dDotProduct3(m_vLocation, m_vRight);
}


GLCameraFrame::GLCameraFrame()
{
	m_fFOV = 120.0f;
	m_fAspect = 800.0f / 600;
	m_fNearPlane = 0.1f;
	m_fFarPlane = 4096.0f;
}
GLCameraFrame::~GLCameraFrame()
{}
void GLCameraFrame::UpdateViewMatrix()
{
	m3dLoadIdentity44(m_viewMatrix);
	M3DVector3f yAxis, zAxis, xAxis, m_vUp1;
	memcpy(zAxis, m_vForward, sizeof(float) * 3);
	m3dNormalizeVector3(zAxis);
	m3dNormalizeVector3(m_vForward);
	float dot = m3dDotProduct3(m_vForward, m_vUp);

	if (dot == 1 || dot == -1) {
		M3DMatrix44f rotM;
		m3dLoadIdentity44(rotM);
		m3dRotationMatrix44(rotM, 20, 1.0, 1.0, 1.0);
		m3dTransformVector3(m_vUp1, m_vUp, rotM);
		memcpy(m_vUp, m_vUp1, sizeof(float) * 3);
	}
	m3dCrossProduct3(xAxis, m_vForward, m_vUp);
	m3dNormalizeVector3(xAxis);
	m3dCrossProduct3(yAxis, xAxis, zAxis);
	m3dNormalizeVector3(yAxis);
	M3DVector3f z = { 0,0,0 };
	m3dSubtractVectors3(zAxis, z, zAxis);
	for (int i = 0; i<3; i++)
	{
		m_viewMatrix[4 * i] = xAxis[i];
		m_viewMatrix[4 * i + 1] = yAxis[i];
		m_viewMatrix[4 * i + 2] = zAxis[i];
	}
	m_viewMatrix[12] = -m3dDotProduct3(m_vLocation, xAxis);
	m_viewMatrix[13] = -m3dDotProduct3(m_vLocation, yAxis);
	m_viewMatrix[14] = -m3dDotProduct3(m_vLocation, zAxis);

	memcpy(m_vRight, xAxis, sizeof(float) * 3);
	memcpy(m_vUp, yAxis, sizeof(float) * 3);
	memcpy(m_vForward, zAxis, sizeof(float) * 3);

}
void GLCameraFrame::MoveForward(GLfloat dis)
{
	for (int i = 0; i<3; i++)
		m_vLocation[i] += dis*m_vForward[i];

	m_viewMatrix[12] = -m3dDotProduct3(m_vRight, m_vLocation);
	m_viewMatrix[13] = -m3dDotProduct3(m_vUp, m_vLocation);
	m_viewMatrix[14] = -m3dDotProduct3(m_vForward, m_vLocation);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
	/*	M3DVector3f vAdd;
	memcpy(vAdd,m_vForward,sizeof(float)*3);
	vAdd[1]=1.0f;
	m3dNormalizeVector3(vAdd);
	float temp=m_vLocation[1];
	for(int i=0;i<3;i++)m_vLocation[i]+=vAdd[i]*dis;
	m_vLocation[1]=temp;
	m_viewMatrix[12]=-m3dDotProduct3(xAxis,m_vLocation);
	m_viewMatrix[13]=-m3dDotProduct3(m_vUp,m_vLocation);
	m_viewMatrix[14]=-m3dDotProduct3(m_vForward,m_vLocation);
	*/
}
void GLCameraFrame::MoveLR(GLfloat dis)
{
	for (int i = 0; i<3; i++)
		m_vLocation[i] += dis*m_vRight[i];
	m_viewMatrix[12] = -m3dDotProduct3(m_vRight, m_vLocation);
	m_viewMatrix[13] = -m3dDotProduct3(m_vUp, m_vLocation);
	m_viewMatrix[14] = -m3dDotProduct3(m_vForward, m_vLocation);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
}
void GLCameraFrame::MoveUD(GLfloat dis)
{
	for (int i = 0; i<3; i++)
		m_vLocation[i] += dis*m_vUp[i];
	m_viewMatrix[12] = -m3dDotProduct3(m_vRight, m_vLocation);
	m_viewMatrix[13] = -m3dDotProduct3(m_vUp, m_vLocation);
	m_viewMatrix[14] = -m3dDotProduct3(m_vForward, m_vLocation);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
}
void GLCameraFrame::RotateLocalX(GLfloat angle)
{
	M3DMatrix44f transMatrix;
	RotateAxis(m_vRight, angle, transMatrix);
	M3DVector3f v1, yAxis, zAxis;

	m3dTransformVector3(v1, m_vUp, transMatrix);
	m3dNormalizeVector3(v1);
	memcpy(yAxis, v1, sizeof(float) * 3);

	m3dCrossProduct3(zAxis, m_vRight, yAxis);
	for (int i = 0; i<3; i++)
	{
		m_viewMatrix[4 * i] = m_vRight[i];
		m_viewMatrix[4 * i + 1] = yAxis[i];
		m_viewMatrix[4 * i + 2] = zAxis[i];
	}
	m_viewMatrix[12] = -m3dDotProduct3(m_vLocation, m_vRight);
	m_viewMatrix[13] = -m3dDotProduct3(m_vLocation, yAxis);
	m_viewMatrix[14] = -m3dDotProduct3(m_vLocation, zAxis);

	memcpy(m_vUp, yAxis, sizeof(float) * 3);
	memcpy(m_vForward, zAxis, sizeof(float) * 3);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
}
void GLCameraFrame::RotateLocalY(GLfloat angle)
{
	M3DMatrix44f transMatrix;
	//for(int i=0;i<3;i++)cout<<" "<<m_vUp[i];
	RotateAxis(m_vUp, angle, transMatrix);
	M3DVector3f v1;

	M3DVector3f xAxis, zAxis;
	m3dTransformVector3(v1, m_vRight, transMatrix);
	m3dNormalizeVector3(v1);
	memcpy(xAxis, v1, sizeof(float) * 3);
	m3dCrossProduct3(zAxis, xAxis, m_vUp);
	for (int i = 0; i<3; i++)
	{
		m_viewMatrix[4 * i] = xAxis[i];
		m_viewMatrix[4 * i + 1] = m_vUp[i];
		m_viewMatrix[4 * i + 2] = zAxis[i];
	}
	m_viewMatrix[12] = -m3dDotProduct3(m_vLocation, xAxis);
	m_viewMatrix[13] = -m3dDotProduct3(m_vLocation, m_vUp);
	m_viewMatrix[14] = -m3dDotProduct3(m_vLocation, zAxis);

	memcpy(m_vRight, xAxis, sizeof(float) * 3);
	memcpy(m_vForward, zAxis, sizeof(float) * 3);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
}
void GLCameraFrame::RotateLocalZ(GLfloat angle)
{
	M3DMatrix44f transMatrix;
	RotateAxis(m_vForward, angle, transMatrix);
	M3DVector3f v1, xAxis, yAxis;
	m3dTransformVector3(v1, m_vUp, transMatrix);
	memcpy(yAxis, v1, sizeof(float) * 3);
	m3dNormalizeVector3(yAxis);
	m3dCrossProduct3(xAxis, yAxis, m_vForward);
	for (int i = 0; i<3; i++)
	{
		m_viewMatrix[4 * i] = xAxis[i];
		m_viewMatrix[4 * i + 1] = yAxis[i];
		m_viewMatrix[4 * i + 2] = m_vForward[i];
	}
	m_viewMatrix[12] = -m3dDotProduct3(m_vLocation, xAxis);
	m_viewMatrix[13] = -m3dDotProduct3(m_vLocation, yAxis);
	m_viewMatrix[14] = -m3dDotProduct3(m_vLocation, m_vForward);

	memcpy(m_vRight, xAxis, sizeof(float) * 3);
	memcpy(m_vUp, yAxis, sizeof(float) * 3);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
}
void GLCameraFrame::RotateAxis(M3DVector3f axis, GLfloat angle, M3DMatrix44f transMatrix)
{

	M3DVector3f Uz, Ux = { 1.0,0.0,0.0 }, Uy;
	memcpy(Uz, axis, sizeof(float) * 3);
	m3dNormalizeVector3(Uz);
	if (m3dDotProduct3(Ux, Uz) >= 0.95f)Ux[1] = 1.0f;
	m3dNormalizeVector3(Ux);
	m3dCrossProduct3(Uy, Ux, Uz);
	m3dCrossProduct3(Ux, Uy, Uz);
	M3DMatrix44f tM, tM1, rotationMatrix, InvtM;
	m3dLoadIdentity44(tM);
	m3dLoadIdentity44(tM1);
	for (int i = 0; i<3; i++)
	{
		tM[4 * i] = Ux[i];
		tM[4 * i + 1] = Uy[i];
		tM[4 * i + 2] = Uz[i];
	}
	tM[12] = -m3dDotProduct3(m_vLocation, Ux);
	tM[13] = -m3dDotProduct3(m_vLocation, Uy);
	tM[14] = -m3dDotProduct3(m_vLocation, Uz);
	m3dRotationMatrix44(rotationMatrix, angle, 0.0f, 0.0f, 1.0f);
	m3dMatrixMultiply44(tM1, tM, rotationMatrix);
	m3dInvertMatrix44(InvtM, tM);
	m3dMatrixMultiply44(transMatrix, tM1, InvtM);

}
void GLCameraFrame::ApplyActorTransform()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m_viewMatrix);
}
void GLCameraFrame::ApplyCameraTransform()
{
	ApplyActorTransform();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(120.0,800.0/600,0.1,4096.0);
	//	gluPerspective(m_fFOV,m_fAspect,m_fNearPlane,m_fFarPlane);
}
void GLCameraFrame::SetBaseViewMatrix(M3DVector3f vUP, M3DVector3f vForward, M3DVector3f pos)
{
	memcpy(m_vUp, vUP, sizeof(float) * 3);
	memcpy(m_vForward, vForward, sizeof(float) * 3);
	m3dSubtractVectors3(m_vForward, m_vForward, pos);
	memcpy(m_vLocation, pos, sizeof(float) * 3);
	UpdateViewMatrix();
}
void GLCameraFrame::SetProjParameter(GLdouble fFov, GLdouble fAspect, GLdouble fNearPlane, GLdouble fFarPlane)
{
	m_fFOV = fFov;
	m_fAspect = fAspect;
	m_fNearPlane = fNearPlane;
	m_fFarPlane = fFarPlane;
}