#pragma once
#include"math3d.h"
#include<glut.h>

#define PI 3.1416


#if !defined _GLFRAME_H__ 
#define      _GLFRAME_H__ 

class GLActorFrame
{
public:
	GLActorFrame();
	~GLActorFrame();
protected:
	M3DVector3f m_vLocation;
	M3DVector3f m_vUp;
	M3DVector3f m_vForward;
	M3DVector3f m_vRight;
	M3DMatrix44f m_viewMatrix;
	M3DMatrix44f m_projectionMatrix;
	M3DMatrix44f m_normalMatrix;
	M3DMatrix44f m_MVP;

public:
	inline void GetViewMatrix(M3DMatrix44f ViewMat)
	{
		memcpy(ViewMat, m_viewMatrix, sizeof(float) * 16);
	}
	inline void GetPosition(M3DVector3f CameraPos)
	{
		memcpy(CameraPos, m_vLocation, sizeof(float) * 3);
	}
	inline void GetInverViewMatrix(M3DMatrix44f InvViewMat)
	{
		m3dInvertMatrix44(InvViewMat, m_viewMatrix);
	}
	virtual void MoveForward(GLfloat dis);
	virtual void MoveLR(GLfloat dis);
	virtual void MoveUD(GLfloat dis);
	virtual void RotateLocalX(GLfloat angle);
	virtual void RotateLocalY(GLfloat angle);
	virtual void RotateLocalZ(GLfloat angle);
	void ResetModel();
	virtual void UpdateViewMatrix();
	virtual void ApplyActorTransform();
	void SetOrigin(M3DVector3f vPosition);
};

class  GLCameraFrame :public GLActorFrame
{
public:
	GLCameraFrame();
	~GLCameraFrame();
protected:
	GLdouble     m_fFOV;
	GLdouble     m_fAspect;
	GLdouble    m_fNearPlane;
	GLdouble    m_fFarPlane;
public:
	void UpdateViewMatrix();
	const M3DMatrix44f & GetViewMatrix() {
		M3DMatrix44f mv;
		glGetFloatv(GL_MODELVIEW_MATRIX, mv);
		return mv;
	}
	const M3DMatrix44f & GetProjectionMatrix() { return m_projectionMatrix; }
	M3DMatrix44f & GetNormalMatrix() {
		m3dExtractRotationMatrix33(m_normalMatrix, m_viewMatrix);
		return m_normalMatrix;
	}
	M3DMatrix44f & GetModelViewProjectionMatrix()
	{
		M3DMatrix44f mv;
		glGetFloatv(GL_MODELVIEW_MATRIX, mv);
		m3dMatrixMultiply44(m_MVP, m_projectionMatrix, mv);
		return m_MVP;
	}
	M3DMatrix44f & GetModelViewMatrix() {
		M3DMatrix44f mv;
		glGetFloatv(GL_MODELVIEW_MATRIX, mv);
		return mv;
	}
	M3DMatrix44f & returnModelViewMatrix() {
		return m_viewMatrix;
	}
	M3DMatrix44f & returnModelViewProjectionMatrix()
	{
		m3dMatrixMultiply44(m_MVP, m_projectionMatrix, m_viewMatrix);
		return m_MVP;
	}
	void ApplyActorTransform();
	void ApplyCameraTransform();
	void AdjustCamera();
	void ResetCamera();
	void SetProjectMatrix() {
		float xmin, xmax, ymin, ymax;       // Dimensions of near clipping plane
		float xFmin, xFmax, yFmin, yFmax;   // Dimensions of far clipping plane

											// Do the Math for the near clipping plane
		ymax = m_fNearPlane * float(tan(m_fFOV * M3D_PI / 360.0));
		ymin = -ymax;
		xmin = ymin * m_fAspect;
		xmax = -xmin;

		// Construct the projection matrix
		m3dLoadIdentity44(m_projectionMatrix);
		m_projectionMatrix[0] = (2.0f * m_fNearPlane) / (xmax - xmin);
		m_projectionMatrix[5] = (2.0f * m_fNearPlane) / (ymax - ymin);
		m_projectionMatrix[8] = (xmax + xmin) / (xmax - xmin);
		m_projectionMatrix[9] = (ymax + ymin) / (ymax - ymin);
		m_projectionMatrix[10] = -((m_fFarPlane + m_fNearPlane) / (m_fFarPlane - m_fNearPlane));
		m_projectionMatrix[11] = -1.0f;
		m_projectionMatrix[14] = -((2.0f * m_fFarPlane * m_fNearPlane) / (m_fFarPlane - m_fNearPlane));
		m_projectionMatrix[15] = 0.0f;

	}
	virtual void MoveForward(GLfloat dis);
	virtual void MoveLR(GLfloat dis);
	virtual void MoveUD(GLfloat dis);
	virtual void RotateLocalX(GLfloat angle);
	virtual void RotateLocalY(GLfloat angle);
	virtual void RotateLocalZ(GLfloat angle);
	void RotateAxis(M3DVector3f axis, GLfloat angle, M3DMatrix44f transMatrix);
	void SetBaseViewMatrix(M3DVector3f vUP, M3DVector3f vForward, M3DVector3f pos);
	void SetProjParameter(GLdouble fFov, GLdouble fAspect, GLdouble fNearPlane, GLdouble fFarPlane);
};




#endif 
