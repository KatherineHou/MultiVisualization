#include"MultiVis_Viewer.h"

#include <stdio.h>
#include <glui.h>
#include"GLframe.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<tchar.h>


using namespace std;

M3DVector3f vUp = { 0,1,0 };
M3DVector3f vForward = { 0,0,0 };
M3DVector3f vLocation = { 0,4.1,4.1 }; // camera position for torus
//M3DVector3f vLocation = { 0,2.1,2.1 }; // camera position for sphere
//M3DVector3f vLocation = { 0,0,2.6 }; // camera position for plane

float rotateX = 0.0f, rotateY = 0.0f;
float translateX = 0.0f, translateY = 0.0f, translateZ = 0.0f;

int   main_window;
int   mouseX, mouseY;
int   mouseButton = 0;


GLCameraFrame *G_PCAMERA = new GLCameraFrame();
MULTIVIS_RENDER * MultiVis_renderer;

void myGlutIdle(void)
{
	/* According to the GLUT specification, the current window is
	undefined during an idle callback.  So we need to explicitly change
	it if necessary */
	if (glutGetWindow() != main_window)
		glutSetWindow(main_window);

	glutPostRedisplay();
}
void motion(int x, int y)
{
	float dx = (float)(x - mouseX);
	float dy = (float)(y - mouseY);
	if (mouseButton == 1) {
		rotateX += dy * 0.01f;
		rotateY += dx * 0.01f;
	}
	else if (mouseButton == 2) {
		translateX += dx * 0.02f;
		translateY -= dy * 0.02f;
	}
	else if (mouseButton == 4) {
		translateZ += dy * 0.02f;
	}
	mouseX = x;
	mouseY = y;
}
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouseButton |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouseButton = 0;
	}

	mouseX = x;
	mouseY = y;
	glutPostRedisplay();
}


void setupRC()
{
	float NearPlane, FarPlane;
	MultiVis_renderer = new MULTIVIS_RENDER(GRIDWIDTH, GRIDHIGHT, GRIDDEPTH);
	if (MultiVis_renderer->ObjectSelection == 0)
	{
		NearPlane = 0.5;
		FarPlane = 4.5;
	}
	if (MultiVis_renderer->ObjectSelection == 1)
	{
		NearPlane = 0.5;
		FarPlane = 8.5;
	}
	if (MultiVis_renderer->ObjectSelection == 2)
	{
		NearPlane = 1.0;
		FarPlane = 2.0;
	}
	if (MultiVis_renderer->ObjectSelection == 3)
	{
		NearPlane = 0.5;
		FarPlane = 4.5;
	}
	G_PCAMERA->SetBaseViewMatrix(vUp, vForward, vLocation);
	G_PCAMERA->SetProjParameter(90.0, 800 / 600, NearPlane, FarPlane);
	//G_PCAMERA->SetProjParameter(90.0, 800 / 600, 0.5, 4.5);
	//G_PCAMERA->SetProjParameter(90.0, 800 / 600, 1.0f, 2.0f);
	G_PCAMERA->SetProjectMatrix();

	//MultiVis_renderer = new MULTIVIS_RENDER(GRIDWIDTH, GRIDHIGHT, GRIDDEPTH);
}
void Move(int value)
{
	glutPostRedisplay();
	glutTimerFunc(33, Move, 0);
}
void MULTIVIS_renderer()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	GLActorFrame frame_LIC;


	G_PCAMERA->SetBaseViewMatrix(vUp, vForward, vLocation);
	G_PCAMERA->ApplyCameraTransform();

	frame_LIC.ApplyActorTransform();
	frame_LIC.MoveLR(translateX);
	frame_LIC.MoveUD(translateY);
	frame_LIC.MoveForward(translateZ);
	frame_LIC.RotateLocalX(rotateX);
	frame_LIC.RotateLocalY(rotateY);

	M3DMatrix44f m1;
	memcpy(m1, G_PCAMERA->GetModelViewProjectionMatrix(), sizeof(float) * 16);
	MultiVis_renderer->setProjectMatrix(m1);
	memcpy(m1, G_PCAMERA->GetViewMatrix(), sizeof(float) * 16);
	MultiVis_renderer->setModelViewMatrix(m1);
	
	MultiVis_renderer->MultiVisRendering();
	//MultiVis_renderer->SampleDisplay();
	//MultiVis_renderer->Cuda_SampleDisplay();

	glPopMatrix();

	glutSwapBuffers();
}


void ChangeSize(GLsizei w, GLsizei h)
{
	GLfloat aspectratio;
	if (h == 0)h = 1;
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	aspectratio = (GLfloat)w / (GLfloat)h;

	float NearPlane, FarPlane;
	if (MultiVis_renderer->ObjectSelection == 0)
	{
		NearPlane = 0.5;
		FarPlane = 4.5;
	}
	if (MultiVis_renderer->ObjectSelection == 1)
	{
		NearPlane = 0.5;
		FarPlane = 8.5;
	}
	if (MultiVis_renderer->ObjectSelection == 2)
	{
		NearPlane = 1.0;
		FarPlane = 2.0;
	}
	if (MultiVis_renderer->ObjectSelection == 3)
	{
		NearPlane = 0.5;
		FarPlane = 4.5;
	}
	gluPerspective(90.0f, aspectratio, NearPlane, FarPlane);
	//gluPerspective(90.0f, aspectratio, 0.5f, 4.5f);
	//gluPerspective(90.0f, aspectratio, 1.0f, 2.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
}
void shutDown()
{
	delete(G_PCAMERA);
	delete(MultiVis_renderer);
}

void clean()
{
	MultiVis_renderer->~MULTIVIS_RENDER();

	delete(MultiVis_renderer);

}
int _tmain(int argc, _TCHAR* argv[])
{
	glutInit(&argc, (char **)argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition(200, 200);
	main_window = glutCreateWindow("Sample Distribution");
	glewInit();
	setupRC();
	glutDisplayFunc(MULTIVIS_renderer);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(ChangeSize);
	glutTimerFunc(33, Move, 0);

	GLUI *glui = GLUI_Master.create_glui("GLUI");
	new GLUI_Checkbox(glui, "Parameters", &(MultiVis_renderer->Parameters));
	GLUI_Spinner *mir_spinner = new GLUI_Spinner(glui, "IsoValue:", &(MultiVis_renderer->IsoValue));
	mir_spinner->set_int_limits(0, 32);
	GLUI_Spinner *sm_spinner = new GLUI_Spinner(glui, "SampleNumber:", &(MultiVis_renderer->SampleNumber));
	sm_spinner->set_int_limits(0, 10000);
	GLUI_Master.set_glutIdleFunc(myGlutIdle);

	glutMainLoop();
	shutDown();
	MultiVis_renderer->cleanup();
	clean();
	return 1;
}