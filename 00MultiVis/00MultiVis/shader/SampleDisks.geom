#version 430 core

layout(points)in;
layout(triangle_strip)out;
layout(max_vertices = 60)out;

in vec4 diskNormal[];

uniform mat4 mvpMatrix;

void main(void)
{
int segment=20;
float radius=0.03;
float PI=3.1416;

vec4 diskVertex[20];
vec4 diskCenter;

int i;
float x, y, z;
mat4 transformMatrix;
float Nx, Ny, Nz, Nxz;

Nx=diskNormal[0].x;
Ny=diskNormal[0].y;
Nz=diskNormal[0].z;

Nxz=sqrt(Nx*Nx+Nz*Nz)+0.0001;

diskCenter=gl_in[0].gl_Position;

transformMatrix[0][0]=(-1.0)*Nz/Nxz;
transformMatrix[0][1]=Nx;
transformMatrix[0][2]=(-1.0)*Nx*Ny/Nxz;
transformMatrix[0][3]=diskCenter.x;
transformMatrix[1][0]=0;
transformMatrix[1][1]=Ny;
transformMatrix[1][2]=Nxz;
transformMatrix[1][3]=diskCenter.y;
transformMatrix[2][0]=Nx/Nxz;
transformMatrix[2][1]=Nz;
transformMatrix[2][2]=(-1.0)*Ny*Nz/Nxz;
transformMatrix[2][3]=diskCenter.z;
transformMatrix[3][0]=0;
transformMatrix[3][1]=0;
transformMatrix[3][2]=0;
transformMatrix[3][3]=1;

for (i=0;i<segment;i++)
{
   x=radius*cos((float(i)/float(segment)) * 2.0*PI);
   z=radius*sin((float(i)/float(segment)) * 2.0*PI);
   y=0;
  
 //diskVertex[i]=transformMatrix*vec4(x, y, z, 1);

diskVertex[i].x=transformMatrix[0][0]*x+transformMatrix[0][2]*z+diskCenter.x;
diskVertex[i].y=transformMatrix[1][0]*x+transformMatrix[1][2]*z+diskCenter.y;
diskVertex[i].z=transformMatrix[2][0]*x+transformMatrix[2][2]*z+diskCenter.z;
diskVertex[i].w=1;

}

for (i=0;i<(segment-1);i++)
{
       gl_Position = mvpMatrix * diskVertex[i];
      EmitVertex();
      gl_Position = mvpMatrix * diskVertex[i+1];
      EmitVertex();
      gl_Position = mvpMatrix * diskCenter;
      EmitVertex();
      EndPrimitive();

}
      gl_Position = mvpMatrix * diskVertex[i];
      EmitVertex();
      gl_Position = mvpMatrix * diskVertex[0];
      EmitVertex();
      gl_Position = mvpMatrix * diskCenter;
      EmitVertex();
      EndPrimitive();

}
