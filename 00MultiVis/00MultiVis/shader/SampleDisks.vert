#version 430 core

in vec4 vVertex;
in vec4 nNormal;
//in vec4 SampleVertex;
uniform mat4 mvpMatrix;
//out vec4 diskCenter;
out vec4 diskNormal;

void main()
{
   //gl_Position = mvpMatrix*vVertex ;
   gl_Position =vVertex ;
   //diskCenter=vVertex;
   diskNormal=nNormal;
}