#version 430 core

in vec4 vVertex;
//in vec4 SampleVertex;
uniform mat4 mvpMatrix;

void main()
{
   gl_Position = mvpMatrix*vVertex ;
}