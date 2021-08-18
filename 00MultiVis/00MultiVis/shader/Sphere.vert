#version 430 core

in vec4 vVertex;
//in vec4 SampleVertex;
uniform mat4 mvpMatrix;
out vec4 Color;

void main()
{
   gl_Position = mvpMatrix*vVertex ;

   Color=vVertex;
   if (Color.x<0)
   Color.x*=-1;
  if (Color.y<0)
   Color.y*=-1;
   if (Color.z<0)
   Color.z*=-1;
   Color.w=1;
  // Color=vVertex+vec4(1, 0.5, 1, 0);

   /*Color.x=vVertex.x*0.5+0.5;
   Color.y=vVertex.y*0.5+0.5;
   Color.z=vVertex.z*0.5+0.5;*/
}