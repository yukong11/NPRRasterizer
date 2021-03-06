// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#define DEPTHBUFFERSIZE 30
#define DEPTHPRECISION 1e8
#define zNear 0.01
#define zFar 100.0
#define DEPTHEPSILON 1e-3f

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"

struct cudaXBO{
  float* vbo;
  float* nbo;
  float* cbo;
  int* ibo;
  int vbosize;
  int nbosize;
  int cbosize;
  int ibosize;
};

struct fin{
	glm::vec2 p0;
	glm::vec2 p1;
	glm::vec2 p2;
	float r0;
	float r1;
	float r2;
	int LOD;
};

struct line{
  glm::vec3 pe0;
  glm::vec3 pe1;
  glm::vec3 p0;
  glm::vec3 p1;
  float width;
  glm::vec3 normal;
};

struct triangle {
  glm::vec3 pe0;
  glm::vec3 pe1;
  glm::vec3 pe2;
  glm::vec3 ne0;
  glm::vec3 ne1;
  glm::vec3 ne2;
  glm::vec3 p0;
  glm::vec3 p1;
  glm::vec3 p2;
  glm::vec3 c0;
  glm::vec3 c1;
  glm::vec3 c2;
  bool Edge01;
  bool Edge12;
  bool Edge02;
  float width;
};

struct fragment{
  glm::vec3 color;
  glm::vec3 normal;   //eye space
  glm::vec3 position;   //eye space
  int triangleID;
  float depth;  //clipped depth
  
};

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec4 multiplyMV4(cudaMat4 m, glm::vec4 v){
  glm::vec4 r(0,0,0,0);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  r.w =(m.w.x*v.x)+(m.w.y*v.y)+(m.w.z*v.z)+(m.w.w*v.w);
  return r;
}

__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  //float w=(m.w.x*v.x)+(m.w.y*v.y)+(m.w.z*v.z)+(m.w.w*v.w);
  //if(w!=0.0f)
	 //r/=w;
  return r;
}
__host__ __device__ glm::vec3 multiplyMV3(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  float w=(m.w.x*v.x)+(m.w.y*v.y)+(m.w.z*v.z)+(m.w.w*v.w);
  r*=1.0f/w;
  return r;
}
//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint){
  minpoint = glm::vec3(min(min(tri.p0.x, tri.p1.x),tri.p2.x), 
        min(min(tri.p0.y, tri.p1.y),tri.p2.y),
        min(min(tri.p0.z, tri.p1.z),tri.p2.z));
  maxpoint = glm::vec3(max(max(tri.p0.x, tri.p1.x),tri.p2.x), 
        max(max(tri.p0.y, tri.p1.y),tri.p2.y),
        max(max(tri.p0.z, tri.p1.z),tri.p2.z));
}

__host__ __device__ void getAABBForLine(line li, glm::vec3& minpoint, glm::vec3& maxpoint){
  minpoint = glm::vec3(min( li.p0.x,  li.p1.x), 
        min( li.p0.y,  li.p1.y),
        min( li.p0.z,  li.p1.z));
  maxpoint = glm::vec3(max( li.p0.x,  li.p1.x), 
        max( li.p0.y,  li.p1.y),
        max( li.p0.z,  li.p1.z));
}
//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri){
  return 0.5*((tri.p2.x - tri.p0.x)*(tri.p1.y - tri.p0.y) - (tri.p1.x - tri.p0.x)*(tri.p2.y - tri.p0.y));
}

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri){
  triangle baryTri;
  baryTri.p0 = glm::vec3(a,0); baryTri.p1 = glm::vec3(b,0); baryTri.p2 = glm::vec3(c,0);
  return calculateSignedArea(baryTri)/calculateSignedArea(tri);
}

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(triangle tri, glm::vec2 point){
  float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri.p0.x,tri.p0.y), point, glm::vec2(tri.p2.x,tri.p2.y), tri);
  float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.p0.x,tri.p0.y), glm::vec2(tri.p1.x,tri.p1.y), point, tri);
  float alpha = 1.0-beta-gamma;
  return glm::vec3(alpha,beta,gamma);
}

//LOOK: checks if a barycentric coordinate is within the boundaries of a triangle
__host__ __device__ bool isBarycentricCoordInBounds(glm::vec3 barycentricCoord){
   return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
          barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
          barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

//LOOK: for a given barycentric coordinate, return the corresponding z position on the triangle
__host__ __device__ float getZAtCoordinate(glm::vec3 barycentricCoord, triangle tri){
  return (barycentricCoord.x*tri.p0.z + barycentricCoord.y*tri.p1.z + barycentricCoord.z*tri.p2.z);
}

__host__ __device__ int calculateDepth(glm::vec3 barycentricCoord, triangle tri){
	//return (int)((pow(2.0,DEPTHBUFFERSIZE)-1.0)*((zFar*zNear/(zFar-zNear)/getZAtCoordinate(barycentricCoord,tri))+0.5*(zFar+zNear)/(zFar-zNear)+0.5));
	return (int)(10000*getZAtCoordinate(barycentricCoord,tri));
}

__host__ __device__ float getZfromDepth(int depth){
	//return (float)((zFar*zNear/(zFar-zNear))/((double)depth/(pow(2.0,DEPTHBUFFERSIZE)-1.0)-0.5-0.5*(zFar+zNear)/(zFar-zNear)));;
	return (float)(depth/10000);
}

__host__ __device__ glm::vec3  getColorAtCoordinate(glm::vec3 barycentricCoord, triangle tri){
	return (barycentricCoord.x*tri.c0 + barycentricCoord.y*tri.c1 + barycentricCoord.z*tri.c2);
}

__host__ __device__ glm::vec3 getTexture(unsigned char* texture,int width,int height,glm::vec3 uv){
	float x = 1.0-fmod(1.0f+uv.x,1.0f);
	float y = 1.0-fmod(1.0f+uv.y,1.0f);
	int row = (int)(x * width);
    int col = (int)(y * height);
	int index = (row + col*width);

	glm::vec3 result;
	result.x = (float)(texture[3*index]) / 255.0;
	result.y = (float)(texture[3*index + 1]) / 255.0;
	result.z = (float)(texture[3*index + 2]) / 255.0;

	return result;
}

__host__ __device__ glm::vec3  getPosInEyeSpaceAtCoordinate(glm::vec3 barycentricCoord, triangle tri){
	return (barycentricCoord.x*tri.pe0 + barycentricCoord.y*tri.pe1 + barycentricCoord.z*tri.pe2);
}
__host__ __device__ glm::vec3 getNormalInEyeSpace(glm::vec3 barycentricCoord,triangle tri){
  
	//glm::vec3 edge1=tri.pe1-tri.pe0;
	//glm::vec3 edge2=tri.pe2-tri.pe0;
	//glm::vec3 normal=glm::normalize(glm::cross(edge1,edge2));
	glm::vec3 normal =glm::normalize((barycentricCoord.x*tri.ne0 + barycentricCoord.y*tri.ne1 + barycentricCoord.z*tri.ne2));
	return normal;
}

__host__ __device__ glm::vec3 getNormalInEyeSpace(triangle tri){
  
	glm::vec3 edge1=tri.pe1-tri.pe0;
	glm::vec3 edge2=tri.pe2-tri.pe0;
	glm::vec3 normal=glm::normalize(glm::cross(edge1,edge2));
	return normal;
}
__host__ __device__ glm::vec3 getNormal(triangle tri){
  
	glm::vec3 edge1=tri.p1-tri.p0;
	glm::vec3 edge2=tri.p2-tri.p0;
	glm::vec3 normal=glm::normalize(glm::cross(edge1,edge2));
	return normal;
}
#endif