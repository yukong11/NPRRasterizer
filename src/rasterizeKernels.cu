// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include <thrust/random.h>
#include "glm/gtc/matrix_transform.hpp"
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "Scene.h"

glm::vec3* framebuffer;

fragment* depthbuffer;
float* device_vbo;
float* device_vbo_eyeCoord;
float* device_nbo;
float* device_nbo_eyeCoord;
float* device_cbo;
int* device_ibo;
triangle* primitives;

unsigned char* device_Texture;

bool *backfaceFlag;
bool* partialfill;
bool *newbackfaceFlag;
triangle* newprimitives;

glm::mat4 cameraMatrix;
glm::mat4 projectionMatrix;

int tileSize = 32;
dim3 threadsPerBlock;
dim3 fullBlocksPerGrid;
fragment* shadowMap=NULL;

vector<cudaXBO> cudaXBOList;
vector<Texture>device_Texture_List;


void cleanXBOs(){
	if(initXBOFlag){
	for(int i=0;i<cudaXBOList.size();i++){
		cudaFree(cudaXBOList[i].cbo);
		cudaFree(cudaXBOList[i].ibo);
		cudaFree(cudaXBOList[i].nbo);
		cudaFree(cudaXBOList[i].vbo);
	}
	cudaXBOList.clear();
	}
}

void setXBO( float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){
  cudaXBO xboData;

  xboData.ibo = NULL;
  cudaMalloc((void**)&xboData.ibo, ibosize*sizeof(int));
  cudaMemcpy( xboData.ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  xboData.vbo = NULL;
  cudaMalloc((void**)&xboData.vbo, vbosize*sizeof(float));
  cudaMemcpy( xboData.vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  
  xboData.nbo = NULL;
  cudaMalloc((void**)&xboData.nbo, nbosize*sizeof(float));
  cudaMemcpy( xboData.nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);


  xboData.cbo = NULL;
  cudaMalloc((void**)&xboData.cbo, cbosize*sizeof(float));
  cudaMemcpy( xboData.cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  xboData.vbosize=vbosize;
  xboData.nbosize=nbosize;
  xboData.cbosize=cbosize;
  xboData.ibosize=ibosize;
  cudaXBOList.push_back(xboData);


}
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 


void setUpCudaTexture(std::vector<Texture> textList){
	for(unsigned int i=0;i<textList.size();i++)
	{
	    unsigned char* textureElement;
		cudaMalloc((void**)&textureElement, 3*textList[i].width*textList[i].height*sizeof(unsigned char));
		cudaMemcpy( textureElement, textList[i].ptr, 3*textList[i].width*textList[i].height*sizeof(unsigned char), cudaMemcpyHostToDevice);
		Texture deviceData;
		deviceData=textList[i];
		deviceData.ptr=textureElement;
		device_Texture_List.push_back(deviceData);
	}
}

void cleanTexture(){
	for(unsigned int i=0;i<device_Texture_List.size();i++)
	{
		cudaFree(device_Texture_List[i].ptr);
	}
	device_Texture_List.clear();
}
//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment somevalue){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<resolution.x && y<resolution.y){
      buffer[index] = somevalue;
    }
}
__global__ void clearFlag(bool * boolArray, int Count,bool value){
	   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	   if(index<Count){
		   boolArray[index]=value;
	   }

}

__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
	  index=x + ((resolution.y-y) * resolution.x);
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shaderdevice_vbo, vbosize,ModelViewCudaMatrix,device_vbo_eyeCoord,ProjectionCudaMatrix
__global__ void vertexShadeKernel(float* invbo,float* vbo, int vbosize,cudaMat4 modelView, float* vbo_eyeSpace,cudaMat4 Projection,float* nbo,float* nbo_eyespace, cudaMat4 normalMatrix ){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  // vertice in ObjSpace
	  glm::vec4 vertice4=glm::vec4(invbo[3*index],invbo[3*index+1],invbo[3*index+2],1.0f);  
	  // vertice in EyeSpace
	  vertice4=multiplyMV4(modelView,vertice4);

	   glm::vec4 normal=glm::vec4(nbo[3*index],nbo[3*index+1],nbo[3*index+2],1.0f);  
	  // normal in EyeSpace
	  normal=multiplyMV4(normalMatrix,normal);

	  vbo_eyeSpace[3*index]=vertice4.x;
	  vbo_eyeSpace[3*index+1]=vertice4.y;
	  vbo_eyeSpace[3*index+2]=vertice4.z;

	  nbo_eyespace[3*index]=normal.x;
	  nbo_eyespace[3*index+1]=normal.y;
	  nbo_eyespace[3*index+2]=normal.z;

	  // vertice in ClippingSpace
	  vertice4=multiplyMV4(Projection,vertice4);
	  // vertice in NDC
	  //if((abs(vertice4.w)>1e-3))
	  vertice4*=1.0f/vertice4.w;
	  vbo[3*index]=vertice4.x;
	  vbo[3*index+1]=vertice4.y;
	  vbo[3*index+2]=vertice4.z;
	  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(glm::vec2 resolution,float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float* vbo_eyeSpace,float* nbo_eyeCoord,bool* backfaceflags,int textureID){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  triangle tr;
	  int VerticeIndice0=ibo[index*3];
	  int VerticeIndice1=ibo[index*3+1];
	  int VerticeIndice2=ibo[index*3+2];

	  tr.p0=glm::vec3(vbo[3*VerticeIndice0],vbo[3*VerticeIndice0+1],vbo[3*VerticeIndice0+2]);
	  tr.p1=glm::vec3(vbo[3*VerticeIndice1],vbo[3*VerticeIndice1+1],vbo[3*VerticeIndice1+2]);
	  tr.p2=glm::vec3(vbo[3*VerticeIndice2],vbo[3*VerticeIndice2+1],vbo[3*VerticeIndice2+2]);

	  tr.pe0=glm::vec3(vbo_eyeSpace[3*VerticeIndice0],vbo_eyeSpace[3*VerticeIndice0+1],vbo_eyeSpace[3*VerticeIndice0+2]);
	  tr.pe1=glm::vec3(vbo_eyeSpace[3*VerticeIndice1],vbo_eyeSpace[3*VerticeIndice1+1],vbo_eyeSpace[3*VerticeIndice1+2]);
	  tr.pe2=glm::vec3(vbo_eyeSpace[3*VerticeIndice2],vbo_eyeSpace[3*VerticeIndice2+1],vbo_eyeSpace[3*VerticeIndice2+2]);

	  tr.ne0=glm::vec3(nbo_eyeCoord[3*VerticeIndice0],nbo_eyeCoord[3*VerticeIndice0+1],nbo_eyeCoord[3*VerticeIndice0+2]);
	  tr.ne1=glm::vec3(nbo_eyeCoord[3*VerticeIndice1],nbo_eyeCoord[3*VerticeIndice1+1],nbo_eyeCoord[3*VerticeIndice1+2]);
	  tr.ne2=glm::vec3(nbo_eyeCoord[3*VerticeIndice2],nbo_eyeCoord[3*VerticeIndice2+1],nbo_eyeCoord[3*VerticeIndice2+2]);


	  /*tr.c0=glm::vec3(cbo[0],cbo[1],cbo[2]);
	  tr.c1=glm::vec3(cbo[3],cbo[4],cbo[5]);
	  tr.c2=glm::vec3(cbo[6],cbo[7],cbo[8]);*/
	  if(textureID==-1){
		  glm::vec3 color=glm::vec3(cbo[0],cbo[1],cbo[2]);
		  tr.c0=color;
		  tr.c1=color;
		  tr.c2=color;
	  
	  }else{
		  tr.c0=glm::vec3(cbo[3*VerticeIndice0],cbo[3*VerticeIndice0+1],cbo[3*VerticeIndice0+2]);
		  tr.c1=glm::vec3(cbo[3*VerticeIndice1],cbo[3*VerticeIndice1+1],cbo[3*VerticeIndice1+2]);
		  tr.c2=glm::vec3(cbo[3*VerticeIndice2],cbo[3*VerticeIndice2+1],cbo[3*VerticeIndice2+2]);
	  }
	  tr.Edge01=false;
	  tr.Edge12=false;
	  tr.Edge12=false;

	  //tr.Edge01=true;
	  //tr.Edge12=true;
	  //tr.Edge12=true;
	  tr.width=0.01f;

#ifdef BackFaceScaling
	  glm::vec3 normal=getNormalInEyeSpace(tr);
	  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	  if(glm::dot(normal,eyeDir)<0.0f){
	  tr.c0=glm::vec3(0,0,0);
	  tr.c1=glm::vec3(0,0,0);
	  tr.c2=glm::vec3(0,0,0);

#ifdef ImageSpaceScaling
	 glm::vec2 p0=glm::vec2(tr.p0.x,tr.p0.y);
	  glm::vec2 p1=glm::vec2(tr.p1.x,tr.p1.y);
	  glm::vec2 p2=glm::vec2(tr.p2.x,tr.p2.y);

	  glm::vec2 dis=glm::vec2(2.0f/resolution.x,2.0f/resolution.y)*2.0f;

	 /* glm::vec2 temp0=glm::normalize(glm::normalize(p0-p1)+glm::normalize(p0-p2));
	  glm::vec2 temp1=glm::normalize(glm::normalize(p1-p0)+glm::normalize(p1-p2));
	  glm::vec2 temp2=glm::normalize(glm::normalize(p2-p1)+glm::normalize(p2-p0));
*/

	  glm::vec2 temp0=glm::normalize((p0-p1)+(p0-p2));
	  glm::vec2 temp1=glm::normalize((p1-p0)+(p1-p2));
	  glm::vec2 temp2=glm::normalize((p2-p1)+(p2-p0));

	   tr.p0+=glm::vec3(temp0.x*dis.x,temp0.y*dis.y,0.0f);
	  tr.p1+=glm::vec3(temp1.x*dis.x,temp1.y*dis.y,0.0f);;
	  tr.p2+=glm::vec3(temp2.x*dis.x,temp2.y*dis.y,0.0f);
#endif
#ifdef EyeSpaceScaling
	  /*tr.p0+=glm::normalize(glm::normalize(tr.p0-tr.p1)+glm::normalize(tr.p0-tr.p2))*0.008f;
	  tr.p1+=glm::normalize(glm::normalize(tr.p1-tr.p0)+glm::normalize(tr.p1-tr.p2))*0.008f;
	  tr.p2+=glm::normalize(glm::normalize(tr.p2-tr.p1)+glm::normalize(tr.p2-tr.p0))*0.008f;*/

	 tr.p0+=glm::normalize((tr.p0-tr.p1)+(tr.p0-tr.p2))*0.008f;
	  tr.p1+=glm::normalize((tr.p1-tr.p0)+(tr.p1-tr.p2))*0.008f;
	  tr.p2+=glm::normalize((tr.p2-tr.p1)+(tr.p2-tr.p0))*0.008f;
#endif 	 
	  }

	 
	
#endif 
	 
	  
#ifdef BackFaceCulling
	  glm::vec3 normal=getNormalInEyeSpace(tr);
	  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	 if(glm::dot(eyeDir,normal)<0)
		backfaceflags[index]=true;
	 else
		backfaceflags[index]=false;
#endif

	  primitives[index]=tr;

  }
}

//pass through
__global__ void GeoPass(triangle* primitives, int primitivesCount, triangle* newPrimitives,bool* filledflag,cudaMat4 Projection){
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(index<primitivesCount){

		  triangle tr=primitives[index];
		  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	      glm::vec3 normal=getNormalInEyeSpace(tr);
		  if(glm::dot(eyeDir,normal)<0.0f){
			filledflag[index]=false;
			return;
		  }
		   newPrimitives[index]=tr;
		   filledflag[index]=true;
	 
	 }
}

// sign distance interpolation
__global__ void GeoSignEdge(triangle* primitives, int primitivesCount, triangle* newPrimitives,bool* filledflag,cudaMat4 Projection){
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(index<primitivesCount){

	
		  triangle tr=primitives[index];
		  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	      glm::vec3 normal=getNormalInEyeSpace(tr);

		 
		  float sign[3];
		  sign[0]=glm::dot(tr.ne0,glm::normalize(tr.pe0));
		  sign[1]=glm::dot(tr.ne1,glm::normalize(tr.pe1));
		  sign[2]=glm::dot(tr.ne2,glm::normalize(tr.pe2));
		  if(((sign[0]>0.0f)&&(sign[1]>0.0f)&&(sign[2]>0.0f))||((sign[0]<0.0f)&&(sign[1]<0.0f)&&(sign[2]<0.0f))){
				filledflag[index]=false;
				return;
			}
	
		else{

			  glm::vec3 a,b,c;
			  float width=0.1f;
			  if(sign[0]*sign[1]>0.0f){
				a = (abs(sign[0])*tr.pe2+abs(sign[2])*tr.pe0)/(abs(sign[0])+abs(sign[2]));
				b= (abs(sign[1])*tr.pe2+abs(sign[2])*tr.pe1)/(abs(sign[1])+abs(sign[2]));
				c=tr.pe2;
				width*=0.5f*(glm::length(tr.pe2-tr.pe0)+glm::length(tr.pe1-tr.pe2));
			  }else{
				if(sign[0]*sign[2]>0.0f){
					a = (abs(sign[2])*tr.pe1+abs(sign[1])*tr.pe2)/(abs(sign[1])+abs(sign[2]));
					b= (abs(sign[0])*tr.pe1+abs(sign[1])*tr.pe0)/(abs(sign[1])+abs(sign[0]));
					c=tr.pe1;
					width*=0.5f*(glm::length(tr.pe2-tr.pe1)+glm::length(tr.pe1-tr.pe0));
				}else{
					
					a = (abs(sign[1])*tr.pe0+abs(sign[0])*tr.pe1)/(abs(sign[0])+abs(sign[1]));
					b= (abs(sign[2])*tr.pe0+abs(sign[0])*tr.pe2)/(abs(sign[0])+abs(sign[2]));
					c=tr.pe0;
					width*=0.5f*(glm::length(tr.pe1-tr.pe0)+glm::length(tr.pe0-tr.pe2));
				 }
			  } 

			  //float dis=1.0df;
			  //a+=glm::vec3(0.0f,0.0f,dis);
			  //a1+=glm::vec3(0.0f,0.0f,dis);
			  //b+=glm::vec3(0.0f,0.0f,dis);
			  //b1+=glm::vec3(0.0f,0.0f,dis);

			  glm::vec3 clipppedA=multiplyMV3(Projection,glm::vec4(a,1.0f));
			  glm::vec3 clipppedB=multiplyMV3(Projection,glm::vec4(b,1.0f));
			  glm::vec3 clipppedC=multiplyMV3(Projection,glm::vec4(c,1.0f));


			  glm::vec3 zmove=glm::vec3(0.0,0.0,-0.2f);
			  triangle tr1;
			  tr1.pe0=c;
			  tr1.pe1=a;
			  tr1.pe2=b;
			  tr1.p0=clipppedC+zmove;
			  tr1.p1=clipppedB+zmove;
			  tr1.p2=clipppedA+zmove;
			  tr1.c0=glm::vec3(0.0f,0.0f,0.0f);
			  tr1.c1=glm::vec3(0.0f,0.0f,0.0f);
			  tr1.c2=glm::vec3(0.0f,0.0f,0.0f);
			  tr1.ne0=normal;
			  tr1.ne1=normal;
			  tr1.ne2=normal;
			  tr1.Edge01=false;
			  tr1.Edge02=false;
			  tr1.Edge12=true;
			  tr1.width=width;
		      newPrimitives[index]=tr1;
			  
	

			  filledflag[index]=true;
		  }
	 
	 }
}

__global__ void GeoFace(triangle* primitives, int primitivesCount, triangle* newPrimitives,bool* filledflag,cudaMat4 Projection){
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(index<primitivesCount){

		  triangle tr=primitives[index];
		  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	      glm::vec3 normal=getNormalInEyeSpace(tr);

		  if(glm::dot(eyeDir,normal)<0.0f){
			filledflag[index]=false;
			return;
		  }

		   glm::vec3 a,b,a1,b1;
		   float widtha=0.4f,widthb=0.4f;
		   float threshold=0.96f;
		   glm::vec3 dis=glm::vec3(0.0f,0.0f,-0.1f);
		   
		   triangle tr1=tr;
		   tr1.c0=glm::vec3(0.0f,0.0f,0.0f);
		   tr1.c1=glm::vec3(0.0f,0.0f,0.0f);
		   tr1.c2=glm::vec3(0.0f,0.0f,0.0f);
		   tr1.p0+=dis;
		   tr1.p1+=dis;
		   tr1.p2+=dis;
		   tr1.width =1.0f/3.0f*(glm::length(tr.pe0-tr.pe1)+glm::length(tr.pe0-tr.pe2)+glm::length(tr.pe2-tr.pe1));
		   if(glm::dot(normal,glm::normalize(tr.ne0+tr.ne1))<threshold)
			     tr1.Edge01=true;
		  
		   
		   if(glm::dot(normal,glm::normalize(tr.ne1+tr.ne2))<threshold)
			   tr1.Edge12 =true;
		  
		   
		   if(glm::dot(normal,glm::normalize(tr.ne0+tr.ne2))<threshold)
			   tr1.Edge02=true;
		   
		   if((!tr1.Edge01)&&(!tr1.Edge02)&&(!tr1.Edge12)){   
			  filledflag[index]=false;
			 
		   }else{
			 filledflag[index]=true;
			 newPrimitives[index]=tr1;
		   }
		   
		   
	 }
}


__global__ void GeoLine(triangle* primitives, int primitivesCount, line* newPrimitives,bool* filledflag,cudaMat4 Projection){
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(index<primitivesCount){

		  triangle tr=primitives[index];
		  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	      glm::vec3 normal=getNormalInEyeSpace(tr);

		  if(glm::dot(eyeDir,normal)<0.0f){
			filledflag[6*index]=false;
			filledflag[6*index+1]=false;
			filledflag[6*index+2]=false;
			filledflag[6*index+3]=false;
			filledflag[6*index+4]=false;
			filledflag[6*index+5]=false;
			return;
		  }

		  glm::vec3 root=(tr.pe0+tr.pe2+tr.pe1)*1.0f/3.0f;
		   glm::vec3 rootNormal=getNormalInEyeSpace(glm::vec3(0.5f,0.5f,0.5f),tr);
		   glm::vec3 lat=-glm::normalize(glm::cross(rootNormal,-root));
		   glm::vec3 newGeoNormal=glm::normalize(glm::cross(rootNormal,lat));
		
			float phi=0.9f;
		    float revphi=/*1.0f/0.618f*/3.0f;
			float cof=clamp(glm::length(tr.pe0-0.5f*(tr.pe1+tr.pe2)),0.0f,0.01f);

			glm::vec3 p0=root+cof*lat;
			glm::vec3 p1=root-cof*lat;
			
			glm::vec3 p2=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*phi;
			glm::vec3 p3=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*(phi-1.0f);
			
			glm::vec3 p4=(p2+p3)*0.5f+phi*revphi*cof*rootNormal;
			glm::vec3 p5=(p2+p3)*0.5f+phi*revphi*cof*rootNormal-cof*lat*2.0f*phi*phi;

			glm::vec3 p6=root-cof*lat+revphi*cof*rootNormal*(1.0f+phi+phi*phi);

			glm::vec3 cl0=multiplyMV3(Projection,glm::vec4(p0,1.0f));
		    glm::vec3 cl1=multiplyMV3(Projection,glm::vec4(p1,1.0f));
			
			glm::vec3 cl2=multiplyMV3(Projection,glm::vec4(p2,1.0f));
		    glm::vec3 cl3=multiplyMV3(Projection,glm::vec4(p3,1.0f));
			
			glm::vec3 cl4=multiplyMV3(Projection,glm::vec4(p4,1.0f));
		    glm::vec3 cl5=multiplyMV3(Projection,glm::vec4(p5,1.0f));
			
			glm::vec3 cl6=multiplyMV3(Projection,glm::vec4(p6,1.0f));

			float width=0.01f;

		    line l1;
			l1.pe0=p0;
			l1.pe1=p2;
			l1.p0=cl0;
			l1.p1=cl2;
			l1.width=0.1f;
			l1.normal=newGeoNormal;
			newPrimitives[6*index]=l1;

			line l2;
			l2.pe0=p2;
			l2.pe1=p4;
			l2.p0=cl2;
			l2.p1=cl4;
			l2.width=width;
			l2.normal=newGeoNormal;
			newPrimitives[6*index+1]=l2;

			line l3;
			l3.pe0=p4;
			l3.pe1=p6;
			l3.p0=cl4;
			l3.p1=cl6;
			l3.width=width;
			l3.normal=newGeoNormal;
			newPrimitives[6*index+2]=l3;

			line l4;
			l4.pe0=p6;
			l4.pe1=p5;
			l4.p0=cl6;
			l4.p1=cl5;
			l4.width=width;
			l4.normal=newGeoNormal;
			newPrimitives[6*index+3]=l4;

			line l5;
			l5.pe0=p5;
			l5.pe1=p3;
			l5.p0=cl5;
			l5.p1=cl3;
			l5.width=width;
			l5.normal=newGeoNormal;
			newPrimitives[6*index+4]=l5;

			line l6;
			l6.pe0=p3;
			l6.pe1=p1;
			l6.p0=cl3;
			l6.p1=cl1;
			l6.width=width;
			l6.normal=newGeoNormal;
			newPrimitives[6*index+5]=l6;


			filledflag[6*index]=true;
			filledflag[6*index+1]=false;
			filledflag[6*index+2]=false;
			filledflag[6*index+3]=false;
			filledflag[6*index+4]=false;
			filledflag[6*index+5]=false;
		  
	 
	 }
}

__global__ void GeoFin(triangle* primitives, int primitivesCount, triangle* newPrimitives,bool* filledflag,cudaMat4 Projection){
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(index<primitivesCount){

		 triangle tr=primitives[index];

		  if((glm::dot(-tr.pe0,tr.ne0)<0.0f)&&(glm::dot(-tr.pe1,tr.ne1)<0.0f)&&(glm::dot(-tr.pe2,tr.ne2)<0.0f)){
		 // if(glm::dot(eyeDir,normal)<0.0f){
			filledflag[5*index]=false;
			filledflag[5*index+1]=false;
			filledflag[5*index+2]=false;
			filledflag[5*index+3]=false;
		    filledflag[5*index+4]=false;
			return;
		  }
		   glm::vec3 root=(tr.pe0+tr.pe2+tr.pe1)*1.0f/3.0f;
		   glm::vec3 rootNormal=getNormalInEyeSpace(glm::vec3(0.5f,0.5f,0.5f),tr);
		   glm::vec3 lat=-glm::normalize(glm::cross(rootNormal,-root));
		   glm::vec3 newGeoNormal=glm::normalize(glm::cross(rootNormal,lat));
		
			float phi=0.9f;
		    float revphi=/*1.0f/0.618f*/3.0f;
			float cof=clamp(glm::length(tr.pe0-0.5f*(tr.pe1+tr.pe2)),0.0f,0.01f);

			glm::vec3 p0=root+cof*lat;
			glm::vec3 p1=root-cof*lat;
			
			glm::vec3 p2=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*phi;
			glm::vec3 p3=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*(phi-1.0f);
			
			glm::vec3 p4=(p2+p3)*0.5f+phi*revphi*cof*rootNormal;
			glm::vec3 p5=(p2+p3)*0.5f+phi*revphi*cof*rootNormal-cof*lat*2.0f*phi*phi;

			glm::vec3 p6=root-cof*lat+revphi*cof*rootNormal*(1.0f+phi+phi*phi);

			glm::vec3 cl0=multiplyMV3(Projection,glm::vec4(p0,1.0f));
		    glm::vec3 cl1=multiplyMV3(Projection,glm::vec4(p1,1.0f));
			
			glm::vec3 cl2=multiplyMV3(Projection,glm::vec4(p2,1.0f));
		    glm::vec3 cl3=multiplyMV3(Projection,glm::vec4(p3,1.0f));
			
			glm::vec3 cl4=multiplyMV3(Projection,glm::vec4(p4,1.0f));
		    glm::vec3 cl5=multiplyMV3(Projection,glm::vec4(p5,1.0f));
			
			glm::vec3 cl6=multiplyMV3(Projection,glm::vec4(p6,1.0f));

			glm::vec3 color=glm::vec3(0.9,0.9,0.9);
			//glm::vec3 color=glm::vec3(0.0,0.0,0.0);
			float width=0.001f;
			triangle tr1;
			tr1.pe0=p0;
			tr1.pe1=p2;
			tr1.pe2=p1;
			tr1.p0=cl0;
			tr1.p1=cl2;
			tr1.p2=cl1;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*index+0]=tr1;

			tr1.pe0=p1;
			tr1.pe1=p2;
			tr1.pe2=p3;
			tr1.p0=cl1;
			tr1.p1=cl2;
			tr1.p2=cl3;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=true;
			tr1.Edge01=false;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*index+1]=tr1;

			tr1.pe0=p2;
			tr1.pe1=p4;
			tr1.pe2=p3;
			tr1.p0=cl2;
			tr1.p1=cl4;
			tr1.p2=cl3;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*index+2]=tr1;
			
			tr1.pe0=p3;
			tr1.pe1=p4;
			tr1.pe2=p5;
			tr1.p0=cl3;
			tr1.p1=cl4;
			tr1.p2=cl5;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=true;
			tr1.Edge01=false;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*index+3]=tr1;
			
			tr1.pe0=p4;
			tr1.pe1=p6;
			tr1.pe2=p5;
			tr1.p0=cl4;
			tr1.p1=cl6;
			tr1.p2=cl5;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=true;
			tr1.width=width;
			newPrimitives[5*index+4]=tr1;
						

			filledflag[5*index]=true;
			filledflag[5*index+1]=true;
			filledflag[5*index+2]=true;
			filledflag[5*index+3]=true;
		    filledflag[5*index+4]=true;
	 
	 }
}




//TODO: Implement a rasterization method, such as scanline.
__device__ unsigned int lock = 0u;

__global__ void rasterizationKernel(triangle* primitivesList, int Count, fragment* depthbuffer, glm::vec2 resolution,bool* backfaceflags,bool* filledFlag,unsigned char* texture,int textureWidth,int textureHeight){
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<Count){
	 #ifdef BackFaceCulling
	  if(backfaceflags[index])return;
	 #endif

	  #ifdef EDGEDETECTION
	    if(filledFlag!=NULL){
	    if(!filledFlag[index])return;
	  }
	 #endif
    

	  triangle tr=primitivesList[index];
	  glm::vec3 minBoundary,maxBoundary;
	  getAABBForTriangle(tr,minBoundary,maxBoundary);

	  
	  if(minBoundary.x>1.0f)return;
	  if(maxBoundary.x<-1.0f)return;
	  if(minBoundary.y>1.0f)return;
	  if(maxBoundary.y<-1.0f)return;
     
	  double dx=2.0/resolution.x;
	  double dy=2.0/resolution.y;
	  int start_x,end_x;
	  int start_y,end_y;

      if(minBoundary.x<-1.0f)start_x=0;
	  else start_x=(int)((minBoundary.x+1.0f)*resolution.x/2.0)-2;
	  if(minBoundary.y<-1.0f)start_y=0;
	  else start_y=(int)((minBoundary.y+1.0f)*resolution.y/2.0)-2;

	  if(maxBoundary.x>1.0f)end_x=resolution.x-1;
	  else end_x=(int)((maxBoundary.x+1.0f)*resolution.x/2.0)+2;
	  if(maxBoundary.y>1.0f)end_y=resolution.y-1;
	  else end_y=(int)((maxBoundary.y+1.0f)*resolution.y/2.0)+2;
	 
	 for(int j=start_y;j<=end_y;++j)
	 {
		 for(int i=start_x;i<=end_x;++i)

		 {
           	float x_value=-1.0f+(float)(i+0.5f)*dx;
			float y_value=-1.0f+(float)(j+0.5f)*dy;

			glm::vec3 barycoord=calculateBarycentricCoordinate(tr,glm::vec2(x_value,y_value));
				if(!isBarycentricCoordInBounds(barycoord)){
					continue;
				}else{
					fragment fr;
					fr.normal=getNormalInEyeSpace(barycoord,tr);
					fr.color=getColorAtCoordinate(barycoord,tr);
					if(texture!=NULL){
						fr.color=getTexture(texture,textureWidth,textureHeight,fr.color);
					}
					fr.position=getPosInEyeSpaceAtCoordinate(barycoord,tr);
					fr.depth=getZAtCoordinate(barycoord,tr);
					fr.triangleID=index;
					int DepthBufferIndex= i + (j * resolution.x);
					bool hold = true;
					while (hold) 
					{
						if (atomicExch(&lock, 1u)==0u) 
						{
							fragment oldValue=depthbuffer[DepthBufferIndex];
								if(fr.depth<oldValue.depth){
									depthbuffer[DepthBufferIndex]=fr;
								}
							hold = false;
							atomicExch(&lock,0u);
						}
					} 
			  }

		  } //for x
		}//for y
	}//index if
}
__global__ void rasterizationKernel_wireframe(triangle* primitivesList, int Count, fragment* depthbuffer, glm::vec2 resolution,bool* backfaceflags,bool* filledFlag,unsigned char* texture,int textureWidth,int textureHeight){
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<Count){
	 #ifdef BackFaceCulling
	  if(backfaceflags[index])return;
	 #endif

	  #ifdef EDGEDETECTION
	    if(filledFlag!=NULL){
	    if(!filledFlag[index])return;
	  }
	 #endif
     

	  triangle tr=primitivesList[index];

	  float width=tr.width;
	  glm::vec3 minBoundary,maxBoundary;
	  getAABBForTriangle(tr,minBoundary,maxBoundary);

	  
	  if(minBoundary.x>1.0f)return;
	  if(maxBoundary.x<-1.0f)return;
	  if(minBoundary.y>1.0f)return;
	  if(maxBoundary.y<-1.0f)return;
     
	  double dx=2.0/resolution.x;
	  double dy=2.0/resolution.y;
	  int start_x,end_x;
	  int start_y,end_y;

      if(minBoundary.x<-1.0f)start_x=0;
	  else start_x=(int)((minBoundary.x+1.0f)*resolution.x/2.0)-2;
	  if(minBoundary.y<-1.0f)start_y=0;
	  else start_y=(int)((minBoundary.y+1.0f)*resolution.y/2.0)-2;

	  if(maxBoundary.x>1.0f)end_x=resolution.x-1;
	  else end_x=(int)((maxBoundary.x+1.0f)*resolution.x/2.0)+2;
	  if(maxBoundary.y>1.0f)end_y=resolution.y-1;
	  else end_y=(int)((maxBoundary.y+1.0f)*resolution.y/2.0)+2;
	 
	 for(int j=start_y;j<=end_y;++j)
	 {
		 for(int i=start_x;i<=end_x;++i)

		 {
           	float x_value=-1.0f+(float)(i+0.5f)*dx;
			float y_value=-1.0f+(float)(j+0.5f)*dy;

			glm::vec3 barycoord=calculateBarycentricCoordinate(tr,glm::vec2(x_value,y_value));
				if(!isBarycentricCoordInBounds(barycoord)){
					continue;
				}else{
					glm::vec3 posInEYE=getPosInEyeSpaceAtCoordinate(barycoord,tr);
					bool discard=true;
					if(tr.Edge01){
						float distance=glm::length(glm::cross((tr.pe1-tr.pe0),(tr.pe0-posInEYE)))/glm::length(tr.pe1-tr.pe0);
						if(distance<width)
							discard=false;
					}
					if(tr.Edge02){
						float distance=glm::length(glm::cross((tr.pe2-tr.pe0),(tr.pe0-posInEYE)))/glm::length(tr.pe2-tr.pe0);
						if(distance<width)
							discard=false;
					}
					if(tr.Edge12){
						float distance=glm::length(glm::cross((tr.pe2-tr.pe1),(tr.pe1-posInEYE)))/glm::length(tr.pe2-tr.pe1);
						if(distance<width)
							discard=false;
					}
					if(discard)continue;
					fragment fr;
					fr.normal=getNormalInEyeSpace(barycoord,tr);
					fr.color=getColorAtCoordinate(barycoord,tr);
					if(texture!=NULL){
						fr.color=getTexture(texture,textureWidth,textureHeight,fr.color);
					}
					fr.position=getPosInEyeSpaceAtCoordinate(barycoord,tr);
					fr.depth=getZAtCoordinate(barycoord,tr);
					fr.triangleID=-1;
					int DepthBufferIndex= i + (j * resolution.x);
					bool hold = true;
					while (hold) 
					{
						if (atomicExch(&lock, 1u)==0u) 
						{
							fragment oldValue=depthbuffer[DepthBufferIndex];
								if(fr.depth<oldValue.depth){
									depthbuffer[DepthBufferIndex]=fr;
								}
							hold = false;
							atomicExch(&lock,0u);
						}
					} 
			  }

		  } //for x
		}//for y
	}//index if
}

__global__ void LineRasterizationKernel(line* primitivesList, int Count, fragment* depthbuffer, glm::vec2 resolution,glm::vec3 defaultColor,bool* filledFlag){
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<Count){

	  #ifdef EDGEDETECTION
	    if(filledFlag!=NULL){
	    if(!filledFlag[index])return;
	  }
	 #endif
     

	  line ls=primitivesList[index];
	  float width=ls.width;
	  glm::vec3 minBoundary,maxBoundary;
	  getAABBForLine(ls,minBoundary,maxBoundary);

	  
	  if(minBoundary.x>1.0f)return;
	  if(maxBoundary.x<-1.0f)return;
	  if(minBoundary.y>1.0f)return;
	  if(maxBoundary.y<-1.0f)return;
     
	  double dx=2.0/resolution.x;
	  double dy=2.0/resolution.y;
	  int start_x,end_x;
	  int start_y,end_y;

      if(minBoundary.x<-1.0f)start_x=0;
	  else start_x=(int)((minBoundary.x+1.0f)*resolution.x/2.0);

	  if(maxBoundary.x>1.0f)end_x=resolution.x-1;
	  else end_x=(int)((maxBoundary.x+1.0f)*resolution.x/2.0);

	 
	  for(int i=start_x;i<=end_x;++i)
		 {
           	float x_value=-1.0f+(float)(i+0.5f)*dx;

			if((x_value-ls.p0.x)/(ls.p1.x-ls.p0.x)<0.0f)
				continue;

			if((ls.p1.x-x_value)/(ls.p1.x-ls.p0.x)<0.0f)
				continue;
			glm::vec3 currentPosition= ls.p0*(x_value-ls.p0.x)/(ls.p1.x-ls.p0.x)+ls.p1*(ls.p1.x-x_value)/(ls.p1.x-ls.p0.x);

			if(currentPosition.y>1.0f)continue;
	        if(currentPosition.y<-1.0f)continue;

			bool discard=true;
			float distance=glm::length(glm::cross((ls.p1-ls.p0),(ls.p0-currentPosition)))/glm::length(ls.p1-ls.p0);
			if(distance<width)
					discard=false;
	        if(discard) continue;
			int y_index=(int)((currentPosition.y+1.0f)*resolution.y/2.0);
			

			fragment fr;
			fr.normal=ls.normal;
			fr.color=defaultColor;
			fr.position=ls.pe0*(x_value-ls.pe0.x)/(ls.pe1.x-ls.pe0.x)+ls.pe1*(ls.pe1.x-x_value)/(ls.pe1.x-ls.pe0.x);
			fr.depth=currentPosition.z;
			int DepthBufferIndex= i + (y_index * resolution.x);
			bool hold = true;
			while (hold) 
			{
				if (atomicExch(&lock, 1u)==0u) 
				{
					fragment oldValue=depthbuffer[DepthBufferIndex];
						if(fr.depth<oldValue.depth){
							depthbuffer[DepthBufferIndex]=fr;
						}
					hold = false;
					atomicExch(&lock,0u);
				}
			} 

	       
		}

	}//index if
}
__global__ void initialdepthflags(glm::vec2 resolution, bool* flagarray,bool value){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      flagarray[index] = value;
    }

}
//TODO: Implement a fragment shader


__global__ void fragmentShadeKernelPHONG(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 light,glm::vec3 lightColor){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<resolution.x && y<resolution.y){
	  //simple Phong Model
	  fragment fr=depthbuffer[index];
	  if(abs(fr.depth-FLT_MAX)<DEPTHEPSILON)return;
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 lightDirection=glm::normalize(light-IntersectionPosition);
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  float diffuse = clamp(glm::dot(normal, lightDirection), 0.0f,1.0f);
	  
	  glm::vec3 incidentDir=-lightDirection;
	  glm::vec3 reflectionedlightdirection=glm::normalize(incidentDir-2.0f*normal*glm::dot(incidentDir,normal));
	  float specular =clamp(pow(max(glm::dot(reflectionedlightdirection, viewdirection),0.0f), 20.0f), 0.0f,1.0f);

	  glm::vec3 newColor=glm::vec3(0,0,0);
	  newColor=0.1f*fr.color;
	 // newColor+=specular*lightColor;
	  newColor+=diffuse* glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z);
	  depthbuffer[index].color=newColor;

  }
}

__global__ void fragmentShadeKernelPHONGSHADOW(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 light,glm::vec3 lightColor, fragment* lightmap,cudaMat4 lightMatrix){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	/*  fragment fr=lightmap[index];
	  
	  int zvalue=getZfromDepth(fr.depth);
	  zvalue=0.5*zvalue+0.5;
	  fr.color=glm::vec3(zvalue,zvalue,zvalue);
	  depthbuffer[index]=fr;
*/
	//  /*
	  //simple Phong Model
	  fragment fr=depthbuffer[index];
	  if(abs(fr.depth-FLT_MAX)<DEPTHEPSILON)return;
	  glm::vec3 IntersectionPosition=fr.position; //eye space
	  glm::vec3 lightDirection=glm::normalize(light-IntersectionPosition);
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  float diffuse = clamp(glm::dot(normal, lightDirection), 0.0f,1.0f);
	  
	  glm::vec3 incidentDir=-lightDirection;
	  glm::vec3 reflectionedlightdirection=glm::normalize(incidentDir-2.0f*normal*glm::dot(incidentDir,normal));
	  float specular =clamp(pow(max(glm::dot(reflectionedlightdirection, viewdirection),0.0f), 20.0f), 0.0f,1.0f);

	  glm::vec3 newColor=glm::vec3(0,0,0);
	  newColor=0.1f*fr.color;
	  newColor+=specular*lightColor;
	  newColor+=diffuse* glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z);

	  glm::vec4 lightClip4=multiplyMV4(lightMatrix,glm::vec4(fr.position,1.0));
	  lightClip4*=1.0f/lightClip4.w;

	  if((lightClip4.x<-1.0f)||(lightClip4.x>1.0f)||(lightClip4.y<-1.0f)||(lightClip4.y>1.0f)){
		    newColor= 0.1f*newColor;
	  }else{
		int x_index=(int)((lightClip4.x+1.0f)*resolution.x/2.0);
		int y_index=(int)((lightClip4.y+1.0f)*resolution.y/2.0);
		int map_index = x_index + (y_index * resolution.x);
		if((x_index<=resolution.x )&& (y_index<=resolution.y)){
		   
		    fragment shadowfr=lightmap[index];
			int oldDepthvalue=shadowfr.depth;

			if(abs(oldDepthvalue-FLT_MAX)>DEPTHEPSILON){
	
			if((oldDepthvalue<(fr.depth))&&(abs(oldDepthvalue-(fr.depth))>DEPTHEPSILON))
				newColor=glm::vec3(1.0f,0.0f,0.0f);
			}
		}
	  }
	  depthbuffer[index].color=newColor;
//	  */
  }
}

__global__ void fragmentShadeKernelNPR(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 light,glm::vec3 lightColor,unsigned char* texture,int textureWidth,int textureHeight){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  //simple Phong Model
	  fragment fr=depthbuffer[index];
	  if(abs(fr.depth-FLT_MAX)<DEPTHEPSILON)return;
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 lightDirection=glm::normalize(light-IntersectionPosition);
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  float Ka=0.1f,Kd=1.0,Ks=0.3f,Kr=1.0f;
	  //------------------diffuse Term
	  float diffuse1 = clamp(glm::dot(normal, lightDirection), 0.0f,1.0f);
	  glm::vec3 diffuse=diffuse1* glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z); 
	  float basicLambert =glm::dot(normal, lightDirection);
	  float halfLambert=0.5f*basicLambert+0.5f;
	  float chl=clamp(halfLambert,0.01f,1.0f);
	  //Warp_Diffuse_ID
	  glm::vec3 warpColor=2.0f*getTexture(texture,textureWidth, textureHeight,glm::vec3(0.0,chl,0.0));
	  glm::vec3 diffuseColor=glm::vec3(0,0,0);
	  // newColor = warpColor;
	  //newColor=halfLambert*glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z);
	  //newColor=glm::vec3(lightColor.x*fr.color.x*warpColor.x,lightColor.y*fr.color.y*warpColor.y,lightColor.z*fr.color.z*warpColor.z);
	  diffuseColor = glm::vec3(lightColor.x*warpColor.x,lightColor.y*warpColor.y,lightColor.z*warpColor.z);
	  //diffuseColor+=glm::vec3(Ka,Ka,Ka);
	  diffuseColor=Kd*glm::vec3(fr.color.x*diffuseColor.x,fr.color.y*diffuseColor.y,fr.color.z*diffuseColor.z);


	  //----------specular and rim--------------//
	  glm::vec3 incidentDir=-lightDirection;
	  glm::vec3 reflectionedIncidentdirection=glm::normalize(incidentDir-2.0f*normal*glm::dot(incidentDir,normal)); //R
	  glm::vec3 reflectionedlightdirection=glm::normalize(lightDirection-2.0f*normal*glm::dot(lightDirection,normal)); //R2
      
	  float interP=clamp(glm::dot(normal,incidentDir),0.0f,1.0f);
	 // glm::vec3 fres=glm::vec3(1.0,1.0,1.0)*interP+ glm::vec3(0.95,0.6,0.6);
	  float fres= 1.0*(interP)+ 0.65*(1.0-interP);
	  float rimfres = pow(1.0f-glm::dot(normal,viewdirection),2.0f);
	  float specular =/*(1.0f-fres)**/clamp(pow(max(glm::dot(reflectionedIncidentdirection, viewdirection),0.0f), 10.0f), 0.01f,0.99f);
	  float rimspecular =rimfres*clamp(pow(max(glm::dot(reflectionedlightdirection, viewdirection),0.0f), 10.0f), 0.01f,0.99f);
	  
	  glm::vec3 mixedspecular=Ks*max(specular,rimspecular)* lightColor;

	  float upward = clamp(glm::dot(normal,glm::vec3(0,1,0)),0.0f,1.0f);
	  float uprim= upward*rimfres*Kr*(Ka*5.0f);

	  glm::vec3 newColor;
	  newColor=diffuseColor+mixedspecular+glm::vec3(uprim,uprim,uprim);
	  //newColor=diffuse;
	 // newColor=warpColor;
	  //newColor=glm::vec3(uprim,uprim,uprim);
	  //newColor=Ks*specular*lightColor;
	  //newColor=Ks*rimspecular*lightColor;
	  //newColor=glm::vec3(uprim,uprim,uprim);
	  depthbuffer[index].color=newColor;

  }
}


__global__ void fragmentShadeKernelEDGENPR(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 light,glm::vec3 lightColor,unsigned char* texture,int textureWidth,int textureHeight){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  //simple Phong Model
	  fragment fr=depthbuffer[index];
	  if(abs(fr.depth-FLT_MAX)<DEPTHEPSILON)return;
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 lightDirection=glm::normalize(light-IntersectionPosition);
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  float Ka=0.1f,Kd=1.0,Ks=0.3f,Kr=1.0f;
	  //------------------diffuse Term
	  float diffuse1 = clamp(glm::dot(normal, lightDirection), 0.0f,1.0f);
	  glm::vec3 diffuse=diffuse1* glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z); 
	  float basicLambert =glm::dot(normal, lightDirection);
	  float halfLambert=0.5f*basicLambert+0.5f;
	  float chl=clamp(halfLambert,0.01f,1.0f);
	  //Warp_Diffuse_ID
	  glm::vec3 warpColor=2.0f*getTexture(texture,textureWidth, textureHeight,glm::vec3(0.0,chl,0.0));
	  glm::vec3 diffuseColor=glm::vec3(0,0,0);
	  // newColor = warpColor;
	  //newColor=halfLambert*glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z);
	  //newColor=glm::vec3(lightColor.x*fr.color.x*warpColor.x,lightColor.y*fr.color.y*warpColor.y,lightColor.z*fr.color.z*warpColor.z);
	  diffuseColor = glm::vec3(lightColor.x*warpColor.x,lightColor.y*warpColor.y,lightColor.z*warpColor.z);
	  diffuseColor=Kd*glm::vec3(fr.color.x*diffuseColor.x,fr.color.y*diffuseColor.y,fr.color.z*diffuseColor.z);

	//----------specular and rim--------------//
	  glm::vec3 incidentDir=-lightDirection;
	  glm::vec3 reflectionedIncidentdirection=glm::normalize(incidentDir-2.0f*normal*glm::dot(incidentDir,normal)); //R
	  glm::vec3 reflectionedlightdirection=glm::normalize(lightDirection-2.0f*normal*glm::dot(lightDirection,normal)); //R2
      
	  float interP=clamp(glm::dot(normal,incidentDir),0.0f,1.0f);
	 // glm::vec3 fres=glm::vec3(1.0,1.0,1.0)*interP+ glm::vec3(0.95,0.6,0.6);
	  float fres= 1.0*(interP)+ 0.65*(1.0-interP);
	  float rimfres = pow(1.0f-glm::dot(normal,viewdirection),2.0f);
	  float specular =/*(1.0f-fres)**/clamp(pow(max(glm::dot(reflectionedIncidentdirection, viewdirection),0.0f), 10.0f), 0.01f,0.99f);
	  float rimspecular =rimfres*clamp(pow(max(glm::dot(reflectionedlightdirection, viewdirection),0.0f), 10.0f), 0.01f,0.99f);
	  
	  glm::vec3 mixedspecular=Ks*max(specular,rimspecular)* lightColor;

	  float upward = clamp(glm::dot(normal,glm::vec3(0,1,0)),0.0f,1.0f);
	  float uprim= upward*rimfres*Kr*(Ka*5.0f);

	  glm::vec3 newColor;
	  newColor=diffuseColor+mixedspecular+glm::vec3(uprim,uprim,uprim);
	  if((x>0)&&(x<resolution.x)&&(y>0)&&(y<resolution.y)){


		int A = (x-1) + ((y-1) * resolution.x);
		int H = (x+1) + ((y+1) * resolution.x);
		int F = (x-1) + ((y+1) * resolution.x);
		int C = (x+1) + ((y-1) * resolution.x);
		
		fragment Afrag=depthbuffer[A];
		fragment Hfrag=depthbuffer[H];
		fragment Ffrag=depthbuffer[F];
		fragment Cfrag=depthbuffer[C];

		float In=0.5f*(glm::dot(Afrag.normal,Hfrag.normal)+glm::dot(Cfrag.normal,Ffrag.normal));
		float Iz= (float)pow(1.0f-0.5f*(Afrag.depth-Hfrag.depth)*(Afrag.depth-Hfrag.depth),2.0f)*(float)pow(1.0f-0.5f*(Cfrag.depth-Ffrag.depth)*(Cfrag.depth-Ffrag.depth),2.0f);

		if(glm::dot(normal,viewdirection)<0.2f){
			newColor=newColor*(1.0f-Iz)+glm::vec3(0.0f,0.0f,0.0f)*Iz;
		}else{
			newColor=newColor*In+glm::vec3(0.0f,0.0f,0.0f)*(1.0f-In);
		    if(In<0.99f)
				newColor=glm::vec3(0,0,0);
		}
	
	  }

		
	  depthbuffer[index].color=newColor;

  }
}
__global__ void fragmentShadeKernelEDGEENHANCE(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 light,glm::vec3 lightColor){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  //simple Phong Model
	  fragment fr=depthbuffer[index];
	  if(abs(fr.depth-FLT_MAX)<DEPTHEPSILON)return;
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 lightDirection=glm::normalize(light-IntersectionPosition);
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  float diffuse = clamp(glm::dot(normal, lightDirection), 0.0f,1.0f);
	  
	  glm::vec3 incidentDir=-lightDirection;
	  glm::vec3 reflectionedlightdirection=glm::normalize(incidentDir-2.0f*normal*glm::dot(incidentDir,normal));
	  float specular =clamp(pow(max(glm::dot(reflectionedlightdirection, viewdirection),0.0f), 20.0f), 0.0f,1.0f);

	  glm::vec3 newColor=glm::vec3(0,0,0);
	  //newColor=0.1f*fr.color;
	  newColor+=diffuse* glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z);
	  //newColor=glm::vec3(1.0f,1.0f,1.0f);
	  newColor=fr.color;
	
	  if((x>0)&&(x<resolution.x)&&(y>0)&&(y<resolution.y)){


		int A = (x-1) + ((y-1) * resolution.x);
		int H = (x+1) + ((y+1) * resolution.x);
		int F = (x-1) + ((y+1) * resolution.x);
		int C = (x+1) + ((y-1) * resolution.x);
		
		fragment Afrag=depthbuffer[A];
		fragment Hfrag=depthbuffer[H];
		fragment Ffrag=depthbuffer[F];
		fragment Cfrag=depthbuffer[C];

		float In=0.5f*(glm::dot(Afrag.normal,Hfrag.normal)+glm::dot(Cfrag.normal,Ffrag.normal));
		float Iz= (float)pow(1.0f-0.5f*(Afrag.depth-Hfrag.depth)*(Afrag.depth-Hfrag.depth),2.0f)*(float)pow(1.0f-0.5f*(Cfrag.depth-Ffrag.depth)*(Cfrag.depth-Ffrag.depth),2.0f);

		if(glm::dot(normal,viewdirection)<0.2f){
			newColor=newColor*(1.0f-Iz)+glm::vec3(0.0f,0.0f,0.0f)*Iz;
		}else{
			newColor=newColor*In+glm::vec3(0.0f,0.0f,0.0f)*(1.0f-In);
		    if(In<0.995f)
				newColor=glm::vec3(0,0,0);
		}
	
	  }

		
	  depthbuffer[index].color=newColor;

  }
}
__global__ void setDenseValue(float* denseMap,fragment* depthbuffer, glm::vec2 resolution,float value){
 
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  fragment fr=depthbuffer[index];
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	 

	  if(glm::dot(normal,viewdirection)<0.2f){
			denseMap[index]=value;
		}else if((x>0)&&(x<resolution.x)&&(y>0)&&(y<resolution.y)){


		int A = (x-1) + ((y-1) * resolution.x);
		int H = (x+1) + ((y+1) * resolution.x);
		int F = (x-1) + ((y+1) * resolution.x);
		int C = (x+1) + ((y-1) * resolution.x);
		
		fragment Afrag=depthbuffer[A];
		fragment Hfrag=depthbuffer[H];
		fragment Ffrag=depthbuffer[F];
		fragment Cfrag=depthbuffer[C];

		float In=0.5f*(glm::dot(Afrag.normal,Hfrag.normal)+glm::dot(Cfrag.normal,Ffrag.normal));

		if(In<0.992f)
			denseMap[index]=value*(In);
		else 
			denseMap[index]=0.0f;	
	  }else{
		  denseMap[index]=0.0f;
		}
  }
}

__global__ void DenseGeoFin(triangle* primitives, int primitivesCount, triangle* newPrimitives,bool* filledflag,cudaMat4 Projection,fragment* depthbuffer, glm::vec2 resolution,float* denseMap, float dvalue){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<resolution.x && y<resolution.y){
	  float denseValue=denseMap[index];
	  if(denseValue<0.1f)return;
	   fragment fr=depthbuffer[index];
	   triangle tr=primitives[fr.triangleID];

		  if((glm::dot(-tr.pe0,tr.ne0)<0.0f)&&(glm::dot(-tr.pe1,tr.ne1)<0.0f)&&(glm::dot(-tr.pe2,tr.ne2)<0.0f)){
			return;
		  }
		   if(filledflag[5*fr.triangleID])return;
		   glm::vec3 root=(tr.pe0+tr.pe2+tr.pe1)*1.0f/3.0f;
		   glm::vec3 rootNormal=getNormalInEyeSpace(glm::vec3(0.5f,0.5f,0.5f),tr);
		   glm::vec3 lat=-glm::normalize(glm::cross(rootNormal,-root));
		   glm::vec3 newGeoNormal=glm::normalize(glm::cross(rootNormal,lat));
		
			float phi=0.9f;
		    float revphi=1.5f;
			float cof=clamp(glm::length(tr.pe0-0.5f*(tr.pe1+tr.pe2)),0.0f,0.01f);

			glm::vec3 p0=root+cof*lat;
			glm::vec3 p1=root-cof*lat;
			
			glm::vec3 p2=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*phi/2.0f;
			glm::vec3 p3=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*(phi/2.0f-1.0f);
			
			glm::vec3 p4=(p2+p3)*0.5f+phi*revphi*cof*rootNormal;
			glm::vec3 p5=(p2+p3)*0.5f+phi*revphi*cof*rootNormal-cof*lat*2.0f*phi*phi/2.0f;


			glm::vec3 p6=root-cof*lat+revphi*cof*rootNormal*(1.0f+phi+phi*phi);

			glm::vec3 cl0=multiplyMV3(Projection,glm::vec4(p0,1.0f));
		    glm::vec3 cl1=multiplyMV3(Projection,glm::vec4(p1,1.0f));
			
			glm::vec3 cl2=multiplyMV3(Projection,glm::vec4(p2,1.0f));
		    glm::vec3 cl3=multiplyMV3(Projection,glm::vec4(p3,1.0f));
			
			glm::vec3 cl4=multiplyMV3(Projection,glm::vec4(p4,1.0f));
		    glm::vec3 cl5=multiplyMV3(Projection,glm::vec4(p5,1.0f));
			
			glm::vec3 cl6=multiplyMV3(Projection,glm::vec4(p6,1.0f));

			glm::vec3 color=glm::vec3(0.9,0.9,0.9);
			float width=0.005f;
			triangle tr1;
			tr1.pe0=p0;
			tr1.pe1=p2;
			tr1.pe2=p1;
			tr1.p0=cl0;
			tr1.p1=cl2;
			tr1.p2=cl1;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+0]=tr1;
	
			tr1.pe0=p1;
			tr1.pe1=p2;
			tr1.pe2=p3;
			tr1.p0=cl1;
			tr1.p1=cl2;
			tr1.p2=cl3;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=true;
			tr1.Edge01=false;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+1]=tr1;

			tr1.pe0=p2;
			tr1.pe1=p4;
			tr1.pe2=p3;
			tr1.p0=cl2;
			tr1.p1=cl4;
			tr1.p2=cl3;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+2]=tr1;
			
			tr1.pe0=p3;
			tr1.pe1=p4;
			tr1.pe2=p5;
			tr1.p0=cl3;
			tr1.p1=cl4;
			tr1.p2=cl5;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=true;
			tr1.Edge01=false;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+3]=tr1;
			
			tr1.pe0=p4;
			tr1.pe1=p6;
			tr1.pe2=p5;
			tr1.p0=cl4;
			tr1.p1=cl6;
			tr1.p2=cl5;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=true;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+4]=tr1;
						

			

			filledflag[5*fr.triangleID]=true;
			filledflag[5*fr.triangleID+1]=true;
			filledflag[5*fr.triangleID+2]=true;
			filledflag[5*fr.triangleID+3]=true;
		    filledflag[5*fr.triangleID+4]=true;
		
			tr1.p0=cl0;
			tr1.p1=cl1;
			tr1.p2=cl6;;

		glm::vec3 minBoundary,maxBoundary;
        getAABBForTriangle(tr1,minBoundary,maxBoundary);

	  int start_x,end_x;
	  int start_y,end_y;

      if(minBoundary.x<-1.0f)start_x=0;
	  else start_x=(int)((minBoundary.x+1.0f)*resolution.x/2.0);
	  if(minBoundary.y<-1.0f)start_y=0;
	  else start_y=(int)((minBoundary.y+1.0f)*resolution.y/2.0);

	  if(maxBoundary.x>1.0f)end_x=resolution.x-1;
	  else end_x=(int)((maxBoundary.x+1.0f)*resolution.x/2.0);
	  if(maxBoundary.y>1.0f)end_y=resolution.y-1;
	  else end_y=(int)((maxBoundary.y+1.0f)*resolution.y/2.0);

	  for(int i=start_x;i<=end_x;i++)
	  for(int j=start_y;j<=end_y;j++)
	  {
		  int PIndex=i + (j * resolution.x);
		  	bool hold = true;
					while (hold) 
					{
						if (atomicExch(&lock, 1u)==0u) 
						{
							float value=denseMap[PIndex];
							value-=dvalue;
							denseMap[PIndex]=value;
							hold = false;
							atomicExch(&lock,0u);
						}
					} 
	  }

	  float coff[9]={1.0f,2.0f,1.0f,2.0f,4.0f,2.0f,1.0f,2.0f,1.0f};
	  int count=0;
	  for(int i=x-1;i<=x+1;i++)
	  for(int j=y-1;j<=y+1;j++)
	  {
		  if((i<=resolution.x) && (j<=resolution.y) && (i>=0)&&(j>=0)){
		  int PIndex=i + (j * resolution.x);
		  	bool hold = true;
					while (hold) 
					{
						if (atomicExch(&lock, 1u)==0u) 
						{
							float value=denseMap[PIndex];
							value-=coff[count]/16.0*dvalue;
							denseMap[PIndex]=value;
							hold = false;
							atomicExch(&lock,0u);
						}
					} 

			}

		count++;
	  }

	 
  }

		
}
__global__ void PosGeoFin(triangle* primitives, int primitivesCount, triangle* newPrimitives,bool* filledflag,cudaMat4 Projection,fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  fragment fr=depthbuffer[index];
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  bool flag=false;

	  if(glm::dot(normal,viewdirection)<0.1f){
			flag=true;
		}else if((x>0)&&(x<resolution.x)&&(y>0)&&(y<resolution.y)){


		int A = (x-1) + ((y-1) * resolution.x);
		int H = (x+1) + ((y+1) * resolution.x);
		int F = (x-1) + ((y+1) * resolution.x);
		int C = (x+1) + ((y-1) * resolution.x);
		
		fragment Afrag=depthbuffer[A];
		fragment Hfrag=depthbuffer[H];
		fragment Ffrag=depthbuffer[F];
		fragment Cfrag=depthbuffer[C];

		float In=0.5f*(glm::dot(Afrag.normal,Hfrag.normal)+glm::dot(Cfrag.normal,Ffrag.normal));

		if(In<0.99f) flag=true;
				
	  }

	  if(!flag) return;

	  triangle tr=primitives[fr.triangleID];

		  if((glm::dot(-tr.pe0,tr.ne0)<0.0f)&&(glm::dot(-tr.pe1,tr.ne1)<0.0f)&&(glm::dot(-tr.pe2,tr.ne2)<0.0f)){
			return;
		  }
		   if(filledflag[5*fr.triangleID])return;
		   glm::vec3 root=(tr.pe0+tr.pe2+tr.pe1)*1.0f/3.0f;
		   glm::vec3 rootNormal=getNormalInEyeSpace(glm::vec3(0.5f,0.5f,0.5f),tr);
		   glm::vec3 lat=-glm::normalize(glm::cross(rootNormal,-root));
		   glm::vec3 newGeoNormal=glm::normalize(glm::cross(rootNormal,lat));
		
			float phi=0.9f;
		    float revphi=3.0f;
			float cof=clamp(glm::length(tr.pe0-0.5f*(tr.pe1+tr.pe2)),0.0f,0.01f);

			glm::vec3 p0=root+cof*lat;
			glm::vec3 p1=root-cof*lat;
			
			glm::vec3 p2=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*phi;
			glm::vec3 p3=root+revphi*cof*rootNormal+cof*lat*2.0f*phi*(phi-1.0f);
			
			glm::vec3 p4=(p2+p3)*0.5f+phi*revphi*cof*rootNormal;
			glm::vec3 p5=(p2+p3)*0.5f+phi*revphi*cof*rootNormal-cof*lat*2.0f*phi*phi;

			glm::vec3 p6=root-cof*lat+revphi*cof*rootNormal*(1.0f+phi+phi*phi);

			glm::vec3 cl0=multiplyMV3(Projection,glm::vec4(p0,1.0f));
		    glm::vec3 cl1=multiplyMV3(Projection,glm::vec4(p1,1.0f));
			
			glm::vec3 cl2=multiplyMV3(Projection,glm::vec4(p2,1.0f));
		    glm::vec3 cl3=multiplyMV3(Projection,glm::vec4(p3,1.0f));
			
			glm::vec3 cl4=multiplyMV3(Projection,glm::vec4(p4,1.0f));
		    glm::vec3 cl5=multiplyMV3(Projection,glm::vec4(p5,1.0f));
			
			glm::vec3 cl6=multiplyMV3(Projection,glm::vec4(p6,1.0f));

			glm::vec3 color=glm::vec3(0.9,0.9,0.9);
			float width=0.01f;
			triangle tr1;
			tr1.pe0=p0;
			tr1.pe1=p2;
			tr1.pe2=p1;
			tr1.p0=cl0;
			tr1.p1=cl2;
			tr1.p2=cl1;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+0]=tr1;

			tr1.pe0=p1;
			tr1.pe1=p2;
			tr1.pe2=p3;
			tr1.p0=cl1;
			tr1.p1=cl2;
			tr1.p2=cl3;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=true;
			tr1.Edge01=false;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+1]=tr1;

			tr1.pe0=p2;
			tr1.pe1=p4;
			tr1.pe2=p3;
			tr1.p0=cl2;
			tr1.p1=cl4;
			tr1.p2=cl3;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+2]=tr1;
			
			tr1.pe0=p3;
			tr1.pe1=p4;
			tr1.pe2=p5;
			tr1.p0=cl3;
			tr1.p1=cl4;
			tr1.p2=cl5;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=true;
			tr1.Edge01=false;
			tr1.Edge12=false;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+3]=tr1;
			
			tr1.pe0=p4;
			tr1.pe1=p6;
			tr1.pe2=p5;
			tr1.p0=cl4;
			tr1.p1=cl6;
			tr1.p2=cl5;;
			tr1.c0=color;
			tr1.c1=color;
			tr1.c2=color;
			tr1.ne0=newGeoNormal;
			tr1.ne1=newGeoNormal;
			tr1.ne2=newGeoNormal;
			tr1.Edge02=false;
			tr1.Edge01=true;
			tr1.Edge12=true;
			tr1.width=width;
			newPrimitives[5*fr.triangleID+4]=tr1;
						

			filledflag[5*fr.triangleID]=true;
			filledflag[5*fr.triangleID+1]=true;
			filledflag[5*fr.triangleID+2]=true;
			filledflag[5*fr.triangleID+3]=true;
		    filledflag[5*fr.triangleID+4]=true;
	 
	 }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer,float frame){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
#ifdef AntiAliasing
	glm::vec3 previousColor=framebuffer[index];
    framebuffer[index] = previousColor*(frame)/(frame+1.0f)+depthbuffer[index].color/(frame+1.0f);
#else
    framebuffer[index] = depthbuffer[index].color;
#endif
  }
}


glm::mat4 calculateProjectMatrix(glm::vec2 resolution){
	
	float aspect=(float)resolution.x/(float)resolution.y;
	assert(zNear>0.0f);
	assert(zFar>0.0f);
	float range=1.0f;
	float left = -range * aspect;
	float right = range * aspect;
	float bottom = -range;
	float top = range;

	glm::mat4 result(0.0f);
	result[0][0] = (2.0f * zNear) / (right - left);
	result[2][0] = (right+left)/(right-left);
	result[1][1] = (2.0f * zNear) / (top - bottom);
	result[2][1] = (top+bottom)/(top-bottom);
	result[2][2] = - (zFar + zNear) / (zFar - zNear);
	result[2][3] = - 1.0f;
	result[3][2] = - (2* zFar * zNear) / (zFar - zNear);

	result=glm::perspective(45.0f,((float)resolution.y)/((float)resolution.x),(float)zNear,(float)zFar);

	return result;
}	

void setUpCudaThread(glm::vec2 resolution){ 
	
  dim3 a(tileSize, tileSize);
  dim3 b((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
  threadsPerBlock=  dim3(tileSize, tileSize);;
  fullBlocksPerGrid= dim3 ((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
}

void setUpframeBuffer(glm::vec2 resolution){

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
}

void clearframeBuffer(glm::vec2 resolution){
   clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
   cudaDeviceSynchronize();
   checkCUDAError("clear frame buffer");
}

void setUpdepthbuffer(glm::vec2 resolution){
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));
  checkCUDAError("set depthbuffer failed!");
}

void cleardepthbuffer(glm::vec2 resolution,glm::vec3 color){
  checkCUDAError("no error entering clear depth buffer0");
  fragment frag;
  frag.color = color;
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,0);
  frag.depth=FLT_MAX;
  frag.triangleID=-1;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);
  cudaDeviceSynchronize();
  checkCUDAError("clear depth buffer fail3");

}




void setUpCamera(glm::vec3 translation, glm::vec3 rotation,glm::vec2 resolution){
	cameraMatrix =glm::lookAt(translation,rotation,glm::vec3(0,1,0));
}

void setUpCamera(glm::mat4 camMat){
   cameraMatrix=camMat;
}
void setUpProjection(glm::mat4 matrix){
   projectionMatrix=matrix;
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame,int ObjectID,glm::mat4 modelMatrix,int textureID){

 
	if(ObjectID>=cudaXBOList.size())
	{
		std::cout<<" can not find the object vbo info;"<<std::endl;
		exit(-1);
	}

	int ibosize=cudaXBOList[ObjectID].ibosize;
	int vbosize=cudaXBOList[ObjectID].vbosize;
	int nbosize=cudaXBOList[ObjectID].nbosize;
	int cbosize=cudaXBOList[ObjectID].cbosize;


  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));

  device_vbo_eyeCoord = NULL;
  cudaMalloc((void**)&device_vbo_eyeCoord, vbosize*sizeof(float));
  

  device_nbo_eyeCoord = NULL;
  cudaMalloc((void**)&device_nbo_eyeCoord, vbosize*sizeof(float));

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

 
  device_Texture=NULL;
  int textureWidth=0,textureHeight=0;
  if(textureID!=-1){
	  device_Texture=device_Texture_List[textureID].ptr;
	  textureWidth=device_Texture_List[textureID].width;
	  textureHeight=device_Texture_List[textureID].height;
  }

  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

   //get Trans Matrix
   cudaMat4 ModelViewCudaMatrix=utilityCore::glmMat4ToCudaMat4(cameraMatrix*modelMatrix);
   cudaMat4 ProjectionCudaMatrix=utilityCore::glmMat4ToCudaMat4(projectionMatrix);
   cudaMat4 NormalMatrix=utilityCore::glmMat4ToCudaMat4(glm::inverse(glm::transpose(cameraMatrix*modelMatrix)));


  //------------------------------
  //vertex shader
  //------------------------------
  checkCUDAError("no error entering vertex shader");
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(cudaXBOList[ObjectID].vbo,device_vbo, vbosize,ModelViewCudaMatrix,device_vbo_eyeCoord,ProjectionCudaMatrix,cudaXBOList[ObjectID].nbo,device_nbo_eyeCoord, NormalMatrix);
  cudaDeviceSynchronize();
  checkCUDAError("Kernel vertex failed!");

  //------------------------------
  //primitive assembly
  //------------------------------
  backfaceFlag=NULL;
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(resolution,device_vbo, vbosize, cudaXBOList[ObjectID].cbo, cbosize, cudaXBOList[ObjectID].ibo, ibosize, primitives,device_vbo_eyeCoord,device_nbo_eyeCoord,backfaceFlag,textureID);
  cudaDeviceSynchronize();
  checkCUDAError("Kernel assembly failed!");
 
  //------------------------------
  //rasterization
  //-----------------------------
  
  partialfill=NULL;
  //rasterizationKernel_wireframe<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution,backfaceFlag,NULL,device_Texture,textureWidth,textureHeight);
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution,backfaceFlag,partialfill,device_Texture,textureWidth,textureHeight);
  
  cudaDeviceSynchronize();

  checkCUDAError("Kernel rasterization failed!");


 if(addGraftalFlag){

  int ratio=1;
  bool *filledflag=NULL;
  triangle* edgeList = NULL;
  int newprimitiveBlocks=0;
  line* lineedgeList = NULL;
  switch(FGtye){
	case EDGE:
		 ratio=1;
		 newprimitiveBlocks=ceil(((float)ibosize/3*ratio)/((float)tileSize));
		 cudaMalloc((void**)&filledflag, (ibosize/3*ratio)*sizeof(bool));
		 cudaMalloc((void**)&edgeList, (ibosize/3*ratio)*sizeof(triangle));
		 GeoSignEdge<<<primitiveBlocks,tileSize>>>(primitives,ibosize/3,edgeList,filledflag,ProjectionCudaMatrix);
		 checkCUDAError("Edge geoShader  failed!");
		 rasterizationKernel_wireframe<<<newprimitiveBlocks, tileSize>>>(edgeList, ibosize/3*ratio, depthbuffer, resolution,backfaceFlag,filledflag,NULL,0,0);
		 cudaDeviceSynchronize();
		 break;
	case LineSegment:
		 ratio=6;
		 newprimitiveBlocks=ceil(((float)ibosize/3*ratio)/((float)tileSize));
		 cudaMalloc((void**)&filledflag, (ibosize/3*ratio)*sizeof(bool));
		 cudaMalloc((void**)&lineedgeList, (ibosize/3*ratio)*sizeof(line));
		 GeoLine<<<primitiveBlocks,tileSize>>>(primitives,ibosize/3,lineedgeList,filledflag,ProjectionCudaMatrix);
		 cudaDeviceSynchronize();
		 checkCUDAError("line geoShader  failed!");
		 LineRasterizationKernel<<<newprimitiveBlocks, tileSize>>>(lineedgeList, ibosize/3*ratio, depthbuffer, resolution,glm::vec3(1.0,1.0,1.0),filledflag);
		 cudaDeviceSynchronize();
	   break;
	case FIN:
		 ratio=5;
	     cudaMalloc((void**)&filledflag, (ibosize/3*ratio)*sizeof(bool));
		 cudaMalloc((void**)&edgeList, (ibosize/3*ratio)*sizeof(triangle));
		 GeoFin<<<primitiveBlocks,tileSize>>>(primitives,ibosize/3,edgeList,filledflag,ProjectionCudaMatrix);
		 cudaDeviceSynchronize();
		 checkCUDAError("fin geoShader  failed!");
		 newprimitiveBlocks=ceil(((float)ibosize/3*ratio)/((float)tileSize));
		 rasterizationKernel_wireframe<<<newprimitiveBlocks, tileSize>>>(edgeList, ibosize/3*ratio, depthbuffer, resolution,backfaceFlag,filledflag,NULL,0,0);
		 cudaDeviceSynchronize();
		break;

	case FACEANGLE:
		 ratio=1;
		 newprimitiveBlocks=ceil(((float)ibosize/3*ratio)/((float)tileSize));
		 cudaMalloc((void**)&filledflag, (ibosize/3*ratio)*sizeof(bool));
		 cudaMalloc((void**)&edgeList, (ibosize/3*ratio)*sizeof(triangle));
		 GeoFace<<<primitiveBlocks,tileSize>>>(primitives,ibosize/3,edgeList,filledflag,ProjectionCudaMatrix);
		 cudaDeviceSynchronize();
		 checkCUDAError("Face Angle geoShader  failed!");
		 rasterizationKernel_wireframe<<<newprimitiveBlocks, tileSize>>>(edgeList, ibosize/3*ratio, depthbuffer, resolution,backfaceFlag,filledflag,NULL,0,0);
		 cudaDeviceSynchronize();
		 break;
	case PosImageFin: //work for only one obj
		ratio=5;
	    newprimitiveBlocks=ceil(((float)ibosize/3*ratio)/((float)tileSize));
		cudaMalloc((void**)&filledflag, (ibosize/3*ratio)*sizeof(bool));
		clearFlag<<<newprimitiveBlocks,tileSize>>>(filledflag,ibosize/3*ratio,false);
		cudaMalloc((void**)&edgeList, (ibosize/3*ratio)*sizeof(triangle));
		PosGeoFin<<<fullBlocksPerGrid, threadsPerBlock>>>(primitives,ibosize/3,edgeList,filledflag,ProjectionCudaMatrix,depthbuffer, resolution);
		cudaDeviceSynchronize();
		checkCUDAError("pos image fin  failed!");
		rasterizationKernel_wireframe<<<newprimitiveBlocks, tileSize>>>(edgeList, ibosize/3*ratio, depthbuffer, resolution,backfaceFlag,filledflag,NULL,0,0);
		cudaDeviceSynchronize();
	    break;
	case DenseBasedFin:
		ratio=5;
	    newprimitiveBlocks=ceil(((float)ibosize/3*ratio)/((float)tileSize));
		cudaMalloc((void**)&filledflag, (ibosize/3*ratio)*sizeof(bool));
		clearFlag<<<newprimitiveBlocks,tileSize>>>(filledflag,ibosize/3*ratio,false);
		cudaMalloc((void**)&edgeList, (ibosize/3*ratio)*sizeof(triangle));

		float* denseMap=NULL;
        cudaMalloc((void**)&denseMap, (int)resolution.x*(int)resolution.y*sizeof(float));
		setDenseValue<<<fullBlocksPerGrid, threadsPerBlock>>>(denseMap,depthbuffer, resolution,1.0f);
		cudaDeviceSynchronize();
		checkCUDAError("set dense value  failed!");
		DenseGeoFin<<<fullBlocksPerGrid, threadsPerBlock>>>(primitives,ibosize/3,edgeList,filledflag,ProjectionCudaMatrix,depthbuffer,resolution,denseMap,1.0f);
		cudaDeviceSynchronize();
		checkCUDAError("dense based fin  failed!");
		rasterizationKernel_wireframe<<<newprimitiveBlocks, tileSize>>>(edgeList, ibosize/3*ratio, depthbuffer, resolution,backfaceFlag,filledflag,NULL,0,0);
		cudaDeviceSynchronize();
		cudaFree(denseMap);
		break;


  }
  checkCUDAError("edge rasterization failed!");
  cudaFree(lineedgeList);
  cudaFree(filledflag);
  cudaFree(edgeList);
  
 }

}


void saveDepthbufferAsTexture(glm::vec2 resolution){

  //set up depthbuffer
  shadowMap = NULL;
  cudaMalloc((void**)&shadowMap, (int)resolution.x*(int)resolution.y*sizeof(fragment));
  cudaMemcpy( shadowMap, depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment), cudaMemcpyDeviceToDevice);
    
}

void sendColor(glm::vec2 resolution,FragmentTYPE type){

  
  glm::vec4 lightPos4=cameraMatrix*glm::vec4(-lightPosition.x,lightPosition.y,lightPosition.z,1.0f);
  glm::vec3 lightPos=glm::vec3(lightPos4.x,lightPos4.y,lightPos4.z);
  glm::vec3 lColor=glm::vec3(1.0f,1.0f,1.0f);
  int textureWidth=0,textureHeight=0;
  glm::mat4 cameraToWorld=glm::inverse(glm::lookAt(cameraTrans,cameraRot,glm::vec3(0,1,0)));
  glm::mat4 worldToLight=glm::lookAt(lightPosition,lightRot,glm::vec3(0,1,0));
  glm::mat4 cameraToLightClip=lightProjectionMatrix*worldToLight*cameraToWorld;
  cudaMat4 TolightMatrix=utilityCore::glmMat4ToCudaMat4(cameraToLightClip);
  checkCUDAError("no error so far enter fragment");

  switch(type){  
  case PASS:
	   break;
  case PHONG:
	    //phong
		fragmentShadeKernelPHONG<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,lightPos,lColor);
		cudaDeviceSynchronize();
	    break;
  case EDGEENHANCE:
	  fragmentShadeKernelEDGEENHANCE<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,lightPos,lColor);
	  cudaDeviceSynchronize();
	  checkCUDAError("Edge enhancement  failed!");
	  break;
  case NPR:
	   //NPR
		device_Texture=NULL;
		device_Texture=device_Texture_List[Warp_Diffuse_ID].ptr;
		textureWidth=device_Texture_List[Warp_Diffuse_ID].width;
		textureHeight=device_Texture_List[Warp_Diffuse_ID].height;
		 fragmentShadeKernelNPR<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,lightPos,lColor,device_Texture,textureWidth,textureHeight);
		 cudaDeviceSynchronize();
		 checkCUDAError("NPR failed!");
		 break;
   case  EDGENPR:
	   //EDGE ENHANCE NPR
		device_Texture=NULL;
		device_Texture=device_Texture_List[Warp_Diffuse_ID].ptr;
		textureWidth=device_Texture_List[Warp_Diffuse_ID].width;
		textureHeight=device_Texture_List[Warp_Diffuse_ID].height;
		fragmentShadeKernelEDGENPR<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,lightPos,lColor,device_Texture,textureWidth,textureHeight);
		cudaDeviceSynchronize();
		checkCUDAError("EDGE NPR failed!");
		break;
  case PHONGSHADOW:
	    
	    fragmentShadeKernelPHONGSHADOW<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,lightPos,lColor,shadowMap,TolightMatrix);
		cudaDeviceSynchronize();
		checkCUDAError("phong shadow failed!");
		break;

 

  }
 
}
void sendResult(uchar4* PBOpos, glm::vec2 resolution, float frame){
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer,frame);
  checkCUDAError("render result failed");
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();
  checkCUDAError("set result failed!");

}

void drawkernelCleanup(){

  #ifdef BackFaceCulling
  cudaFree(backfaceFlag);
 #endif

#ifdef GEOMETRYSHADER
	#ifdef BackFaceCulling
	 cudaFree(newbackfaceFlag);
	#endif
	cudaFree(newprimitives);
#endif

#ifdef AntiAliasing
	cudaFree(partialfill);
#endif

  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_vbo_eyeCoord );
  cudaFree( device_nbo );
  cudaFree( device_nbo_eyeCoord );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
}

void bufferCleanup(){
  cudaFree( framebuffer );
  cudaFree( depthbuffer );

#ifdef SHADOWMAP
  cudaFree(shadowMap);
#endif

}
