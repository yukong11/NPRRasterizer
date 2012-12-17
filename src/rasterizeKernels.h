// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_access.inl"
#include <string>
#include <array>
#include <vector>
#include "Scene.h"

void cleanXBOs();
void setXBO( float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize);
void saveDepthbufferAsTexture(glm::vec2 resolution);
void sendColor(glm::vec2 resolution,FragmentTYPE type);
void saveDepthbufferAsTexture();
void setUpCudaThread(glm::vec2 resolution);
void bufferCleanup();
void drawkernelCleanup();
void setUpdepthbuffer(glm::vec2 resolution);
void cleardepthbuffer(glm::vec2 resolution,glm::vec3 color);
void setUpframeBuffer(glm::vec2 resolution);
void clearframeBuffer(glm::vec2 resolution);
void setUpCudaTexture(std::vector<Texture> textList);
void cleanTexture();
void setUpCamera(glm::vec3 translation, glm::vec3 rotation,glm::vec2 resolution);
void setUpCamera(glm::mat4 camMat);
void setUpProjection(glm::mat4);
void sendResult(uchar4* PBOpos, glm::vec2 resolution, float frame);
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color);
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame,int ObjectID,glm::mat4 modelView,int textureID);

#endif //RASTERIZEKERNEL_H
