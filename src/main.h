// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H


#include <GL/glew.h>
#include <GL/glut.h>


#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <string>
#include <array>
#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "rasterizeKernels.h"
#include "utilities.h"
#include "ObjCore/objloader.h"
#include "stb_image/stb_image.h"
#include "Scene.h"
#include "glm/gtc/matrix_transform.hpp"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;
obj* mesh;


float* vbo;
int vbosize;
float* nbo;
int nbosize;
float* uvcoord;
float* cbo;
int cbosize;
int* ibo;
int ibosize;


struct ObjData{
	obj* ptr;
	glm::vec3 scale;
	glm::vec3 rotation;
	glm::vec3 translation;
	int TextureID;
	glm::vec3 defaultColor;
};

//----------CUDA STUFF-----------
//-------------------------------

int width=1000; int height=1000;
bool firstTime=true;
bool initXBOFlag=true;

vector<ObjData> meshList;
std::vector<Texture> textList;
glm::vec3 backgroundColor=glm::vec3(1.0,1.0,1.0);
//-------------------------------
//----------Object Scene----------
//-------------------------------


// put your model obj file name in objFiles array, put your texture file name in textureFiles

array<string,1> objFiles={"BunnyUV.obj"};  
//array<string,3> objFiles={"body.obj","eyesUV.obj","hatUV.obj"};
//std::array<std::string,3> textureFiles={"hat.jpg","paper.jpg","warp_ramp.jpg"};
//array<string,1> objFiles={"torus.obj"};
//array<string,1> objFiles={"house.obj"};
//array<string,1> objFiles={"terrain.obj"};
std::array<std::string,0> textureFiles={};

int Warp_Diffuse_ID=2; // NPR effect warp id

glm::vec3 cameraTrans = glm::vec3(0.0,2.0,4.0);
glm::vec3 cameraRot = glm::vec3(0,1.0,0); 
glm::mat4 cameraProjection=glm::perspective(45.0f,((float)width)/((float)height),0.01f,100.0f);

//glm::vec3 lightPosition=glm::vec3(-2.0,5.0,1.0); //Dante EDGE NPR
glm::vec3 lightPosition = glm::vec3(-2.0,2.0,20.0);
glm::vec3 lightRot=glm::vec3(0.0,2.0,0.0);
glm::mat4 lightProjectionMatrix=glm::perspective(45.0f,((float)width)/((float)height),0.01f,100.0f);

FragmentTYPE FStype=PASS;
bool addGraftalFlag=true;
FinGeoTYPE  FGtye=DenseBasedFin;

//-------------------------------
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x,int y); 
void deleteImage();
//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init(int argc, char* argv[]);
void initObjTextures();
void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif