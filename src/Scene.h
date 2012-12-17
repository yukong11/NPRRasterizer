#ifndef SCENE_H
#define SCENE_H

#include "glm/glm.hpp"
#include <string>
#include <vector>
#include <array>
#include "ObjCore/objloader.h"



struct Texture{
	unsigned char* ptr;
	int width;
	int height;
	int depth;
};

enum FragmentTYPE{
	PASS,
	PHONG,
	NPR,
	PHONGSHADOW,
	EDGEENHANCE,
	EDGENPR
};

enum FinGeoTYPE{
	HAND,
	EDGE, //normal sign method
	FIN,
	FACEANGLE,
	LineSegment,
	PosImageFin,
	DenseBasedFin
};
extern FragmentTYPE FStype;
extern FinGeoTYPE  FGtye;
extern std::vector<Texture> textList;
extern glm::vec3 cameraTrans;
extern glm::vec3 cameraRot;
extern glm::vec3 lightPosition;
extern glm::vec3 lightRot;
extern int Warp_Diffuse_ID;
extern glm::mat4 lightProjectionMatrix;
extern bool initXBOFlag;
extern bool firstTime;
extern bool addGraftalFlag;
#define EDGEDETECTION 
//#define SHADOWMAP


//------------parameters you can play with
// #define BackFaceScaling      //uncomment to enable back face scaling you must also umcomment one of the following line to tell the scalling method 
	 //#define ImageSpaceScaling    
	 //#define EyeSpaceScaling
#endif