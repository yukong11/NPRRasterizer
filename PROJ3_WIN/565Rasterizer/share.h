#ifndef SHARE_H
#define SHARE_H

#include <stdlib.h>
#include <string>
#include <array>

struct Texture{
	unsigned int id;
	unsigned char* ptr;
	int width;
	int height;
	int depth;
};

array<string,2> textureFiles={"leather.jpg","fract.jpg"};
vector<Texture> textList;
#endif