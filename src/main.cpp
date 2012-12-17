// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------


bool initObjScene(){

   int count=0;
   for(unsigned int i=0; i<objFiles.size(); i++){
    string name=objFiles[i];
	string header; string type;
    istringstream liness(name);
    getline(liness, header, '.');
	getline(liness, type);
	string path ="../../objs/";
    if(strcmp(type.c_str(), "obj")==0)
   {
      //renderScene = new scene(data);
      mesh = new obj();
      objLoader* loader = new objLoader(path+name, mesh);
	  mesh->buildVBOs();
	  ObjData data;
	  data.ptr=mesh;
	  data.scale=glm::vec3(1.0,1.0,1.0);
	  data.translation =glm::vec3 (0,0,0);
	  data.rotation =glm::vec3(0.0f,0.0f,0.0f);
	  data.TextureID=-1;
	  data.defaultColor=glm::vec3(0.2f,0.2f,0.2f);
	 if(strcmp(header.c_str(), "backgroundUV")==0){
		  data.scale=glm::vec3(0.5,0.3,0.4);
		  data.translation =glm::vec3 (0,0,0);
		  data.rotation =glm::vec3(0,0,0);
		  data.TextureID=1;
	  }else if(strcmp(header.c_str(), "BunnyUV")==0){
		  cameraTrans = glm::vec3(0.0,2.0,4.0);
		  //cameraTrans = glm::vec3(0.0,2.0,1.0);
		  cameraRot = glm::vec3(0,1.0,0);
		  data.scale=glm::vec3(0.2,0.2,0.2);
		  data.translation =glm::vec3 (0,0,0);
		  data.rotation =glm::vec3(0,0,0);
		  data.TextureID=-1;
		  data.defaultColor=glm::vec3(1.0, 0.5, 0.25);
		  backgroundColor=glm::vec3(0.0,0.0,0.0);
		 // data.defaultColor=glm::vec3(0.0, 0.0, 0.0);
	  }else if(strcmp(header.c_str(), "hatUV")==0){
		  data.scale=glm::vec3(0.4,0.4,0.4);
		  data.translation =glm::vec3 (0,0,0);
		  data.rotation =glm::vec3(0,0,0);
		  data.TextureID=0;
	  }else if(strcmp(header.c_str(), "eyesUV")==0){
		  cameraTrans = glm::vec3(0.0,5.0,7.0); 
		  cameraRot = glm::vec3(0,2.0,0);
		  data.scale=glm::vec3(0.4,0.4,0.4);
		  data.translation =glm::vec3 (0,0,0);
		  data.rotation =glm::vec3(0,0,0);
		  data.defaultColor=glm::vec3(0.0,0.0,0.0);
		  data.TextureID=-1;
	  }
	  else if(strcmp(header.c_str(), "body")==0){
		  data.scale=glm::vec3(0.4,0.4,0.4);
		  data.translation =glm::vec3 (0,0,0);
		  data.rotation =glm::vec3(0,0,0);
		  data.defaultColor=glm::vec3(0.9,0.9,0.9);
		  data.TextureID=-1;
	  } else if(strcmp(header.c_str(), "house")==0){
		  cameraTrans = glm::vec3(0.0,2.0,15.0);
		  lightPosition = glm::vec3(-5.0,2.0,20.0);
		  data.scale=glm::vec3(0.1,0.1,0.1);
		  data.defaultColor=glm::vec3(0.5,0.5,0.5);
	
	  } else if(strcmp(header.c_str(), "torus")==0){
		  cameraTrans = glm::vec3(0.0,3.0,7.0);
		  cameraRot=glm::vec3(0,1.0,0);
		  data.scale=glm::vec3(1.0,1.0,1.0);
		   data.rotation =glm::vec3(10.0f,0.0f,0.0f);
		  data.defaultColor=glm::vec3(0.9,0.9,0.9);
		  //data.defaultColor=glm::vec3(1.0,1.0,1.0);
	  }  else if(strcmp(header.c_str(), "terrain")==0){

		  cameraTrans = glm::vec3(0.0,4.0,30);
		  cameraRot = glm::vec3(0.0,5.0,0.0);
		  data.scale=glm::vec3(1.0,1.0,1.0);
		  data.rotation =glm::vec3(20.0f,-10.0f,0.0f);
		  data.defaultColor=glm::vec3(1.0,0.0,0.0);
		  backgroundColor=glm::vec3(1.0,1.0,1.0);
	  } 

	  if(data.TextureID!=-1){
		    mesh->buildTBO();
	  }
	  meshList.push_back(data);
      delete loader;
      count++;
    }
  }
  initObjTextures();
  if(count==meshList.size())
	  return true;
  else 
	  return false;
    
}

int main(int argc, char** argv){

  bool loadedScene = false;
   if(initObjScene())
	   loadedScene=true;
   else {
	  cout << "unsupport obj format" << endl;
      return 0;
   }

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // Launch CUDA/GL

  init(argc, argv);


  initCuda();
  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

 
	glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

    glutMainLoop();
	drawkernelCleanup();
	bufferCleanup();
	cleanXBOs(); 
	cleanTexture();

  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------
glm::vec3 defaultColor= glm::vec3(1.0,0.5,0.25);
void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
 if(firstTime==false){
	  frame++;
      fpstracker++;
	  return;
  }

  dptr=NULL;
  cudaGLMapBufferObject((void**)&dptr, pbo);
  glm::vec2 resolution = glm::vec2(width, height);
  setUpCudaThread(resolution);
  
  setUpframeBuffer(resolution);
  //glm::vec3 backgroundColor=glm::vec3(0.93,0.69,0.62);
  //glm::vec3 backgroundColor=glm::vec3(0.53,0.243,0.192);
 // glm::vec3 backgroundColor=glm::vec3(1.0,1.0,1.0);

  clearframeBuffer(resolution);
  setUpdepthbuffer(resolution);
 
 

#ifdef SHADOWMAP
  FStype=PHONGSHADOW;
  cleardepthbuffer(resolution,backgroundColor);
  setUpCamera(lightPosition,lightRot,resolution);
  //lightProjectionMatrix=glm::perspective(45.0f,((float)resolution.y)/((float)resolution.x),0.01f,100.0f);
  setUpProjection(lightProjectionMatrix);
   if(initXBOFlag){
	  cleanXBOs();
	  for(unsigned int i=0;i <meshList.size();i++){

		  mesh=meshList[i].ptr;
	  
		  vbo = mesh->getVBO();
		  vbosize = mesh->getVBOsize();

		  nbo =mesh->getNBO();
		  nbosize =mesh->getNBOsize();

 

		  if(meshList[i].TextureID==-1){
			  float a[3];
			  a[0]=meshList[i].defaultColor.x;
			  a[1]=meshList[i].defaultColor.y;
			  a[2]=meshList[i].defaultColor.z;
			  cbo=a;
			  cbosize=3;
		  }else{
			  cbo=mesh->getTBO();
			  cbosize=mesh->getTBOsize();
		  }


		  ibo = mesh->getIBO();
		  ibosize = mesh->getIBOsize();
		  setXBO(vbo, vbosize, nbo,nbosize,cbo, cbosize, ibo, ibosize);
		}
	   
      initXBOFlag=false;
  }

  for(unsigned int i=0;i <meshList.size();i++){
     glm::mat4 modelMatrix= utilityCore::buildTransformationMatrix(meshList[i].translation,meshList[i].rotation,meshList[i].scale);
	 cudaRasterizeCore(dptr,resolution, frame, i,modelMatrix,meshList[i].TextureID);
	 drawkernelCleanup();
 
  }
  saveDepthbufferAsTexture(resolution);

#endif

 cleardepthbuffer(resolution,backgroundColor);
  setUpCamera(cameraTrans,cameraRot,resolution);
  setUpProjection(cameraProjection);
  if(initXBOFlag){
	  cleanXBOs();
	  for(unsigned int i=0;i <meshList.size();i++){

		  mesh=meshList[i].ptr;
	  
		  vbo = mesh->getVBO();
		  vbosize = mesh->getVBOsize();

		  nbo =mesh->getNBO();
		  nbosize =mesh->getNBOsize();

 

		  if(meshList[i].TextureID==-1){
			  float a[3];
			  a[0]=meshList[i].defaultColor.x;
			  a[1]=meshList[i].defaultColor.y;
			  a[2]=meshList[i].defaultColor.z;
			  cbo=a;
			  cbosize=3;
		  }else{
			  cbo=mesh->getTBO();
			  cbosize=mesh->getTBOsize();
		  }
		  ibo = mesh->getIBO();
		  ibosize = mesh->getIBOsize();
		  setXBO(vbo, vbosize, nbo,nbosize,cbo, cbosize, ibo, ibosize);
		}
	   
      initXBOFlag=false;
  }

  for(unsigned int i=0;i <meshList.size();i++){
     glm::mat4 modelMatrix= utilityCore::buildTransformationMatrix(meshList[i].translation,meshList[i].rotation,meshList[i].scale);
	 cudaRasterizeCore(dptr,resolution, frame, i,modelMatrix,meshList[i].TextureID);
	 drawkernelCleanup();
 
  }
  
  sendColor(resolution,FStype);
  sendResult(dptr,resolution,frame);
  bufferCleanup();
  cudaGLUnmapBufferObject(pbo);
  firstTime=false;

  vbo = NULL;
  cbo = NULL;
  ibo = NULL;

  frame++;
  fpstracker++;

}


  void display(){
    runCuda();
	time_t seconds2 = time (NULL);

    if(seconds2-seconds >= 1){

      fps = fpstracker/(seconds2-seconds);
      fpstracker = 0;
      seconds = seconds2;

    }

    string title = "CIS565 Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";
    glutSetWindowTitle(title.c_str());

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
        GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glClear(GL_COLOR_BUFFER_BIT);   

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

    glutPostRedisplay();
    glutSwapBuffers();
  }

  void keyboard(unsigned char key, int x, int y)
  {
    switch (key) 
    {
       case(27):
         shut_down(1);    
         break;
    }
  }

  enum actions{ROT,FOCUS,ZOME,NONE};
	static GLint        action=NONE;
	static int     xStart = 0.0, yStart = 0.0;

	void mouse(int button, int state, int x, int y){
	 
		 if(state == GLUT_DOWN)
		{
			if(button == GLUT_LEFT_BUTTON)
			{
				action=ROT;
			}else if(button == GLUT_MIDDLE_BUTTON)
			{
				action=FOCUS;
			}else if(button == GLUT_RIGHT_BUTTON)
			{
				action=ZOME;
			}
			xStart=x;
			yStart=y;		  
		}
		else
		{
			action=NONE;
		}
	}

	void motion(int x,int y){
		switch(action){
		case ROT:
			//cameraRot.y-=(float)(x-xStart)*0.05f;
			//cameraRot.x+=(float)(y-yStart)*0.05f;
			cameraTrans.x+=(float)(x-xStart)*0.01f;
			cameraTrans.y-=(float)(y-yStart)*0.01f;

			//glm::vec3 viewDir=cameraTrans-cameraRot;
			//float length= glm::length(viewDir);
			//float angleY=atan(viewDir.y,viewDir.z);
			//float angleX=atan(
			firstTime=true;
			frame=0;
			break;
		case ZOME:
			//cameraTrans.z+=float(y-yStart)*0.01f;
			float length= glm::length(cameraTrans-cameraRot);
			cameraTrans=cameraTrans-glm::normalize(cameraTrans-cameraRot)*float(y-yStart)*0.01f;
			//if(cameraTrans.z>-1.0)cameraTrans.z=-1.0;
			firstTime=true;
			frame=0;
			break;	
		}
		xStart=x;
		yStart=y;
		glutPostRedisplay();
	}


  
//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initObjTextures(){
	string path ="../../textures/";
	for(unsigned int i=0; i<textureFiles.size();i++){
		Texture tex;
		tex.ptr =stbi_load((path+textureFiles[i]).c_str(),&tex.width,&tex.height,&tex.depth,0);
		textList.push_back(tex);
	}
	
}

  void init(int argc, char* argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("CIS565 Rasterizer");

    // Init GLEW
    glewInit();
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
      /* Problem: glewInit failed, something is seriously wrong. */
      std::cout << "glewInit failed, aborting." << std::endl;
      exit (1);
    }

    initVAO();
    initTextures();
  }


void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  setUpCudaTexture(textList);
  runCuda();
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}


void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  drawkernelCleanup();
  bufferCleanup();
  cleanXBOs();
  cleanTexture();
  cudaDeviceReset();
  exit(return_code);
}
