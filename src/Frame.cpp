//
// Created by aditya on 08/04/22.
//

#include "Frame.h"
#include <cuda_gl_interop.h>

Frame::Frame(int width, int height)
:m_width(width), m_height(height)
{
	glGenTextures(1, &m_textureID);
	
	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, m_textureID);
	
	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
	m_data = new unsigned char[m_width* m_height*4]; //4 = (R,G,B, alpha)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_data);
	
	cudaArray *bitmap_d;
	cudaGraphicsResource *cudaTextureID;
	cudaGraphicsGLRegisterImage(&cudaTextureID, m_textureID, GL_TEXTURE_2D,
	                            cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cudaTextureID, 0);
	cudaGraphicsSubResourceGetMappedArray(&bitmap_d, cudaTextureID, 0, 0);
	struct cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = bitmap_d;
	cudaCreateSurfaceObject(&m_bitmap_surface, &resDesc);
}

cudaSurfaceObject_t Frame::get_bitmap_surface() {
	return m_bitmap_surface;
}

GLuint Frame::get_texture_ID() {
	return m_textureID;
}
