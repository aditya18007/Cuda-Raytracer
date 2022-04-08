//
// Created by aditya on 08/04/22.
//

#ifndef HELIOS_FRAME_H
#define HELIOS_FRAME_H


#include "glad/glad.h"
#include <cuda_gl_interop.h>
class Frame {
	GLuint m_textureID{};
	const int m_width;
	const int m_height;
	unsigned char* m_data;
	cudaSurfaceObject_t m_bitmap_surface;
public:
	Frame(int width, int height);
	cudaSurfaceObject_t get_bitmap_surface();
	GLuint get_texture_ID();
};


#endif //HELIOS_FRAME_H
