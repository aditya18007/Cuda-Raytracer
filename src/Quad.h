//
// Created by aditya on 08/04/22.
//

#ifndef HELIOS_QUAD_H
#define HELIOS_QUAD_H


class Quad {
	float m_vertices[16] = {
			//<-position->  <-tex_coords->
			1.0f, 1.0f, 1.0f , 1.0f,// top right
			1.0f,-1.0f, 1.0f , 0.0f,// bottom right
			-1.0f,-1.0f, 0.0f, 0.0f,// bottom left
			-1.0f,1.0f, 0.0f, 1.0f// top left
	};
	
	unsigned int m_indices[6] = {
			0, 1, 3,   // first triangle
			1, 2, 3    // second triangle
	};
	
	unsigned int m_VAO{};
	unsigned int m_EBO{};
public:
	Quad();
	void enable_attributes();
	void draw();
};


#endif //HELIOS_QUAD_H
