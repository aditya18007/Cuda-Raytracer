//
// Created by aditya on 08/04/22.
//

#ifndef HELIOS_SHADER_H
#define HELIOS_SHADER_H

#include <string>
#include "glad/glad.h"

class Shader {
	unsigned int shader_program;
	const std::string vertex_shader_filename;
	const std::string fragment_shader_filename;
	unsigned int m_shader_program;
	unsigned int get_shader(const std::string& filename, GLenum shaderType);
public:
	Shader(std::string , std::string );
	void compile();
	void use();
};


#endif //HELIOS_SHADER_H
