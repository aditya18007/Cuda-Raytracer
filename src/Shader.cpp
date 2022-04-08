//
// Created by aditya on 08/04/22.
//

#include "Shader.h"

#include <utility>
#include <iostream>
#include <fstream>

Shader::Shader(std::string  v_file, std::string  f_file)
: vertex_shader_filename(std::move(v_file)), fragment_shader_filename(std::move(f_file))
{}

unsigned int Shader::get_shader(const std::string &filename, GLenum shaderType) {
	std::string pwd(__FILE__);
	
	auto last = pwd.find_last_of('/');
	auto directory = pwd.substr(0, last);
	directory += "/Shaders/";
	auto shader_location = directory + filename;
	std::ifstream shader_file(shader_location);
	std::string file_contents((std::istreambuf_iterator<char>(shader_file)),
	                          std::istreambuf_iterator<char>());
	unsigned int shader = glCreateShader(shaderType);
	const char * c_str_fragment_shader= file_contents.c_str();
	glShaderSource(shader, 1, &c_str_fragment_shader, nullptr);
	glCompileShader(shader);
	int success;
	char infoLog[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	return shader;
}

void Shader::compile() {
	unsigned int vertexShader = get_shader(vertex_shader_filename, GL_VERTEX_SHADER);
	unsigned int fragmentShader = get_shader(fragment_shader_filename, GL_FRAGMENT_SHADER);
	m_shader_program = glCreateProgram();
	glAttachShader(m_shader_program, fragmentShader);
	glAttachShader(m_shader_program, vertexShader);
	glLinkProgram(m_shader_program);
	
	int success;
	char infoLog[512];
	glGetProgramiv(m_shader_program, GL_LINK_STATUS, &success);
	
	
	if (!success) {
		glGetProgramInfoLog(m_shader_program, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Shader::use() {
	glUseProgram(m_shader_program);
}