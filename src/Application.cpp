//
// Created by aditya on 07/04/22.
//

#include <iostream>
#include <fstream>
#include "glad/glad.h"
#include "Application.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "imgui_impl_opengl3_loader.h"


static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

unsigned int get_shader(const std::string& filename, GLenum shaderType){
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

unsigned int get_shader_program(const std::string& vertex_shader_file,const std::string& fragment_shader_file){
	unsigned int vertexShader = get_shader(vertex_shader_file, GL_VERTEX_SHADER);
	unsigned int fragmentShader = get_shader(fragment_shader_file, GL_FRAGMENT_SHADER);
	unsigned int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, fragmentShader);
	glAttachShader(shaderProgram, vertexShader);
	glLinkProgram(shaderProgram);
	
	int success;
	char infoLog[512];
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	
	
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return shaderProgram;
}

Application::Application() : m_width(1280), m_height(720), m_window(nullptr){
	this->init_window();
}

Application::Application(int w, int h) : m_width(w), m_height(h), m_window(nullptr) {
	this->init_window();
}

void Application::init_window() {
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()){
		std::cerr << "Failed to initialize Application\n";
		exit(-1);
	}
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	m_window = glfwCreateWindow(m_width, m_height, m_title, nullptr, nullptr);
	if (m_window == nullptr) {
		std::cerr << "Failed to create Application\n";
		exit(-1);
	}
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1); // Enable vsync
	
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init(m_glsl_version);
}

Application::~Application() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Application::run() {
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	
	float vertices[] = {
			0.8f , 0.8f,  // top right
			0.8f , -0.8f,  // bottom right
			-0.8f, -0.8f, // bottom left
			-0.8f, 0.8f  // top left
	};
	
	unsigned int indices[] = {  // note that we start from 0!
			0, 1, 3,   // first triangle
			1, 2, 3    // second triangle
	};
	
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VAO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	
	unsigned int EBO;
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	
	unsigned int shaderProgram = get_shader_program("vertex_shader.glsl", "fragment_shader.glsl");
	
	while (!glfwWindowShouldClose(m_window))
	{
		
		glfwPollEvents();
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		{
			ImGui::Begin("Statistics");
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
		}
		
		ImGui::Render();
		
		int display_w, display_h;
		glfwGetFramebufferSize(m_window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(m_clear_color.x * m_clear_color.w, m_clear_color.y * m_clear_color.w, m_clear_color.z * m_clear_color.w, m_clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
		glEnableVertexAttribArray(0);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(m_window);
	}
}
