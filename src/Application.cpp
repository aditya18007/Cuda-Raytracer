//
// Created by aditya on 07/04/22.
//

#include <iostream>
#include "glad/glad.h"
#include "Application.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "imgui_impl_opengl3_loader.h"
#include "Shader.h"
#include "Quad.h"
#include <cuda.h>
#include "call_ray_tracer.h"
#include "Frame.h"

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
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
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}
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
	Frame f(m_width, m_height);
	Quad q;
	q.enable_attributes();
	Shader shader("vertex_shader.glsl", "fragment_shader.glsl");
	shader.compile();
	int i = 0;
	while (!glfwWindowShouldClose(m_window))
	{
		
		launch_kernel(f.get_bitmap_surface(), m_width, m_height, i++);
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
		
		glBindTexture(GL_TEXTURE_2D, f.get_texture_ID());
		
		shader.use();
		q.draw();
		
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(m_window);
	}
}
