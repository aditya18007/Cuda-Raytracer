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
#include "Update_frame.h"
#include "Frame.h"
#include "Camera.h"

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
	io.WantCaptureKeyboard = true;
	io.WantCaptureMouse = true;
	
	Quad quad;
	quad.enable_attributes();
	
	Shader shader("vertex_shader.glsl", "fragment_shader.glsl");
	shader.compile();
	
	Frame frame(m_width, m_height);
	
	Camera camera;
	float angle[3]{0};
	float speed{3.0};
	float last_time = glfwGetTime();
	while (!glfwWindowShouldClose(m_window))
	{
		
		
		glfwPollEvents();
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		{
			ImGui::Begin("Helios");
			ImGui::DragFloat3("Angle of rotation of Camera", angle, 0.005f, -3.14, 3.14f );
			
			if (ImGui::Button("Reset Orientation")){
				angle[0] = 0;
				angle[1] = 0;
				angle[2] = 0;
			}
			
			if (ImGui::Button("Reset Location")){
				camera.reset_location();
			}
			
			ImGui::SliderFloat("Camera Speed", &speed, 0.0f, 10.0f);
			
			ImGui::Text("Use w, a, s, d to move camera.");
			if(ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_W))){
				camera.update_key(Helios_Key::UP);
				ImGui::Text("Key Input : W (UP)");
			}
			
			else if(ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_A))){
				camera.update_key(Helios_Key::LEFT);
				ImGui::Text("Key Input : A (LEFT)");
			}
			
			else if(ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_S))){
				camera.update_key(Helios_Key::DOWN);
				ImGui::Text("Key Input : S (DOWN)");
			}
			
			else if(ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_D))){
				camera.update_key(Helios_Key::RIGHT);
				ImGui::Text("Key Input : D (RIGHT)");
			}
			else {
				ImGui::Text("Key Input : ");
			}
			float curr_time = glfwGetTime();
			float delta_time = curr_time - last_time;
			last_time = curr_time;
			
			/*
			Given that negative z-axis is pointing towards screen,
			 * x-axis is pointing up (along height of screen)
			 * y-axis is pointing right (along width of screen)
			 * I am passing x and y in place of each other to help users
			   maintain general notation of axes:
					1. x-axis along width
					2. y-axis along height
			*/
			camera.update(speed, delta_time, angle[1], angle[0], angle[2]);
			compute_frame(frame, camera);
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
		}
		
		ImGui::Render();
		
		int display_w, display_h;
		glfwGetFramebufferSize(m_window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(m_clear_color.x * m_clear_color.w, m_clear_color.y * m_clear_color.w, m_clear_color.z * m_clear_color.w, m_clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		
		glBindTexture(GL_TEXTURE_2D, frame.get_texture_ID());
		
		shader.use();
		quad.draw();
		
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(m_window);
	}
}
