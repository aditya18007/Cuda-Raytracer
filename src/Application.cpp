//
// Created by aditya on 07/04/22.
//

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
#include "Cuda_utils.h"
#include <iostream>
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

void Application::load_model(Object_Loader &loader) {
	const auto& meshes = loader.get_meshes();
	int start_vertex = 0;
	int start_face = 0;
	int start = 0;
	
	int min_pos = -1;
	int max_count = INT_MAX;
	for(auto& mesh : meshes){
		auto& vertices = mesh.vertices;
		auto& indices = mesh.indices;
		struct Mesh_Positions positions{};
		
		positions.start_vertices = start_vertex;
		positions.num_vertices = vertices.size();
		start_vertex += positions.num_vertices;
		
		positions.start_indices = start_face;
		positions.num_indices = indices.size();
		start_face += positions.num_indices;
		
		m_positions.push_back(positions);
		m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());
		m_indices.insert(m_indices.end(), indices.begin(), indices.end());
	}
	
	float min_x{std::numeric_limits<float>::max()};
	float min_y{std::numeric_limits<float>::max()};
	float min_z{std::numeric_limits<float>::max()};
	
	float max_x{std::numeric_limits<float>::min()};
	float max_y{std::numeric_limits<float>::min()};
	float max_z{std::numeric_limits<float>::min()};
	
	for(auto& vertex: m_vertices){
		max_x = std::max(max_x, vertex.Position.x);
		max_y = std::max(max_y, vertex.Position.y);
		max_z = std::max(max_z, vertex.Position.z);
		
		min_x = std::min(min_x, vertex.Position.x);
		min_y = std::min(min_y, vertex.Position.y);
		min_z = std::min(min_z, vertex.Position.z);
	}
	
	auto deno_x = max_x - min_x;
	auto deno_y = max_y - min_y;
	auto deno_z = max_z - min_z;
	
	for(auto& vertex: m_vertices){
		vertex.Position.x = (vertex.Position.x - min_x) / deno_x;
		vertex.Position.y = (vertex.Position.y - min_y) / deno_y;
		vertex.Position.z = (vertex.Position.z - min_z) / deno_z;
	}
	
	std::cout << "Number of Meshes = " << m_positions.size() << std::endl;
	std::cout << "Number of Indices = " << m_indices.size() << std::endl;
	std::cout << "Number of Vertices = " << m_vertices.size() << std::endl;
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
	float speed{1.0};
	float last_time = glfwGetTime();
	camera.set_target(m_vertices[0].Position);
	camera.set_position(m_vertices[1].Position);
	
	GPU_array<struct Mesh_Positions> d_positions(m_positions.data(), m_positions.size());
	GPU_array<struct Vertex> d_vertices(m_vertices.data(), m_vertices.size());
	GPU_array<unsigned int> d_indices(m_indices.data(), m_indices.size());
	
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
		
			camera.update(speed, delta_time, angle[0], angle[1], angle[2]);
			ImGui::Text("Camera Position = (%f,%f,%f)", camera.get_camera_position().x,camera.get_camera_position().y,camera.get_camera_position().z );
			ImGui::Text("Camera Target = (%f,%f,%f)", camera.get_camera_target().x,camera.get_camera_target().y,camera.get_camera_target().z );
			compute_frame(frame, camera,
						  d_positions.arr(), d_positions.get_size(),
						  d_vertices.arr(),d_vertices.get_size(),
						  d_indices.arr(), d_indices.get_size()
						  );
			
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
