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
#include "BVH.h"

#include <iostream>
static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

Application::Application() : m_width(1280), m_height(720), m_window(nullptr){
}

Application::Application(int w, int h) : m_width(w), m_height(h), m_window(nullptr) {
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
	
	std::cout << "Number of Meshes = " << m_positions.size() << std::endl;
	std::cout << "Number of Faces = " << m_indices.size()/3 << std::endl;
	std::cout << "Number of Vertices = " << m_vertices.size() << std::endl;


    for(int i = 0; i < m_positions.size(); i++){
        int start_index = m_positions[i].start_indices;
        int num_indices = m_positions[i].num_indices;

        int start_vertex = m_positions[i].start_vertices;
        int num_vertices = m_positions[i].num_vertices;
        for(int idx = start_index; idx < start_index + num_indices; idx += 3){
            auto a_pos =  start_vertex + m_indices[idx];
            auto b_pos = start_vertex + m_indices[idx+1];
            auto c_pos = start_vertex + m_indices[idx+2];
            Triangle triangle(m_vertices[a_pos].Position, m_vertices[b_pos].Position, m_vertices[c_pos].Position);
            m_triangles.push_back(triangle);
        }
    }

    BVH_tree tree(m_triangles);
    m_traversal_tree = tree.create_tree();
    m_triangle_indices = tree.get_indices();

    std::cout << "Number of nodes in BVH tree = " << m_traversal_tree.size() << '\n';
    this->init_window();
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


    GPU_array<Triangle> d_triangles(m_triangles.data(), m_triangles.size());
    GPU_array<int> d_triangle_indices(m_triangle_indices.data(), m_triangle_indices.size());
    GPU_array<BVH_node> d_traversal_tree(m_traversal_tree.data(), m_traversal_tree.size());

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
			
			ImGui::DragFloat("Camera Speed", &speed, 0.1f, 0.0f, 1000.0f);
			
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

//            compute_frame(frame, camera,
//						  d_triangles.arr(), d_triangles.get_size()
//						  );
            compute_frame(frame, camera,
                          d_triangles.arr(), d_triangle_indices.arr(), d_triangles.get_size(),
                          d_traversal_tree.arr(), d_traversal_tree.get_size());
			
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