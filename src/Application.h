//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_APPLICATION_H
#define HELIOS_APPLICATION_H

#include <string>
#include <GLFW/glfw3.h>
#include <crt/host_defines.h>
#include "imgui.h"
#include "Object_Loader.h"

class Triangle{

public:
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
    glm::vec3 centroid;
    __device__ __host__ Triangle(glm::vec3& v0, glm::vec3& v1, glm::vec3& v2 ){
        a = v0;
        b = v1;
        c = v2;
        centroid = (a+b+c)/(3.0f);
    }
};

class Application {
	const int m_width;
	const int m_height;
	static constexpr char const * m_title = "Helios";
	static constexpr char const * m_glsl_version = "#version 460";
	const ImVec4 m_clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	GLFWwindow* m_window;
	
	//Model
	std::vector<Mesh_Positions> m_positions;
	std::vector<Vertex> m_vertices;
	std::vector<unsigned int> m_indices;
	
private:
	void init_window();
public:
	Application(int w, int h);
	Application();
	void load_model(Object_Loader& loader);
	void run( );
	~Application();
};


#endif //HELIOS_APPLICATION_H
