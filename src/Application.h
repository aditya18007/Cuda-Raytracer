//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_APPLICATION_H
#define HELIOS_APPLICATION_H

#include <string>
#include <GLFW/glfw3.h>
#include "imgui.h"

class Application {
	const int m_width;
	const int m_height;
	static constexpr char const * m_title = "Helios";
	static constexpr char const * m_glsl_version = "#version 330";
	const ImVec4 m_clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	GLFWwindow* m_window;
	
private:
	void init_window();
public:
	Application(int w, int h);
	Application();
	void run();
	~Application();
};


#endif //HELIOS_APPLICATION_H
