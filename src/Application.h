//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_APPLICATION_H
#define HELIOS_APPLICATION_H

#include <string>
#include <GLFW/glfw3.h>

class Application {
	const int m_width;
	const int m_height;
	static constexpr char const * m_title = "Helios";
	static constexpr char const * m_glsl_version = "#version 330";
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
