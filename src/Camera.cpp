//
// Created by aditya on 08/04/22.
//

#include <cmath>
#include <iostream>
#include <utility>
#include <cmath>
#include <glm/geometric.hpp>
#include "Camera.h"
#include "GLFW/glfw3.h"

Camera::Camera(int width, int height)
: m_width(width), m_height(height),
	lastTime(glfwGetTime()),
	camera_state_position(Camera_State::STILL),
	camera_state_orientation(Camera_State::STILL),
	m_camPos(0,0,5),
	m_camTarget(0,0,0),
	mouse_x(900), mouse_y(450),
	horizontalAngle(3.14f),
	verticalAngle(0.0f),
	speed(3.0f), mouseSpeed(0.1f)
{}

void Camera::update_mouse_pos(float x, float y) {
	
	mouse_x = x;
	mouse_y = y;
	camera_state_orientation = Camera_State::MOUSE_UPDATE_NEEDED;
}

void Camera::update() {
	
	double currentTime = glfwGetTime();
	auto deltaTime = float(currentTime - lastTime);
	lastTime = currentTime;
	if (camera_state_orientation == Camera_State::MOUSE_UPDATE_NEEDED){
		horizontalAngle += mouseSpeed * deltaTime * float(double(m_width)/2 - mouse_x ) ;
		verticalAngle   += mouseSpeed * deltaTime * float(double(m_height)/2 - mouse_y );
	}
	
	glm::vec3 right = glm::vec3(
			std::sin(horizontalAngle - 3.14f/2.0f),
			0,
			std::cos(horizontalAngle - 3.14f/2.0f)
	);
	
	glm::vec3 direction(
			std::cos(verticalAngle) * std::sin(horizontalAngle),
			std::sin(verticalAngle),
			std::cos(verticalAngle) * std::cos(horizontalAngle)
	);
//	glm::vec3 up = glm::cross( right, direction );
	glm::vec3 position = m_camPos;
	
	
		if ( curr_key == Helios_Key::UP){
			//Up
			position += direction * deltaTime * speed;
		}
		// Move backward
		if (curr_key == Helios_Key::DOWN){
			//Down
			position -= direction * deltaTime * speed;
		}
		// Strafe right
		if (curr_key == Helios_Key::RIGHT){
			//Right
			position += right * deltaTime * speed;
		}
		// Strafe left
		if (curr_key == Helios_Key::LEFT){
			//Left
			position -= right * deltaTime * speed;
		}
	
	m_camPos = position;
	m_camTarget =  position+direction;
	camera_state_position = Camera_State::STILL;
	camera_state_orientation = Camera_State::STILL;
	curr_key = Helios_Key::NONE;
}

void Camera::update_keyboard(Helios_Key key) {
	curr_key = key;
}

glm::vec3 Camera::get_camera_position() const{
	return m_camPos;
}

glm::vec3 Camera::get_camera_target() const{
	return m_camTarget;
}

Camera::Camera(glm::vec3 position, glm::vec3 target) {
	m_camPos = position;
	m_camTarget = target;
}
