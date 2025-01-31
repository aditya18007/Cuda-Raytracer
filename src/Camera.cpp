//
// Created by aditya on 08/04/22.
//

#include "glm/glm.hpp"
#include "glm/geometric.hpp"
#include "Camera.h"
#include "glm/gtx/rotate_vector.hpp"

Camera::Camera()
: lookAt(0, 0, 0), lookFrom(3, 3, 3),
original_lookAt(0,0,0), original_lookFrom(3,3,3),
current_key(Helios_Key::NONE),
angle_x(0), angle_y(0), angle_z(0)
{}


glm::vec3 Camera::get_camera_position() const {
	return lookFrom;
}

glm::vec3 Camera::get_camera_target() const {
	return lookAt;
}

void Camera::update_key(Helios_Key key) {
	current_key = key;
}

void Camera::update( float movement_speed, float deltaTime ,  float new_x, float new_y, float new_z) {
	auto theta_x = new_x - angle_x;
	angle_x = new_x;
	
	auto theta_y = new_y - angle_y;
	angle_y = new_y;
	
	auto theta_z = new_z - angle_z;
	angle_z = new_z;
	
	auto direction = lookAt - lookFrom;
	direction = glm::rotateX(direction, theta_x);
	direction = glm::rotateY(direction, theta_y);
	direction = glm::rotateZ(direction, theta_z);
	direction = glm::normalize(direction);
	lookAt = lookFrom + direction;
	
	//Normalization is important so that forward-backward and left-right speed are same
	auto right = glm::normalize(glm::cross(direction, glm::vec3(0.0f,1.0f,0.0f)));
	
	if ( current_key == Helios_Key::UP){
		//Up
		auto delta_t =  direction* deltaTime * movement_speed;
		lookAt += delta_t;
		lookFrom += delta_t;
	}
	
	if (current_key == Helios_Key::DOWN){
		//Down
		auto delta_t = direction * deltaTime * movement_speed;
		lookAt -= delta_t;
		lookFrom -= delta_t;
	}
	
	if (current_key == Helios_Key::RIGHT){
		//Right
		auto delta_t = right * deltaTime * movement_speed;
		lookAt += delta_t;
		lookFrom += delta_t;
	}
	
	if (current_key == Helios_Key::LEFT){
		//Left
		auto delta_t = right * deltaTime * movement_speed;
		lookAt -= delta_t;
		lookFrom -= delta_t;
	}
	
	//See previous commit to see calculations in Cuda Kernel itself.
	//Reduced burden from there to here.
	u = -right;
	v = glm::normalize(cross(direction, u));
	dir = direction * 1.2071067811865475f;
	current_key = Helios_Key::NONE;
}


void Camera::reset_location() {
	lookFrom = original_lookFrom;
	lookAt = original_lookAt;
	angle_x = 0;
	angle_y = 0;
	angle_z = 0;
}

glm::vec3 Camera::get_u() const {
	return u;
}

glm::vec3 Camera::get_v() const {
	return v;
}

glm::vec3 Camera::get_dir() const {
	return dir;
}

void Camera::set_position(const glm::vec3 &new_pos) {
	this->lookFrom = new_pos;
	this->original_lookFrom = new_pos;
}

void Camera::set_target(const glm::vec3 &new_target) {
	this->lookAt = new_target;
	this->original_lookAt = new_target;
}

