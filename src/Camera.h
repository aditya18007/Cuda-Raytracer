//
// Created by aditya on 08/04/22.
//

#ifndef HELIOS_CAMERA_H
#define HELIOS_CAMERA_H

#include <glm/vec3.hpp>
#include "glm/ext/matrix_float4x4.hpp"
#include "Dimensions.h"

enum class Helios_Key{
	NONE,
	UP,
	LEFT,
	DOWN,
	RIGHT
};


class Camera {
	
	glm::vec3 lookFrom;
	glm::vec3 lookAt;
	
	Helios_Key current_key;
	
	float angle_x, angle_y, angle_z;
public:
	Camera();
	
	void update_key(Helios_Key key);
	
	void update(float movement_speed, float deltaTime, float theta_x, float theta_y, float theta_z);
	glm::vec3 get_camera_position() const;
	glm::vec3 get_camera_target() const;
	
};


#endif //HELIOS_CAMERA_H
