//
// Created by aditya on 08/04/22.
//

#ifndef HELIOS_CAMERA_H
#define HELIOS_CAMERA_H

#include <glm/vec3.hpp>

enum class Helios_Key{
	NONE,
	UP,
	LEFT,
	DOWN,
	RIGHT
};

enum class Camera_State{
	MOUSE_UPDATE_NEEDED,
	KEYBOARD_UPDATE_NEEDED,
	STILL
};

class Camera {
	const int m_width;
	const int m_height;
	
	float mouse_x, mouse_y;
	float horizontalAngle = 3.14f;
	float verticalAngle = 0.0f;
	float initialFoV = 45.0f;
	
	float speed = 3.0f; // 3 is optimal. Any faster and it gets messy
	float mouseSpeed = 0.1f;
	double lastTime{};
	
	Helios_Key curr_key{};
	
	glm::vec3 m_camPos= glm::vec3(0.0f, 0.0f, 5.0f);
	glm::vec3 m_camTarget = glm::vec3(0.0f, 0.0f, 0.0f);
	
	Camera_State camera_state_orientation{Camera_State::STILL};
	Camera_State camera_state_position{Camera_State::STILL};
public:
	Camera(int width, int height);
	void update_mouse_pos(float x, float y);
	void update_keyboard(Helios_Key key);
	void update();
	
	glm::vec3 get_camera_position() const;
	glm::vec3 get_camera_target() const;
};


#endif //HELIOS_CAMERA_H
