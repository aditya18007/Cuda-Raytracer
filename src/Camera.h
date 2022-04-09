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
	int m_width;
	int m_height;
	
	float mouse_x, mouse_y;
	float horizontalAngle;
	float verticalAngle;
	
	float speed; // 3 is optimal. Any faster and it gets messy
	float mouseSpeed;
	double lastTime;
	
	Helios_Key curr_key{};
	
	glm::vec3 m_camPos;
	glm::vec3 m_camTarget;
	
	Camera_State camera_state_orientation{Camera_State::STILL};
	Camera_State camera_state_position{Camera_State::STILL};
public:
	Camera(int width, int height);
	Camera(glm::vec3 position, glm::vec3 target);
	void update_mouse_pos(float x, float y);
	void update_keyboard(Helios_Key key);
	void update();
	
	glm::vec3 get_camera_position() const;
	glm::vec3 get_camera_target() const;
};


#endif //HELIOS_CAMERA_H
