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
	UPDATE_NEEDED,
	STILL
};

class Camera {
	const int m_width;
	const int m_height;
	
	float mouse_x, mouse_y;
	glm::vec3 position = glm::vec3( 0, 0, 5 );
	float horizontalAngle = 3.14f;
	float verticalAngle = 0.0f;
	float initialFoV = 45.0f;
	
	float speed = 3.0f; // 3 is optimal. Any faster and it gets messy
	float mouseSpeed = 0.1f;
	double lastTime{};
	
	Helios_Key curr_key{};
	
	glm::vec3 m_camPos{};
	glm::vec3 m_camTarget{};
	
	Camera_State camera_state{Camera_State::STILL};
public:
	Camera(int width, int height);
	void update_mouse_pos(float x, float y);
	void update_keyboard(Helios_Key key);
	void update();
	
	glm::vec3 get_camera_position();
	glm::vec3 get_camera_target();
};


#endif //HELIOS_CAMERA_H
