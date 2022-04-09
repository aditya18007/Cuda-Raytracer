
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <climits>
#include "Frame.h"
#include "Camera.h"
#include "glm/geometric.hpp"
#include "Dimensions.h"

using namespace std;

struct World{
	glm::vec3 bgcolor;
};

struct Material{
	glm::vec3 color;
};

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct Sphere {
	glm::vec3 origin;
	float radius;
	Material m;
};

__device__ bool intersect(Ray r, Sphere s) {
	float a = glm::dot(r.direction,r.direction);
	float b = glm::dot(r.direction, (r.origin-s.origin)*2.0f );
	float c = glm::dot(s.origin, s.origin) + dot(r.origin,r.origin) +-2.0*dot(r.origin,s.origin) - (s.radius*s.radius);
	
	if (b*b < 4*a*c){
		return false;
	}
	return true;
}
__global__ void ray_trace(cudaSurfaceObject_t surface, const glm::vec3 camera_target, const glm::vec3 camera_pos)
{
	
	//pixel index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= WIDTH)
		return;
	if(y >= HEIGHT)
		return;

	glm::vec3 camera_up(0.0, 1.0, 0.0);
	
	float aspect = 2;
	
	float SMALLEST_DIST = 1e-4;
	float FLT_MAX =  3.402823466e+38;
	float t = FLT_MAX;
	
	World world;
	world.bgcolor = glm::vec3(0.28, 0.28, 0.28);

	//	glm::vec3 line_of_sight = camera_target - camera_pos;
	//	glm::vec3 w = -line_of_sight;
	glm::vec3 w = -glm::normalize(camera_pos - camera_target);
	glm::vec3 u = glm::normalize(glm::cross(camera_up, w));
	glm::vec3 v = glm::normalize(cross(w, u));
	


	glm::vec3 dir = -w * 1.2071067811865475f;
	
	float xw = 0.0011111111111111111f*x - 0.9994444444444445;
	//450 is Height/2
	float yw = (y - 450.0f)/HEIGHT + 0.0005555555555555556;
	
	dir += u * xw;
	dir += v * yw;
	
	Ray r{};
	r.origin = camera_pos;
	r.direction = normalize(dir);
	
	Material m{};
	m.color =  glm::vec3(0.1, 0.7, 0.0);
	Sphere s{};
	s.origin = glm::vec3(2, 0, -10);
	s.radius = 3;
	s.m = m;
	
	if (intersect(r,s)){
		uchar4 pixel = { (uint8_t)(s.m.color.x*255),
		                 (uint8_t)(s.m.color.y*255),
		                 (uint8_t)(s.m.color.z*255),
		                 (uint8_t)(1.0*255)};
		surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
		
	}
	else{
		uchar4 pixel = { (uint8_t)(world.bgcolor.x*255),
		                 (uint8_t)(world.bgcolor.y*255),
		                 (uint8_t)(world.bgcolor.z*255),
		                 (uint8_t)(1.0*255)};
		surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
	}
}

void compute_frame(Frame& frame, const Camera& camera){
	

	auto surface = frame.get_bitmap_surface();
	
	int num_threads_x = 32;
	int num_threads_y = 32;
	dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);
	
	
	int num_blocks_x = ( WIDTH / num_threads_x) + 1;
	int num_blocks_y = ( HEIGHT / num_threads_y) + 1;
	
	dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1);
	
	ray_trace<<<grid_shape, block_shape>>>(surface, camera.get_camera_target(), camera.get_camera_position());
	cudaDeviceSynchronize();
}