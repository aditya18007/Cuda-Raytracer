
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <climits>
#include "Frame.h"
#include "Camera.h"
#include "glm/geometric.hpp"
#include "Dimensions.h"
#include "Object_Loader.h"

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

struct Triangle{
	glm::vec3 a;
	glm::vec3 b;
	glm::vec3 c;
};
__device__ bool intersect_triangle(
		Ray& r, Triangle tri,
		float &t)
{
	
	const glm::vec3 v0 = tri.a;
	const glm::vec3 v1 = tri.b;
	const glm::vec3 v2 = tri.c;
	
	// compute plane's normal
	auto v0v1 = v1 - v0;
	auto v0v2 = v2 - v0;
	// no need to normalize
	auto N = glm::cross(v0v1, v0v2);
	float area2 = glm::length(N);
	
	// Step 1: finding P
	
	// check if ray and plane are parallel ?
	float NdotRayDirection = glm::dot(N, r.direction);
	float kEpsilon = 1e-8;
	if (fabs(NdotRayDirection) < kEpsilon) // almost 0
		return false; // they are parallel so they don't intersect !
	
	// compute d parameter using equation 2
	float d = -glm::dot(N, v0);
	
	// compute t (equation 3)
	t = -(glm::dot(N, r.origin) + d) / NdotRayDirection;
	
	// check if the triangle is in behind the ray
	if (t < 0) return false; // the triangle is behind
	
	// compute the intersection point using equation 1
	glm::vec3 P = r.origin + t * r.direction;
	
	// Step 2: inside-outside test
	glm::vec3 C; // vector perpendicular to triangle's plane
	
	// edge 0
	auto edge0 = v1 - v0;
	auto vp0 = P - v0;
	C = glm::cross(edge0, vp0);
	if (glm::dot(N, C) < 0) return false; // P is on the right side
	
	// edge 1
	auto edge1 = v2 - v1;
	auto vp1 = P - v1;
	C = glm::cross(edge1,vp1);
	if ( glm::dot(N, C) < 0)  return false; // P is on the right side
	
	// edge 2
	auto edge2 = v0 - v2;
	auto vp2 = P - v2;
	C = glm::cross(edge2, vp2);
	if (glm::dot(N, C) < 0) return false; // P is on the right side;
	
	return true; // this ray hits the triangle
}

__device__ bool intersect_sphere(Ray& r, Sphere& s) {
	float a = glm::dot(r.direction,r.direction);
	float b = glm::dot(r.direction, (r.origin-s.origin)*2.0f );
	float c = glm::dot(s.origin, s.origin) + glm::dot(r.origin,r.origin) +-2.0*glm::dot(r.origin,s.origin) - (s.radius*s.radius);
	
	if (b*b < 4*a*c){
		return false;
	}
	return true;
}
__global__ void ray_trace(cudaSurfaceObject_t surface, const glm::vec3 camera_pos, glm::vec3 u, glm::vec3 v, glm::vec3 dir, Mesh_Positions* d_positions, int n_positions, Vertex* d_vertices, int n_vertices, unsigned int * d_indices, int n_indices)
{
	
	//pixel index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= WIDTH)
		return;
	if(y >= HEIGHT)
		return;
	
	float SMALLEST_DIST = 1e-4;
//	float FLT_MAX =  3.402823466e+38;
	float t = FLT_MAX;
	
	World world;
	world.bgcolor = glm::vec3(0.28, 0.28, 0.28);

	
	float xw = 0.0011111111111111111f*x - 0.9994444444444445;
	float yw = (y - HEIGHT/2.0f)/HEIGHT + 0.0005555555555555556;
	
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
	
	for(int i = 0; i < n_positions; i++){
		int start_index = d_positions[i].start_indices;
		int num_indices = d_positions[i].num_indices;

		int start_vertex = d_positions[i].start_vertices;
		int num_vertices = d_positions[i].num_vertices;
		for(int idx = start_index; idx < start_index + num_indices; idx += 3){
			auto a_pos =  start_vertex + d_indices[idx];
			auto b_pos = start_vertex + d_indices[idx+1];
			auto c_pos = start_vertex + d_indices[idx+2];
			Triangle triangle;
			triangle.a = d_vertices[a_pos].Position;
			triangle.b = d_vertices[b_pos].Position;
			triangle.c = d_vertices[c_pos].Position;
			if (intersect_triangle(r,triangle, t)){
				uchar4 pixel = { (uint8_t)(s.m.color.x*255),
				                 (uint8_t)(s.m.color.y*255),
				                 (uint8_t)(s.m.color.z*255),
				                 (uint8_t)(1.0*255)};
				surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
				return;
			}
		}
	}
	uchar4 pixel = { (uint8_t)(world.bgcolor.x*255),
	                 (uint8_t)(world.bgcolor.y*255),
	                 (uint8_t)(world.bgcolor.z*255),
	                 (uint8_t)(1.0*255)};
	surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

void compute_frame(Frame& frame, const Camera& camera, Mesh_Positions* d_positions, int n_positions, Vertex* d_vertices, int n_vertices, unsigned int * d_indices, int n_indices){
	

	auto surface = frame.get_bitmap_surface();
	
	int num_threads_x = 32;
	int num_threads_y = 32;
	dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);
	
	
	int num_blocks_x = ( WIDTH / num_threads_x) + 1;
	int num_blocks_y = ( HEIGHT / num_threads_y) + 1;
	
	dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1);
	
	ray_trace<<<grid_shape, block_shape>>>(surface, camera.get_camera_position(), camera.get_u(), camera.get_v(), camera.get_dir(), d_positions, n_positions, d_vertices, n_vertices, d_indices, n_indices);
	cudaDeviceSynchronize();
}