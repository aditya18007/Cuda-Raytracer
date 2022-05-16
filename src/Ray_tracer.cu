
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <climits>
#include "Frame.h"
#include "Camera.h"
#include "glm/geometric.hpp"
#include "Dimensions.h"
#include "Object_Loader.h"
#include "Application.h"
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

__device__ bool intersect_triangle(
		Ray& r, Triangle tri,
		float &t)
{
    const glm::vec3 edge1 = tri.b - tri.a;
    const glm::vec3 edge2 = tri.c - tri.a;
    const glm::vec3 h = glm::cross( r.direction, edge2 );
    const float a = glm::dot( edge1, h );
    if (a > -0.0001f && a < 0.0001f) return false; // ray parallel to triangle
    const float f = 1 / a;
    const glm::vec3 s = r.origin - tri.a;
    const float u = f * glm::dot( s, h );
    if (u < 0 || u > 1) return false;
    const glm::vec3 q = cross( s, edge1 );
    const float v = f * dot( r.direction, q );
    if (v < 0 || u + v > 1) return false;
    float t_poss = f * dot( edge2, q );
    if (t > 0.0001f) t = min( t_poss, t );
    return true;
}


__global__ void ray_trace(cudaSurfaceObject_t surface, const glm::vec3 camera_pos, glm::vec3 u, glm::vec3 v, glm::vec3 dir, Triangle* d_triangles, int n_triangles)
{

    //pixel index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= WIDTH)
        return;
    if(y >= HEIGHT)
        return;

    float SMALLEST_DIST = 1e-4;
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

    for(int i = 0; i < n_triangles; i++){
        Triangle triangle = d_triangles[i];
        intersect_triangle(r,triangle, t);
    }
    if (t - FLT_MAX > 1.0f || FLT_MAX - t > 1.0f){
        uchar4 pixel = { (uint8_t)(m.color.x*255),
                         (uint8_t)(m.color.y*255),
                         (uint8_t)(m.color.z*255),
                         (uint8_t)(1.0*255)};
        surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
        return;
    }
    uchar4 pixel = { (uint8_t)(world.bgcolor.x*255),
                     (uint8_t)(world.bgcolor.y*255),
                     (uint8_t)(world.bgcolor.z*255),
                     (uint8_t)(1.0*255)};
    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

extern void compute_frame(Frame& frame, const Camera& camera, Triangle* d_triangles, int n_triangles){
    auto surface = frame.get_bitmap_surface();

    int num_threads_x = 32;
    int num_threads_y = 32;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);


    int num_blocks_x = ( WIDTH / num_threads_x) + 1;
    int num_blocks_y = ( HEIGHT / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1);

    ray_trace<<<grid_shape, block_shape>>>(surface, camera.get_camera_position(), camera.get_u(), camera.get_v(), camera.get_dir(), d_triangles, n_triangles);
    cudaDeviceSynchronize();
}