
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
    float k_a;
    glm::vec3 c_a;

    float k_d;
    glm::vec3 c_r;

    float k_s;
    glm::vec3 c_p;
    int n;
};

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
    glm::vec3 normal;
    float t;
};

__device__ bool intersect_triangle(
		Ray& r, Triangle tri)
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
    if (r.t > 0.0001f) {
        if (t_poss < r.t){
            r.t = t_poss;
            r.normal = tri.normal;
        }
    }

    return true;
}

__device__ bool intersect_bbox(Ray r, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z ){

    float tx1 = (min_x - r.origin.x) / r.direction.x, tx2 = (max_x - r.origin.x) / r.direction.x;
    float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );

    float ty1 = (min_y - r.origin.y) / r.direction.y, ty2 = (max_y - r.origin.y) / r.direction.y;
    tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );

    float tz1 = (min_z - r.origin.z) / r.direction.z, tz2 = (max_z - r.origin.z) / r.direction.z;
    tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );

    return tmax >= tmin && tmin < r.t && tmax > 0;
}

__global__ void ray_trace(cudaSurfaceObject_t surface, const glm::vec3 camera_pos, glm::vec3 u, glm::vec3 v, glm::vec3 dir, const Triangle* d_triangles, const int* d_triangle_indices, int n_triangles, const BVH_node* d_traversal_tree, int d_traversal_tree_size)
{

    //pixel index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= WIDTH)
        return;
    if(y >= HEIGHT)
        return;


    World world;
    world.bgcolor = glm::vec3(0.28, 0.28, 0.28);


    float xw = 0.0011111111111111111f*x - 0.9994444444444445;
    float yw = (y - HEIGHT/2.0f)/HEIGHT + 0.0005555555555555556;

    dir += u * xw;
    dir += v * yw;

    Ray r{};
    r.origin = camera_pos;
    r.direction = normalize(dir);
    r.t = FLT_MAX;
    Material m{};
    m.k_a = 0.3;
    m.c_a = glm::vec3(1.0f, 0.43f, 0.14f);

    m.k_d = 0.8;
    m.c_r = glm::vec3(1.0f, 0.43f, 0.14f);

    m.k_s = 0.3;
    m.c_p = glm::vec3(1.0f, 1.0f, 1.0f);
    m.n = 22;

    int stack[1024];
    int stack_ptr = 0;
    stack[stack_ptr] = 0;
    while(stack_ptr >= 0){
        int node_idx = stack[stack_ptr];
        const BVH_node& node = d_traversal_tree[node_idx];
        if ( !intersect_bbox( r, node.min_x, node.min_y, node.min_z, node.max_x, node.max_y, node.max_z )){
            stack_ptr--;
            continue;
        }
        if (node.is_leaf()){
            for (int i = 0; i < node.prim_count; i++ ){
                auto& tri = d_triangles[ d_triangle_indices[node.start_idx + i] ];
                intersect_triangle(r, tri);
            }
            stack_ptr--;
            continue;
        }
        stack[++stack_ptr] = node.left_node;
        stack[++stack_ptr] = node.left_node + 1;
    }
    if (r.t - FLT_MAX > 1.0f || FLT_MAX - r.t > 1.0f){
        glm::vec3 poi = r.origin + r.t*r.direction;
        auto v = glm::normalize(r.origin - poi);
        auto n = r.normal;
        auto l = camera_pos - poi;
        auto h = glm::normalize(v+l);

        float diff = max( glm::dot(n,l), 0.0f);
        auto diff_color = m.c_r*diff;

        float spec = pow( max( dot(n,h), 0.0f), 0.0f);
        auto spec_color = spec*m.c_p;

        auto L = m.k_a*m.c_r + m.k_d*diff_color + m.k_s*spec_color;
        uchar4 pixel = { (uint8_t)(L.x*255),
                         (uint8_t)(L.y*255),
                         (uint8_t)(L.z*255),
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

extern void compute_frame(Frame& frame, const Camera& camera, Triangle* d_triangles, int* d_triangle_indices, int n_triangles, BVH_node* d_traversal_tree, int d_traversal_tree_size){
    auto surface = frame.get_bitmap_surface();

    int num_threads_x = 32;
    int num_threads_y = 32;
    dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);


    int num_blocks_x = ( WIDTH / num_threads_x) + 1;
    int num_blocks_y = ( HEIGHT / num_threads_y) + 1;

    dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1);

    ray_trace<<<grid_shape, block_shape>>>(surface, camera.get_camera_position(), camera.get_u(), camera.get_v(), camera.get_dir(), d_triangles, d_triangle_indices,n_triangles, d_traversal_tree, d_traversal_tree_size);
    cudaDeviceSynchronize();
}