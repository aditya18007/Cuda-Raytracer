//
// Created by aditya on 16/5/22.
//

#ifndef HELIOS_BVH_H
#define HELIOS_BVH_H
#include <limits>
#include <cfloat>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <cmath>

class Bounding_Box{
public:
    float min_x{FLT_MAX}, min_y{FLT_MAX}, min_z{FLT_MAX};
    float max_x{FLT_MIN}, max_y{FLT_MIN}, max_z{FLT_MIN};

    void update(glm::vec3 point){
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        min_z = std::min(min_z, point.z);

        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
        max_z = std::max(max_z, point.z);
    }

    float area(){
        float a = max_x-min_x;
        float b = max_y-min_y;
        float c = max_z-min_z;
        return a*b + b*c + c*a;
    }

    glm::vec3 normalize01(glm::vec3 point){
        point.x = (point.x - min_x) / (max_x - min_x);
        point.y = (point.y - min_y) / (max_y - min_y);
        point.z = (point.z - min_z) / (max_z - min_z);
        return point;
    }
};

class Triangle{

public:
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
    glm::vec3 centroid;
    glm::vec3 normal;
    __device__ __host__ Triangle(glm::vec3& v0, glm::vec3& v1, glm::vec3& v2 ){
        a = v0;
        b = v1;
        c = v2;
        centroid = (a+b+c)/(3.0f);
        normal = glm::normalize( glm::cross(b-a, c-a) );
    }
};

class BVH_node {
public:
    float min_x{FLT_MAX}, max_x{FLT_MIN};
    float min_y{FLT_MAX}, max_y{FLT_MIN};
    float min_z{FLT_MAX}, max_z{FLT_MIN};

    int left_node{-1};

    int start_idx{0};
    int prim_count{0};

    __device__ __host__ bool is_leaf() const{
        return prim_count > 0;
    }
};

void print_BVH_node(const BVH_node& node);

class BVH_tree{
    const int N;
    std::vector<BVH_node> m_tree;
    const std::vector<Triangle>& m_triangles;
    std::vector<int> m_triangle_indices;
    int nodes_used{0};
    void create_root();
    void update_bbox(int idx);
    void recurse(int idx);
    float score(BVH_node& node, int axis, float pos);
public:
    BVH_tree(const std::vector<Triangle>& triangles);
    std::vector<BVH_node> create_tree();
    std::vector<int> get_indices(){
        return m_triangle_indices;
    }
};



#endif //HELIOS_BVH_H
