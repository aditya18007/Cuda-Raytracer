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
#include "glm/vec3.hpp"

enum class Axis{
    X,
    Y,
    Z
};

class Triangle{

public:
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
    glm::vec3 centroid;
    __device__ __host__ Triangle(glm::vec3& v0, glm::vec3& v1, glm::vec3& v2 ){
        a = v0;
        b = v1;
        c = v2;
        centroid = (a+b+c)/(3.0f);
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

    __device__ __host__ bool is_leaf(){
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
public:
    BVH_tree(const std::vector<Triangle>& triangles);
    std::vector<BVH_node> create_tree();
    std::vector<int> get_indices(){
        return m_triangle_indices;
    }
};



#endif //HELIOS_BVH_H
