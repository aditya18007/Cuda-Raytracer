//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_TEMP_H
#define HELIOS_TEMP_H
#include "Frame.h"
#include "Camera.h"

extern void compute_frame(Frame& frame, const Camera& camera, Triangle* d_triangles, int n_triangles);
extern void compute_frame(Frame& frame, const Camera& camera, Triangle* d_triangles, int* d_triangle_indices, int n_triangles, BVH_node* d_traversal_tree, int d_traversal_tree_size);
#endif //HELIOS_TEMP_H
