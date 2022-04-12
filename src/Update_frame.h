//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_TEMP_H
#define HELIOS_TEMP_H
#include "Frame.h"
#include "Camera.h"

extern void compute_frame(Frame& frame, const Camera& camera, Mesh_Positions* d_positions, int n_positions, Vertex* d_vertices, int n_vertices, unsigned int * d_indices, int n_indices);
#endif //HELIOS_TEMP_H
