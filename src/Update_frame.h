//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_TEMP_H
#define HELIOS_TEMP_H
#include "Frame.h"
#include "Camera.h"

extern void compute_frame(Frame& frame, const Camera& camera, Triangle* d_triangles, int n_triangles);
#endif //HELIOS_TEMP_H
