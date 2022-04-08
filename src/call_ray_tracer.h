//
// Created by aditya on 07/04/22.
//

#ifndef HELIOS_TEMP_H
#define HELIOS_TEMP_H
#include <cuda_runtime.h>
extern void launch_kernel(cudaSurfaceObject_t surface, int textureWidth, int textureHeight, int i);
#endif //HELIOS_TEMP_H
