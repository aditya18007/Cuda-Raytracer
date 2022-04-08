
#include <cstdio>
#include <cstdint>
#include "Frame.h"
using namespace std;


__global__ void ray_trace(cudaSurfaceObject_t surface, int textureWidth, int textureHeight)
{
	
	//pixel index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= textureWidth)
		return;
	if(y >= textureHeight)
		return;
	
	//Calculate the pretty colors
	float red = blockIdx.x*5;
	float green = blockIdx.y*8;
	float blue = 0.0f;
	
	//Convert each value to an unsigned byte
	uchar4 pixel = { (uint8_t)(red),
	                 (uint8_t)(green),
	                 (uint8_t)(blue*255),
	                 (uint8_t)(1.0*255)};
	
	//Write to the surface.
	surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

void compute_frame(Frame& frame){
	
	int width = frame.get_width();
	int height = frame.get_height();
	auto surface = frame.get_bitmap_surface();
	
	int num_threads_x = 32;
	int num_threads_y = 32;
	dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);
	
	
	int num_blocks_x = (width / num_threads_x) + 1;
	int num_blocks_y = (height / num_threads_y) + 1;
	
	dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1);
	
	ray_trace<<<grid_shape, block_shape>>>(surface, width, height);
	cudaDeviceSynchronize();
}