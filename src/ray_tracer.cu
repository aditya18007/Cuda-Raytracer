
#include <cstdio>
#include <cstdint>

using namespace std;


__global__ void update_surface(cudaSurfaceObject_t surface, int textureWidth, int textureHeight, int i)
{
	//Get the pixel index
	int xPx = threadIdx.x + blockIdx.x * blockDim.x;
	int yPx = threadIdx.y + blockIdx.y * blockDim.y;
	//Don't do any computation if this thread is outside of the surface bounds.
	if(xPx >= textureWidth)
		return;
	if(yPx >= textureHeight)
		return;
	
	
	
	//Calculate the pretty colors
	float red = 0.8f;
	float green = 0.0f;
	float blue = 0.0f;
	float alpha = 1.f;
	
	if (xPx < textureWidth/2){
		red = 0.0f;
	}
	if (yPx > textureHeight/2){
		red = 0.0f;
	}
	//Convert each value to an unsigned byte
	uchar4 pixel = { (uint8_t)(red*255),
	                 (uint8_t)(green*255),
	                 (uint8_t)(blue*255),
	                 (uint8_t)(alpha*255)};
	
	//Write to the surface.
	surf2Dwrite(pixel, surface, xPx * sizeof(uchar4), yPx);
}

void launch_kernel(cudaSurfaceObject_t surface, int width, int height, int i){
	int num_threads_x = 32;
	int num_threads_y = 32;
	dim3 block_shape = dim3( num_threads_x, num_threads_y ,1);
	
	int num_blocks_x = (width / num_threads_x) + 1;
	int num_blocks_y = (height / num_threads_y) + 1;
	
	dim3 grid_shape = dim3( num_blocks_x, num_blocks_y , 1);
	
	update_surface<<<grid_shape, block_shape>>>(surface, width, height, i);
	cudaDeviceSynchronize();
}