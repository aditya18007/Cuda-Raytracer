#ifndef MY_GPU_UTILS_H
#define MY_GPU_UTILS_H


#include <cuda_runtime.h>
#include <iostream>

extern cudaError_t my_errno ;

#define SAFE_CALL( f_call, ... ) \
    my_errno = f_call(__VA_ARGS__); \
    if ( my_errno != cudaSuccess ){ \
        std::cerr << "\nFile : " << __FILE__ << '\n'; \
        std::cerr << "Function : " << __func__ << '\n'; \
        std::cerr << "Line : " << __LINE__ << '\n'; \
        std::cerr << "Cuda error : " << cudaGetErrorString(my_errno) << " !\n\n"; \
        exit(EXIT_FAILURE);\
    } \

template<typename T>
class GPU_array;



template<typename T>
class GPU_array{

    const size_t m_size;
    T* m_data;

public:
    GPU_array(size_t n): m_size(n) {
        SAFE_CALL(  cudaMalloc, (void**)&m_data , m_size * sizeof(T)  )
    }
	

    GPU_array(T* raw_ptr , size_t size): m_size(size) {
        SAFE_CALL( cudaMalloc, (void**)&m_data , m_size * sizeof(T) )
        SAFE_CALL( cudaMemcpy, m_data, raw_ptr, m_size*sizeof(T), cudaMemcpyHostToDevice )
    }
	
	~GPU_array(){
		SAFE_CALL(  cudaFree, m_data )
	}
	
    void write_to_host_ptr(T* ptr){
        SAFE_CALL( cudaMemcpy, ptr, m_data, m_size* sizeof(T), cudaMemcpyDeviceToHost )
    }
	
    T* arr(){
        return m_data;
    }
    size_t get_size(){
        return m_size;
    }
    size_t get_size_bytes() {
        return m_size*sizeof(T);
    }
};

#endif //MY_GPU_UTILS_H