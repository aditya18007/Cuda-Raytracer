#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>

cudaError_t my_errno ;
#define SAFE_CALL( f_call, ... ) \
    my_errno = f_call(__VA_ARGS__); \
    if ( my_errno != cudaSuccess ){ \
        std::cerr << "File : " << __FILE__ << '\n'; \
        std::cerr << "Function : " << __func__ << '\n'; \
        std::cerr << "Line : " << __LINE__ << '\n'; \
        std::cerr << "Cuda error : " << cudaGetErrorString(my_errno) << " !\n\n"; \
        exit(EXIT_FAILURE);\
    } \

template<typename T>
class GPU_array;

template<typename T>
class CPU_array;

template<typename T>
class CPU_array{
	
	const size_t m_size;
	T* m_data;

public:
	CPU_array(size_t n)
			: m_size(n), m_data( new T[n])
	{}
	
	CPU_array(GPU_array<T>& gpu_arr)
			: m_size(gpu_arr.get_size()), m_data( new T[gpu_arr.get_size()])
	{
		SAFE_CALL( cudaMemcpy, m_data, gpu_arr.arr(), m_size* sizeof(T), cudaMemcpyDeviceToHost )
	}
	
	~CPU_array(){
		delete [] m_data;
	}
	
	T* arr(){
		return m_data;
	}
	
	T& operator()(int i){
		return m_data[i];
	}
	
	size_t get_size() {
		return m_size;
	}
};

template<typename T>
class GPU_array{
	
	const size_t m_size;
	T* m_data;

public:
	GPU_array(size_t n): m_size(n) {
		SAFE_CALL(  cudaMalloc, (void**)&m_data , m_size * sizeof(T)  )
	}
	
	GPU_array(CPU_array<T>& cpu_arr ): m_size(cpu_arr.get_size()) {
		SAFE_CALL( cudaMalloc, (void**)&m_data , m_size * sizeof(T) )
		SAFE_CALL( cudaMemcpy, m_data, cpu_arr.arr(), m_size*sizeof(T), cudaMemcpyHostToDevice )
	}
	
	~GPU_array(){
		SAFE_CALL(  cudaFree, m_data )
	}
	T* arr(){
		return m_data;
	}
	size_t get_size(){
		return m_size;
	}
};


#define LENGTH 100000000

__global__
void add(int *a, int *b, int *result){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < LENGTH){
		result[i] = a[i] + b[i];
	}
}

constexpr size_t length{LENGTH};
constexpr size_t num_threads = 256;
constexpr size_t num_blocks = (length+num_threads)/num_threads;

int launch_kernel(){
	
	cudaEvent_t start, stop;
	
	SAFE_CALL( cudaEventCreate, &start )
	SAFE_CALL( cudaEventCreate, &stop )
	float gpu_time_ms;
	
	CPU_array<int> a(length), b(length), c(length);
	
	for(int i = 0; i < length; i++){
		a(i) = i;
		b(i) = i;
	}
	
	
	GPU_array<int> d_a(a), d_b(b), d_c(length);
	
	SAFE_CALL( cudaEventRecord, start )
	
	add<<<num_blocks,num_threads>>>( d_a.arr(), d_b.arr(), d_c.arr() );
	
	SAFE_CALL( cudaEventRecord, stop )
	SAFE_CALL( cudaDeviceSynchronize )
	SAFE_CALL( cudaMemcpy, c.arr(), d_c.arr(), length* sizeof(int), cudaMemcpyDeviceToHost )
	SAFE_CALL( cudaEventSynchronize, stop )
	
	for(int i = 0; i < length; i++){
		auto expected = a(i) + b(i);
		auto got = c(i);
		if( got != expected ){
			std::cerr << "GPU add FAILED!\n";
			std::cerr <<  "Output = " << a(i) << " + " << b(i) << " = " << c(i) << "\n";
			exit(EXIT_FAILURE);
		}
	}
	for(int i=0 ; i< LENGTH/100000 ; i++){
		std::cout << c(i) << std::endl;
	}
	SAFE_CALL( cudaEventElapsedTime, &gpu_time_ms, start, stop )
	std::cout << "Success! Time elapsed on GPU = " << gpu_time_ms << " milliseconds."<< std::endl;
	
	std::ofstream outfile ("time_taken.txt");
	outfile << "Time elapsed on GPU = " << gpu_time_ms << " milliseconds."<< std::endl;
	outfile.close();
}