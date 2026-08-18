# PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES AY 23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME:THOUFEEQ AHMED B</h3>
<h3>ENTER YOUR REGISTER NO:26018418</h3>
<h3>EX. NO:03</h3>
<h3>DATE:18/08/2026</h3>
<h1> <align=center> PARALLEL REDUCTION USING UNROLLING TECHNIQUES </h3>
  Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.</h3>

## AIM:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Initialization and Memory Allocation
2.	Define the input size n.
3.	Allocate host memory (h_idata and h_odata) for input and output data.
Input Data Initialization
4.	Initialize the input data on the host (h_idata) by assigning a value of 1 to each element.
Device Memory Allocation
5.	Allocate device memory (d_idata and d_odata) for input and output data on the GPU.
Data Transfer: Host to Device
6.	Copy the input data from the host (h_idata) to the device (d_idata) using cudaMemcpy.
Grid and Block Configuration
7.	Define the grid and block dimensions for the kernel launch:
8.	Each block consists of 256 threads.
9.	Calculate the grid size based on the input size n and block size.
10.	Start CPU Timer
11.	Initialize a CPU timer to measure the CPU execution time.
12.	Compute CPU Sum
13.	Calculate the sum of the input data on the CPU using a for loop and store the result in sum_cpu.
14.	Stop CPU Timer
15.	Record the elapsed CPU time.
16.	Start GPU Timer
17.	Initialize a GPU timer to measure the GPU execution time.
Kernel Execution
18.	Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.
Data Transfer: Device to Host
19.	Copy the result data from the device (d_odata) to the host (h_odata) using cudaMemcpy.
20.	Compute GPU Sum
21.	Calculate the final sum on the GPU by summing the elements in h_odata and store the result in sum_gpu.
22.	Stop GPU Timer
23.	Record the elapsed GPU time.
24.	Print Results
25.	Display the computed CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.
Memory Deallocation
26.	Free the allocated host and device memory using free and cudaFree.
27.	Exit
28.	Return from the main function.

## PROGRAM:
%%writefile parallel_reduction_unrolling.cu
// parallel_reduction_unrolling.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "Error: %s:%d, code: %d, reason: %s\n",              \
                __FILE__, __LINE__, error, cudaGetErrorString(error));       \
        exit(1);                                                             \
    }                                                                        \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

// ---------------- Host data initialization ----------------
void initialData(int *ip, int size)
{
    for (int i = 0; i < size; i++)
        ip[i] = 1;                 // all ones → sum should = size
}

// ---------------- CPU reduction ----------------
long long cpuReduce(int *data, int size)
{
    long long sum = 0;
    for (int i = 0; i < size; i++)
        sum += data[i];
    return sum;
}

// ---------------- Kernel: reduceUnrolling8 ----------------
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int sum = 0;

    if (idx < n)
    {
        sum += g_idata[idx];
        if (idx + blockDim.x < n)     sum += g_idata[idx + blockDim.x];
        if (idx + 2 * blockDim.x < n) sum += g_idata[idx + 2 * blockDim.x];
        if (idx + 3 * blockDim.x < n) sum += g_idata[idx + 3 * blockDim.x];
        if (idx + 4 * blockDim.x < n) sum += g_idata[idx + 4 * blockDim.x];
        if (idx + 5 * blockDim.x < n) sum += g_idata[idx + 5 * blockDim.x];
        if (idx + 6 * blockDim.x < n) sum += g_idata[idx + 6 * blockDim.x];
        if (idx + 7 * blockDim.x < n) sum += g_idata[idx + 7 * blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ---------------- Kernel: reduceUnrolling16 ----------------
__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int sum = 0;

    if (idx < n)
    {
        for (int i = 0; i < 16; i++)
            if (idx + i * blockDim.x < n)
                sum += g_idata[idx + i * blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ---------------- Main ----------------
int main(int argc, char **argv)
{
    printf("Parallel Reduction Using Unrolling Techniques\n");

    int n = 1 << 24;  
    size_t nBytes = n * sizeof(int);

    int *h_idata = (int*)malloc(nBytes);
    int *h_odata8, *h_odata16;

    initialData(h_idata, n);

    double cpuStart = seconds();
    long long sum_cpu = cpuReduce(h_idata, n);
    double cpuTime = seconds() - cpuStart;

    printf("CPU Sum = %lld, Time = %f sec\n", sum_cpu, cpuTime);

    int *d_idata, *d_odata8, *d_odata16;
    CHECK(cudaMalloc((void**)&d_idata, nBytes));
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));

    int blockSize = 256;

    // ========== reduceUnrolling8 ==========
    int unroll8 = 8;
    int gridSize8 = (n + blockSize * unroll8 - 1) / (blockSize * unroll8);

    CHECK(cudaMalloc((void**)&d_odata8, gridSize8 * sizeof(int)));
    h_odata8 = (int*)malloc(gridSize8 * sizeof(int));

    cudaEvent_t start8, stop8;
    cudaEventCreate(&start8);
    cudaEventCreate(&stop8);

    cudaEventRecord(start8);
    reduceUnrolling8<<<gridSize8, blockSize, blockSize * sizeof(int)>>>(d_idata, d_odata8, n);
    cudaEventRecord(stop8);
    cudaEventSynchronize(stop8);

    float time8;
    cudaEventElapsedTime(&time8, start8, stop8);
    cudaMemcpy(h_odata8, d_odata8, gridSize8 * sizeof(int), cudaMemcpyDeviceToHost);

    long long sum_gpu8 = 0;
    for (int i = 0; i < gridSize8; i++)
        sum_gpu8 += h_odata8[i];

    printf("\nUnrolling 8 → GPU Sum = %lld, Time = %f ms\n", sum_gpu8, time8);

    // ========== reduceUnrolling16 ==========
    int unroll16 = 16;
    int gridSize16 = (n + blockSize * unroll16 - 1) / (blockSize * unroll16);

    CHECK(cudaMalloc((void**)&d_odata16, gridSize16 * sizeof(int)));
    h_odata16 = (int*)malloc(gridSize16 * sizeof(int));

    cudaEvent_t start16, stop16;
    cudaEventCreate(&start16);
    cudaEventCreate(&stop16);

    cudaEventRecord(start16);
    reduceUnrolling16<<<gridSize16, blockSize, blockSize * sizeof(int)>>>(d_idata, d_odata16, n);
    cudaEventRecord(stop16);
    cudaEventSynchronize(stop16);

    float time16;
    cudaEventElapsedTime(&time16, start16, stop16);
    cudaMemcpy(h_odata16, d_odata16, gridSize16 * sizeof(int), cudaMemcpyDeviceToHost);

    long long sum_gpu16 = 0;
    for (int i = 0; i < gridSize16; i++)
        sum_gpu16 += h_odata16[i];

    printf("\nUnrolling 16 → GPU Sum = %lld, Time = %f ms\n", sum_gpu16, time16);

    printf("\nSummary:\n");
    printf(" Unrolling 8  : %f ms\n", time8);
    printf(" Unrolling 16 : %f ms\n", time16);

    free(h_idata);
    free(h_odata8);
    free(h_odata16);

    cudaFree(d_idata);
    cudaFree(d_odata8);
    cudaFree(d_odata16);

    cudaDeviceReset();
    return 0;
}

!nvcc -arch=sm_75 parallel_reduction_unrolling.cu -o reduction
!./reduction

## OUTPUT:
<img width="1044" height="149" alt="{8571C230-BC02-47EB-B62C-BB766A0DE637}" src="https://github.com/user-attachments/assets/75e5ef3b-6826-4438-9005-3e79ddb97fbe" />


## RESULT:
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that _________ has executed with less elapsed time than _____________ with blocks_____,______.
