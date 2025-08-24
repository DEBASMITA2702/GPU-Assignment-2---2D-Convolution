#include <chrono>      
#include <fstream>       
#include <iostream>      
#include <stdio.h>       
#include <cuda.h>        
#include <cuda_runtime.h>       // Modified <cuda/cuda_runtime.h> to <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll; 

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    extern __shared__ long int shared_filter[];     // Declaring shared memory to store the filter data

    long int num_filter = blockIdx.x;     
    long int thread_id_in_image = blockIdx.y * blockDim.x + threadIdx.x;    // Fetching global thread-index within the image

    long int total_elements = r * s * c;      // Total filter elements to be copied to shared memory to this blocked i.e., whole filter has to be copied.
    long int current_thread = threadIdx.x;    // Fetching the thread-index within the block

    for (int element_to_add_to_shared = current_thread; element_to_add_to_shared < total_elements; element_to_add_to_shared += blockDim.x)  //Looping over the total elements to distribute the work of copying of filter elements to shared memory among all the threads in the block
    {
        shared_filter[element_to_add_to_shared] = filter[num_filter * total_elements + element_to_add_to_shared];     // Loading the filter into shared memory
    }

    __syncthreads(); // Synchronize to ensure all threads have loaded the filter

    if (thread_id_in_image < h * w) {  
        long int convolution_sum = 0;   // Declaring a variable to store the convolution result

        // Calculating the current thread's position in the image
        long int current_thread_row_in_image = thread_id_in_image / w; 
        long int current_thread_col_in_image = thread_id_in_image % w;

        // Computing the starting and ending positions for conovlution
        long int start_row_in_image = current_thread_row_in_image - (r / 2);
        long int start_col_in_image = current_thread_col_in_image - (s / 2);
        long int end_row_in_image = start_row_in_image + r - 1;
        long int end_col_in_image = start_col_in_image + s - 1;

        for (long int curr_channel = 0; curr_channel < c; curr_channel++)   // Perform convolution operation across all channels
        {
            long int filter_row = 0;
            for (long int row = start_row_in_image; row <= end_row_in_image; row++) {
                long int filter_col = 0;
                for (long int col = start_col_in_image; col <= end_col_in_image; col++) {
                    if (row >=0 && row < h && col >= 0 && col < w) {         // Applying zero-padding for the out-of-bounds elements.
                        long int filter_element = shared_filter[curr_channel * r * s + filter_row * s + filter_col];
                        long int image_element = matrix[curr_channel * h * w + row * w + col];
                        convolution_sum += (filter_element * image_element);
                    }
                    filter_col++; 
                }
                filter_row++; 
            }
        }
        
        result[(num_filter * h * w) + (current_thread_row_in_image * w) + current_thread_col_in_image] = convolution_sum;       // Storing the final convolution result into the output matrix.
    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++) {
        cin >> h_mat[i]; 
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++) {
        cin >> h_filter[i]; 
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    long int *d_image, *d_filter, *d_result;    // Allocating memory on the device for image, filters, and result matrix
     
    cudaMalloc(&d_image, sizeof(long int) * h * w * c);
    cudaMalloc(&d_filter, sizeof(long int) * r * s * c * k);
    cudaMalloc(&d_result, sizeof(long int) * h * k * w);
    
    
    cudaMemcpy(d_image, h_mat, sizeof(long int) * h * w * c, cudaMemcpyHostToDevice);   // Copying input image data from host (CPU) to device (GPU)
    cudaMemcpy(d_filter, h_filter, sizeof(long int) * r * s * c * k, cudaMemcpyHostToDevice);   // Copying input filter data from host (CPU) to device (GPU)
    
    // Defining CUDA kernel execution configuration
    unsigned block_size = 1024;     // Defining number of threads per block
    unsigned num_blocks = ceil((h * w) / 1024.0);   // Calculating the number of blocks required
    dim3 grid(k, num_blocks, 1);     // Defining the grid dimensions (filters, image pixels)

    dkernel <<<grid, block_size, sizeof(long int) * r * s * c>>> (d_image, d_filter, d_result, h, w, c, r, s, k);      // Launching kernel with shared memory

    cudaMemcpy(h_ans, d_result, sizeof(long int) * h * k * w, cudaMemcpyDeviceToHost);       // Copying back the result from device(GPU) to host (CPU)
    
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);

    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start; 
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
