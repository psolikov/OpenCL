//
//  main.cpp
//  opencl-lab
//
//  Created by Pavel Solikov on 05/06/2019.
//

#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl_platform.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cmath>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {
      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      try
      {
         program.build(devices);
      }
      catch (cl::Error const & e)
      {         
         std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
         std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
         std::cout << log_str;
         return 0;
      }

      // create a message to send to kernel
      std::ifstream ifs("input.txt");
      size_t N, M;
      size_t const block_size = 256;
      size_t const test_array_size = 2;

      ifs >> N >> M;
      std::cout << N << " " << M << '\n';

      std::vector<double> A(N * N);
      std::vector<double> B(M * M);
      std::vector<double> C(N * N);

      for (size_t i = 0; i < N * N; ++i)
      {
         ifs >> A[i];
      }
      for (size_t i = 0; i < M * M; ++i)
      {
         ifs >> B[i];
      }

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * A.size());
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * B.size());
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * C.size());

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * A.size(), &A[0]);
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * B.size(), &B[0]);

      // load named kernel from opencl source
      cl::Kernel kernel_gmem(program, "convolution");
      kernel_gmem.setArg(0, dev_a);
      kernel_gmem.setArg(1, dev_b);
      kernel_gmem.setArg(2, dev_c);
      kernel_gmem.setArg(3, static_cast<int>(N));
      kernel_gmem.setArg(4, static_cast<int>(M));

      size_t gs = N * N;
      if (gs % block_size != 0) {
        gs = ((gs + block_size - 1) / block_size) * block_size;
        std::cout << gs << '\n';
      }       
      queue.enqueueNDRangeKernel(kernel_gmem, cl::NullRange, cl::NDRange(gs), cl::NDRange(block_size));
      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * C.size(), &C[0]);

      std::ofstream ofs("output.txt");

      for (size_t i = 0; i < N; ++i)
      {
        for (size_t j = 0; j < N; ++j){
         std::cout << C[j] << ' ';
         ofs << C[j] << ' ';
        }
        std::cout << '\n';
        ofs << '\n';
      }
   }
   catch (cl::Error const & e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
