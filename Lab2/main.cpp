#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl_platform.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

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
        std::ifstream cl_file("scan.cl");
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
            std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cerr << log_str;
            return 0;
        }

        // create a message to send to kernel
        std::ifstream ifs("input.txt");
        std::ofstream ofs("output.txt");
        int N;
        size_t const block_size = 256;

        ifs >> N;
        int n_blocks = N / block_size + 1;
        int actual_N = n_blocks * block_size;

        std::vector<double> input(actual_N);
        std::vector<double> output(actual_N);
        std::vector<double> block_sum(n_blocks);

        for (int i = 0; i < N; ++i) 
        {
            ifs >> input[i];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * actual_N);
        cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(double) * actual_N);
        cl::Buffer dev_block_sum(context, CL_MEM_READ_WRITE, sizeof(double) * n_blocks);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * actual_N, &input[0]);
        queue.enqueueWriteBuffer(dev_output, CL_TRUE, 0, sizeof(double) * actual_N, &output[0]);
        queue.enqueueWriteBuffer(dev_block_sum, CL_TRUE, 0, sizeof(double) * n_blocks, &block_sum[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_scan(program, "scan_hillis_steele");
        kernel_scan.setArg(0, dev_input);
        kernel_scan.setArg(1, dev_output);
        kernel_scan.setArg(2, dev_block_sum);
        kernel_scan.setArg(3, cl::__local(sizeof(double) * block_size));
        kernel_scan.setArg(4, cl::__local(sizeof(double) * block_size));
        kernel_scan.setArg(5, N);
        queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(actual_N), cl::NDRange(block_size));
        queue.enqueueReadBuffer(dev_block_sum, CL_TRUE, 0, sizeof(double) * n_blocks, &block_sum[0]);

        for (int i = 1; i < n_blocks; ++i) 
        {
            block_sum[i] = block_sum[i] + block_sum[i - 1];
        }

        queue.enqueueWriteBuffer(dev_block_sum, CL_TRUE, 0, sizeof(double) * n_blocks, &block_sum[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_flush(program, "flush");
        kernel_flush.setArg(0, dev_output);
        kernel_flush.setArg(1, dev_block_sum);
        kernel_flush.setArg(2, N);
        queue.enqueueNDRangeKernel(kernel_flush, cl::NullRange, cl::NDRange(actual_N), cl::NDRange(block_size));

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * actual_N, &output[0]);

        for (int i = 0; i < N; ++i) {
            ofs << output[i] << " ";
            std::cout << output[i] << " ";
        }
        ofs << "\n";
        std::cout << "\n";
    }
    catch (cl::Error const & e)
    {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}