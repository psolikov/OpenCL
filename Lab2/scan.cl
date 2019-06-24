#define SWAP(a,b) {__local double * tmp=a; a=b; b=tmp;}

__kernel void flush(__global double * input, __global double * block_sum, int N)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint block_i = get_group_id(0);
    uint n_blocks = N / block_size + 1;
    uint actual_N = n_blocks * block_size;

    if (block_i != 0 && gid < N) 
    {
        input[gid] = input[gid] + block_sum[block_i - 1];
    }
}

__kernel void scan_hillis_steele(__global double * input, __global double * output, __global double * block_sum,
__local double * a, __local double * b, int N)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint n_blocks = N / block_size + 1;
    uint actual_N = n_blocks * block_size;
    uint block_i = get_group_id(0);
 
    if (gid < N) a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if (gid < N)
        {
            if(lid > (s-1))
            {
                b[lid] = a[lid] + a[lid-s];
            }
            else
            {
                b[lid] = a[lid];
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    if (gid < N) output[gid] = a[lid];
    if (lid == block_size - 1) block_sum[block_i] = a[lid];
}