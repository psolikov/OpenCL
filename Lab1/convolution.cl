__kernel void convolution(__global double *A, __global double *B, __global double *C, int N, int M) {
    int id = get_global_id(0);
    int HM = (M - 1) / 2;
    int i = id / N;
    int j = id % N;

    for (int k = -HM; k <= HM; ++k) 
    {
        for (int l = -HM; l <= HM; ++l) 
        {
            int a = i + k;
            int b = j + l;
            if (a >= 0 && b >= 0 && a < N && b < N) 
            {
                C[id] += A[a * N + b] * B[(k + HM) * M + (l + HM)];
            }
        }
    }
}