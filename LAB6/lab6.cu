#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define smalloc(type, ptr, num)                                                \
  if (!(ptr = (type *)malloc(sizeof(type) * (num))))                           \
  exit(1)
#define Verifylen (1024)
#define Seqlen (262144)
#define AD (32)
#define AT (64)
#define BC (AT) // Bc = Br = AT

__global__ void Single_head_flash_atten(float *Q, float *K, float *V, float *O,
                                        unsigned l,
                                        float softmax_scale) { // unsigned d,
  // 线程索引
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= l)
    return;

  float sumexp = 0.0f;    // 当前行指数和
  float O_temp[AD] = {0}; // 中间结果，需要最后归一化

  __shared__ float Qs[BC][AD];
#pragma unroll
  for (int j = 0; j < AD; j++) {
    Qs[threadIdx.x][j] = Q[row * AD + j];
  }

  // 外层循环：遍历K/V的块
  for (unsigned j_block = 0; j_block < l; j_block += BC) {

    __shared__ float Ks[BC][AD];
    __shared__ float Vs[BC][AD];
    // __shared__ float Qs[BC][AD];

    // 协作加载K和V的分块到共享内存
    if (threadIdx.x < BC) {
      unsigned load_idx = j_block + threadIdx.x;
#pragma unroll
      for (int j = 0; j < AD; j++) {
        Ks[threadIdx.x][j] = K[load_idx * AD + j];
        Vs[threadIdx.x][j] = V[load_idx * AD + j];
        // Qs[threadIdx.x][j] = Q[row * AD + j];
      }
    }
    __syncthreads();

    // 计算当前块的S值
    float S_values[BC] = {0};
#pragma unroll
    for (int j = 0; j < BC; j++) {
#pragma unroll
      for (int k = 0; k < AD; k++) {
        S_values[j] += Qs[threadIdx.x][k] * Ks[j][k]; // Q * K^T
      }
      S_values[j] = expf(S_values[j] * softmax_scale);
      sumexp += S_values[j]; // 累加指数和
    }

// 计算O
#pragma unroll
    for (int j = 0; j < BC; j++) {
#pragma unroll
      for (int k = 0; k < AD; k++) {
        O_temp[k] += S_values[j] * Vs[j][k];
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < AD; j++) {
    O[row * AD + j] = O_temp[j] / sumexp;
  }

  return;
}

__host__ void single_head_atten_base(float *Q, float *K, float *V, float *O,
                                     unsigned l, float softmax_scale) {
  unsigned i, j, k;
  float *S, *Ssum;
  smalloc(float, S, l *l);
  smalloc(float, Ssum, l);
  for (i = 0; i < l; i++) {
    Ssum[i] = 0;
    for (j = 0; j < l; j++) {
      S[i * l + j] = 0;
      for (k = 0; k < AD; k++) {
        S[i * l + j] += Q[i * AD + k] * K[k + j * AD]; // Q* KT
      }
      S[i * l + j] = exp(S[i * l + j] * softmax_scale); //
      Ssum[i] += S[i * l + j];
    }
  }

  for (i = 0; i < l; i++) {
    for (j = 0; j < AD; j++) {
      O[i * AD + j] = 0;
      for (k = 0; k < l; k++) {
        O[i * AD + j] += S[i * l + k] * V[k * AD + j] / Ssum[i];
      }
    }
  }
  free(S);
  free(Ssum);
}

__host__ void gen_QKV(float **phQ, float **phK, float **phV, unsigned l,
                      unsigned d) {
  float *hQ, *hK, *hV;
  smalloc(float, hQ, l *d);
  smalloc(float, hK, l *d);
  smalloc(float, hV, l *d);
  unsigned i;
  for (i = 0; i < l * d; i++) {
    hQ[i] = 1.0 * rand() / RAND_MAX;
    hK[i] = 1.0 * rand() / RAND_MAX;
    hV[i] = 1.0 * rand() / RAND_MAX;
  }
  *phQ = hQ;
  *phK = hK;
  *phV = hV;
}
__host__ unsigned compare(float *pred_, float *true_, unsigned n) {
  unsigned i;
  float relative_error;
  for (i = 0; i < n; i++) {
    relative_error = fabs((pred_[i] - true_[i]) / true_[i]);
    if (relative_error >= 1e-5) {
      printf("not equal! relative error: %12.9lf pred: %12.9f true: %12.9f\n",
             relative_error, pred_[i], true_[i]);
      return 1;
    } else {
      // printf("equal! relative error: %12.9lf pred: %12.9f true: %12.9f\n",
      //        relative_error, pred_[i], true_[i]);
    }
  }
  printf("equal!\n");
  return 0;
}

int prinMat(float *A, int m, int n, FILE *fp) {
  int i, j;
  for (i = 0; i < m; i++) {
    fprintf(fp, "%4d:", i);
    for (j = 0; j < n; j++) {
      fprintf(fp, "%12.9f ", A[i * n + j]);
    }
    fprintf(fp, "\n");
  }
  return 0;
}

int main(void) {
  float *dQ, *dK, *dV, *dO, *hQ, *hK, *hV, *hO, *Obase;
  const unsigned Vl = Verifylen, Pl = Seqlen;
  const float softmax_scale = 1 / sqrt(AD);
  unsigned i;
  gen_QKV(&hQ, &hK, &hV, Vl, AD);
  smalloc(float, hO, Vl *AD);
  smalloc(float, Obase, Vl *AD);
  cudaMalloc(&dQ, sizeof(float) * (Vl * AD));
  cudaMalloc(&dK, sizeof(float) * (Vl * AD));
  cudaMalloc(&dV, sizeof(float) * (Vl * AD));
  cudaMalloc(&dO, sizeof(float) * (Vl * AD));
  cudaMemcpy(dQ, hQ, sizeof(float) * (Vl * AD), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, sizeof(float) * (Vl * AD), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, sizeof(float) * (Vl * AD), cudaMemcpyHostToDevice);
  dim3 gridsize(Vl / AT), blocksize(AT);
  Single_head_flash_atten<<<gridsize, blocksize>>>(dQ, dK, dV, dO, Vl,
                                                   softmax_scale);
  cudaMemcpy(hO, dO, sizeof(float) * (Vl * AD), cudaMemcpyDeviceToHost);
  single_head_atten_base(hQ, hK, hV, Obase, Vl, softmax_scale);
  cudaDeviceSynchronize();
  unsigned flag = 0;
  flag |= compare(hO, Obase, Vl * AD);
  if (flag) {
    printf("test fail!\n");
    exit(0);
  }
  printf("test pass!\n");
  free(hQ);
  free(hK);
  free(hV);
  free(hO);
  free(Obase);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);

  gen_QKV(&hQ, &hK, &hV, Pl, AD);
  cudaMalloc(&dQ, sizeof(float) * (Pl * AD));
  cudaMalloc(&dK, sizeof(float) * (Pl * AD));
  cudaMalloc(&dV, sizeof(float) * (Pl * AD));
  cudaMalloc(&dO, sizeof(float) * (Pl * AD));
  cudaMemcpy(dQ, hQ, sizeof(float) * (Pl * AD), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, sizeof(float) * (Pl * AD), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, sizeof(float) * (Pl * AD), cudaMemcpyHostToDevice);
  gridsize = {Pl / AT};
  blocksize = {AT};

  cudaEvent_t start, stop;
  float Time1 = 0.0, temp = 0;
  const unsigned loopnum = 10;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (i = 0; i < loopnum; i++) {
    cudaEventRecord(start, 0);
    Single_head_flash_atten<<<gridsize, blocksize>>>(dQ, dK, dV, dO, Pl,
                                                     softmax_scale);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp, start, stop);
    Time1 += temp;
    temp = 0;
    cudaDeviceSynchronize();
  }

  printf("l: %5.d   time: %12.9f\n", Pl, Time1 / loopnum);
  free(hQ);
  free(hK);
  free(hV);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
}