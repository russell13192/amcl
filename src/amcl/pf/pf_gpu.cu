#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#include "amcl/pf/pf.h"
#include "amcl/pf/pf_gpu.cuh"
#include "amcl/map/map.h"

#define CUDA_SAFE_CALL(call)                            \
{                                                       \
    const cudaError_t error = call;                     \
    if (error != cudaSuccess)                           \
    {                                                   \
        printf("Error: %s:%d,  ", __FILE__, __LINE__);  \
        printf("code:%d, reason: %s\n", error,          \
            cudaGetErrorString(error));                 \
        exit(1);                                        \
    }                                                   \
}

__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =(unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__
void dev_pf_gpu_alloc(pf_sample_t samples[], int max_count)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  i = max(i, 0);
  i = min(i, max_count);

  if (i < max_count)
  {
    samples[i].pose.v[0] = 0.0;
    samples[i].pose.v[1] = 0.0;
    samples[i].pose.v[2] = 0.0;
    samples[i].weight = 1.0 / max_count;
  }
}

void pf_gpu_alloc(pf_sample_set_t *set)
{
  pf_sample_t *dev_samples;

  dim3 block(256);
  dim3 grid((set->sample_count + block.x - 1) / block.x);
  static const int max_count = set->sample_count; 

  CUDA_SAFE_CALL(cudaMalloc(&dev_samples, sizeof(pf_sample_t) * set->sample_count));
  CUDA_SAFE_CALL(cudaMemcpy(dev_samples, set->samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyHostToDevice));

  dev_pf_gpu_alloc<<<grid, block>>>(dev_samples, max_count);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(set->samples, dev_samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyDeviceToHost));
  cudaFree(dev_samples);
}

__global__
void dev_pf_gpu_update_resample(pf_sample_t samples_a[], pf_sample_t samples_b[], 
  int min_count, int max_count, int loop_count, 
  double dev_w_diff, double *dev_c, double *dev_total, 
  map_t *map, map_cell_t map_cells[])
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  i = max(i, min_count);
  i = min(i, max_count);

  curandState s;
  curand_init(0, i, 0, &s);
  double r = curand_uniform_double(&s);

  if (i < max_count)
  {
    // Naive discrete event sampler
    if (r < dev_w_diff)
    {
      printf("*** WARNING : NEW UNIFROM RESAMPLING is not supported ****");
      double min_x, max_x, min_y, max_y;

      min_x = (map->size_x * map->scale)/2.0 - map->origin_x;
      max_x = (map->size_x * map->scale)/2.0 + map->origin_x;
      min_y = (map->size_y * map->scale)/2.0 - map->origin_y;
      max_y = (map->size_y * map->scale)/2.0 + map->origin_y;

      pf_vector_t p;
      for (;;)
      {
        double rr0 = curand_uniform_double(&s); 
        double rr1 = curand_uniform_double(&s); 
        double rr2 = curand_uniform_double(&s); 
        p.v[0] = min_x + rr0 * (max_x - min_x);
        p.v[1] = min_y + rr1 * (max_y - min_y);
        p.v[2] = rr2 * 2 * M_PI - M_PI;

        // Check that it's a free cell
        int pose_x, pose_y;
        pose_x = MAP_GXWX(map, p.v[0]);
        pose_y = MAP_GYWY(map, p.v[1]);
        if (MAP_VALID(map, pose_x, pose_y) && (map_cells[MAP_INDEX(map, pose_x, pose_y)].occ_state == -1))
        {
          break;
        }
      }

      samples_b[i].pose.v[0] = p.v[0];
      samples_b[i].pose.v[1] = p.v[1];
      samples_b[i].pose.v[2] = p.v[2];
    }
    else
    {
      int j;
      for (j=0;j<loop_count;j++)
      {
        if ((dev_c[j] <= r) && (r < dev_c[j+1]))
        {
          break;
        }
      }
      // Add sample to list
      samples_b[i].pose = samples_a[j].pose;
    }
    samples_b[i].weight = 1.0;

    __syncthreads(); 
    atomicAdd_double(dev_total, samples_b[i].weight); 
  }
}

void pf_gpu_update_resample(pf_sample_set_t *set_a, pf_sample_set_t *set_b, pf_t *pf, 
  double w_diff, double *c, double *total, void *random_pose_data)
{
  pf_sample_t *dev_samples_a;
  pf_sample_t *dev_samples_b;
  map_t *map = (map_t *)random_pose_data;

  static const int min_count = set_b->sample_count;
  static const int max_count = pf->max_samples - 1;
  static const int loop_count = set_a->sample_count;
  static const double dev_w_diff = w_diff;
  double *dev_c;
  double *dev_total;
  map_t *dev_map;
  map_cell_t *dev_map_cells;

  CUDA_SAFE_CALL(cudaMalloc(&dev_samples_a, sizeof(pf_sample_t) * set_a->sample_count));
  CUDA_SAFE_CALL(cudaMalloc(&dev_samples_b, sizeof(pf_sample_t) * pf->max_samples));
  CUDA_SAFE_CALL(cudaMalloc(&dev_c, sizeof(double) * (set_a->sample_count + 1)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_total, sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map, sizeof(map_t)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map_cells, sizeof(map_cell_t) * map->size_x * map->size_y));

  CUDA_SAFE_CALL(cudaMemcpy(dev_samples_a, set_a->samples, sizeof(pf_sample_t) * set_a->sample_count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_samples_b, set_b->samples, sizeof(pf_sample_t) * pf->max_samples, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_c, c, sizeof(double) * (set_a->sample_count + 1), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_total, total, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map, map, sizeof(map_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map_cells, map->cells, sizeof(map_cell_t) * map->size_x * map->size_y, cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((pf->max_samples + block.x - 1) / block.x);
  dev_pf_gpu_update_resample<<<grid, block>>>(dev_samples_a, dev_samples_b, min_count, max_count, loop_count, dev_w_diff, dev_c, dev_total, dev_map, dev_map_cells);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  CUDA_SAFE_CALL(cudaMemcpy(set_b->samples, dev_samples_b, sizeof(pf_sample_t) * pf->max_samples, cudaMemcpyDeviceToHost));
  set_b->sample_count = pf->max_samples;
  CUDA_SAFE_CALL(cudaMemcpy(total, dev_total, sizeof(double), cudaMemcpyDeviceToHost));

  for (int i = 0; i < set_b->sample_count; i++)
  {
     pf_sample_t *sample_b = set_b->samples + i;
     pf_kdtree_insert(set_b->kdtree, sample_b->pose, sample_b->weight);
  }

  CUDA_SAFE_CALL(cudaFree(dev_samples_a));
  CUDA_SAFE_CALL(cudaFree(dev_samples_b));
  CUDA_SAFE_CALL(cudaFree(dev_c));
  CUDA_SAFE_CALL(cudaFree(dev_total));
  CUDA_SAFE_CALL(cudaFree(dev_map));
  CUDA_SAFE_CALL(cudaFree(dev_map_cells));
}

