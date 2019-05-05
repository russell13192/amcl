#include <cstdio>
#include <cstdlib>

#include "amcl/sensors/amcl_laser.h"
#include "amcl/sensors/amcl_gpu.cuh"
using namespace amcl;

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

__device__
double atomicAdd_double(double* address, double val)
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

__device__
pf_vector_t dev_pf_vector_coord_add(pf_vector_t a, pf_vector_t b)
{
  pf_vector_t c;

  c.v[0] = b.v[0] + a.v[0] * cos(b.v[2]) - a.v[1] * sin(b.v[2]);
  c.v[1] = b.v[1] + a.v[0] * sin(b.v[2]) + a.v[1] * cos(b.v[2]);
  c.v[2] = b.v[2] + a.v[2];
  c.v[2] = atan2(sin(c.v[2]), cos(c.v[2]));
  
  return c;
}

__global__
void dev_sensor_gpu_LikelihoodFieldModel(pf_sample_t samples[], int max_count,
  double z_hit, double z_rand, double z_hit_denom, double z_rand_mult, 
  int step, pf_vector_t laser_pose, map_t *map, map_cell_t map_cells[], int range_count, double range_max,
  double (*dev_ranges)[2], double *total_weight)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  i = max(i, 0);
  i = min(i, max_count);
  if(i < max_count)
  {
    // Take account of the laser pose relative to the robot
    pf_vector_t pose;
    pose = dev_pf_vector_coord_add(laser_pose, samples[i].pose);

    double p = 1.0;
    int j;

    for (j = 0; j < range_count; j += step)
    {
      double z, pz;
      pf_vector_t hit;
      int mi, mj;
      double obs_range = dev_ranges[j][0];
      double obs_bearing = dev_ranges[j][1];

      // This model ignores max range readings
      if(obs_range >= range_max)
        continue;

      // Check for NaN
      if(obs_range != obs_range)
        continue;

      pz = 0.0;

      // Compute the endpoint of the beam
      hit.v[0] = pose.v[0] + obs_range * cos(pose.v[2] + obs_bearing);
      hit.v[1] = pose.v[1] + obs_range * sin(pose.v[2] + obs_bearing);

      // Convert to map grid coords.
      mi = MAP_GXWX(map, hit.v[0]);
      mj = MAP_GYWY(map, hit.v[1]);
      
      // Part 1: Get distance from the hit to closest obstacle.
      // Off-map penalized as max distance
      if(!MAP_VALID(map, mi, mj)){
        z = map->max_occ_dist;
      } else {
        z = map_cells[MAP_INDEX(map,mi,mj)].occ_dist;
      }

      // Gaussian model
      // NOTE: this should have a normalization of 1/(sqrt(2pi)*sigma)
      pz += z_hit * exp(-(z * z) / z_hit_denom);
      // Part 2: random measurements
      pz += z_rand * z_rand_mult;

      // TODO: outlier rejection for short readings

      //      p *= pz;
      // here we have an ad-hoc weighting scheme for combining beam probs
      // works well, though...
      p += pz*pz*pz;
    }

    samples[i].weight *= p;

     __syncthreads(); 
    atomicAdd_double(total_weight, samples[i].weight);
  }
}

void sensor_gpu_LikelihoodFieldModel(pf_sample_set_t *set, void *arg_self, void *arg_data, double *total_weight)
{
  pf_sample_t *dev_samples;
  AMCLLaser *self = (AMCLLaser *)arg_self;
  AMCLLaserData *data = (AMCLLaserData *)arg_data;

  dim3 block(256);
  dim3 grid((set->sample_count + block.x - 1) / block.x);

  static const int max_count = set->sample_count;

  // Pre-compute a couple of things
  static const double z_hit = self->z_hit;
  static const double z_rand = self->z_rand;
  static const double z_hit_denom = 2 * self->sigma_hit * self->sigma_hit;
  static const double z_rand_mult = 1.0 / data->range_max;
  int step = (data->range_count - 1) / (self->max_beams - 1);

  // Step size must be at least 1
  if(step < 1)
    step = 1;

  static const int dev_step = step;
  static const pf_vector_t dev_laser_pose = self->laser_pose;
  map_t  *dev_map;
  map_cell_t *dev_map_cells;
  static const int dev_range_count = data->range_count;
  static const double dev_range_max = data->range_max;
  double (*dev_ranges)[2];
  double *dev_total_weight;
  
  CUDA_SAFE_CALL(cudaMalloc(&dev_samples, sizeof(pf_sample_t) * set->sample_count));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map, sizeof(map_t)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map_cells, sizeof(map_cell_t) * self->map->size_x * self->map->size_y));
  CUDA_SAFE_CALL(cudaMalloc(&dev_ranges, sizeof(double) * data->range_count * 2));
  CUDA_SAFE_CALL(cudaMalloc(&dev_total_weight, sizeof(double)));

  CUDA_SAFE_CALL(cudaMemcpy(dev_samples, set->samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map, self->map, sizeof(map_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map_cells, self->map->cells, sizeof(map_cell_t) * self->map->size_x * self->map->size_y, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_ranges, data->ranges, sizeof(double) * data->range_count * 2, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_total_weight, total_weight, sizeof(double), cudaMemcpyHostToDevice));

  dev_sensor_gpu_LikelihoodFieldModel<<<grid, block>>>(dev_samples, max_count,
    z_hit, z_rand, z_hit_denom, z_rand_mult, dev_step, dev_laser_pose, 
    dev_map, dev_map_cells, dev_range_count, dev_range_max,
    dev_ranges, dev_total_weight);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  
  CUDA_SAFE_CALL(cudaMemcpy(set->samples, dev_samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_weight, dev_total_weight, sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(dev_samples));
  CUDA_SAFE_CALL(cudaFree(dev_map));
  CUDA_SAFE_CALL(cudaFree(dev_map_cells));
  CUDA_SAFE_CALL(cudaFree(dev_ranges));
  CUDA_SAFE_CALL(cudaFree(dev_total_weight));

}

