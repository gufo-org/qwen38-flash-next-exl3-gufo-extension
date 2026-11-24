// The unfused SwiGLU the ops test compares Exl3LinearGroup's staged product
// with: gate = silu(gate) * up in place, as Flash-Next's SwigluKernel writes
// it (x * (1 / (1 + exp(-x))) * up, the same expression and HIP options), so
// the comparison stays exact without Flash-Next's private kernel header.
#include "tests/exl3_swiglu.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    const float x = gate[i];
    gate[i] = x * (1.0F / (1.0F + __expf(-x))) * up[i];
  }
}

}  // namespace

void Exl3TestSwiglu(float* gate, const float* up, std::size_t count,
                    hipStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>((count + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(SwigluKernel, dim3(blocks), dim3(kThreads), 0, stream,
                     gate, up, count);
}
