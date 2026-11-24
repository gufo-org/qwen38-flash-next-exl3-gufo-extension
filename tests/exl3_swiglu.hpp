// Reference SwiGLU for tests/exl3_ops_test.cpp (tests/exl3_swiglu.hip.cpp).
#ifndef QWEN38_FLASH_NEXT_EXL3_TESTS_EXL3_SWIGLU_HPP_
#define QWEN38_FLASH_NEXT_EXL3_TESTS_EXL3_SWIGLU_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

/// gate[i] = silu(gate[i]) * up[i] for `count` elements, queued on `stream`.
void Exl3TestSwiglu(float* gate, const float* up, std::size_t count,
                    hipStream_t stream);

#endif  // QWEN38_FLASH_NEXT_EXL3_TESTS_EXL3_SWIGLU_HPP_
