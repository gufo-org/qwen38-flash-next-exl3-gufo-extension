// The format_state of an uploaded EXL3 linear: what the stages need beside
// the tiles (Tensor::data) to call the HIP linears.
#ifndef QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_DEVICE_FORMAT_HPP_
#define QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_DEVICE_FORMAT_HPP_

#include <hip/hip_fp16.h>

#include <cstdint>
#include <gufo/ext/families/qwen/models/flash_next/primitives.hpp>

#include "exl3/rocm/exl3.hpp"

namespace gufo::ext::qwen38_flash_next_exl3 {

struct DeviceFormat {
  const __half* suh{nullptr};  ///< [experts][cols]
  const __half* svh{nullptr};  ///< [experts][rows]
  int k{0};                    ///< Bits per weight.
  /// Decode split the tiles are in front order for (Exl3FrontTile).
  int front{0};
};

/// Only this extension's weight source attaches format_state, so a tensor
/// that carries one is an EXL3 linear.
[[nodiscard]] inline bool IsExl3(
    const ::gufo::ext::qwen::flash_next::Tensor& t) noexcept {
  return !t.empty() && t.format_state() != nullptr;
}

[[nodiscard]] inline rocm::Exl3Weights WeightsOf(
    const ::gufo::ext::qwen::flash_next::Tensor& t) noexcept {
  const auto& f = *static_cast<const DeviceFormat*>(t.format_state());
  return {static_cast<const std::uint32_t*>(t.data()),
          f.suh,
          f.svh,
          t.cols(),
          t.rows(),
          t.experts(),
          f.k,
          f.front};
}

}  // namespace gufo::ext::qwen38_flash_next_exl3

#endif  // QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_DEVICE_FORMAT_HPP_
