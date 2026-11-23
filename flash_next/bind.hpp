// Binds an EXL3 checkpoint of Qwen3.8-Flash-Next to Flash-Next's weight
// descriptors (ModelWeights, MtpWeights). Trellis linears stay in the mapped
// checkpoint and carry an Exl3Ref as TensorRef::format_state; the small
// tensors whose layout or encoding differs from the official GGUF are
// converted into HostTensors (TensorRef::shard == kHostShard).
#ifndef QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_BIND_HPP_
#define QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_BIND_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "exl3/pack.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::ext::qwen38_flash_next_exl3 {

using ::gufo::models::qwen38_flash_next::Config;
using ::gufo::models::qwen38_flash_next::ModelWeights;
using ::gufo::models::qwen38_flash_next::MtpWeights;
using ::gufo::models::qwen38_flash_next::TensorRef;

/// One EXL3 trellis linear, or a stack of expert linears of one shape, in a
/// checkpoint: the format_state of its TensorRef. The device stores each
/// matrix's tiles column-block-major; the optional 128-block maps reorder
/// (or select) input and output blocks while uploading, which commutes with
/// the block-diagonal Hadamards.
struct Exl3Ref {
  struct Part {
    std::uint32_t file{0};         ///< Checkpoint file of all three.
    std::uint64_t file_offset{0};  ///< Trellis payload offset in that file.
    std::uint64_t suh_offset{0};   ///< [src_in] fp16
    std::uint64_t svh_offset{0};   ///< [src_out] fp16
    /// Mapped scales; reading them faults the pages in, so uploads read
    /// the file ranges instead.
    const std::uint16_t* suh{nullptr};
    const std::uint16_t* svh{nullptr};
  };
  int k{0};                  ///< Bits per weight of every part.
  std::uint32_t src_in{0};   ///< Stored input width.
  std::uint32_t src_out{0};  ///< Stored output width.
  /// Device input block i reads stored block in_blocks[i]; empty keeps all.
  std::vector<std::uint32_t> in_blocks;
  /// Device output block j reads stored block out_blocks[j]; empty keeps all.
  std::vector<std::uint32_t> out_blocks;
  std::vector<Part> parts;  ///< One per expert; one for a dense matrix.

  [[nodiscard]] std::uint32_t In() const noexcept {
    return in_blocks.empty()
               ? src_in
               : static_cast<std::uint32_t>(in_blocks.size()) * 128;
  }
  [[nodiscard]] std::uint32_t Out() const noexcept {
    return out_blocks.empty()
               ? src_out
               : static_cast<std::uint32_t>(out_blocks.size()) * 128;
  }
  /// Stored trellis bytes of one part.
  [[nodiscard]] std::size_t SourceBytes() const noexcept {
    return static_cast<std::size_t>(src_in) * src_out * k / 8;
  }
  /// Device tile bytes of one part.
  [[nodiscard]] std::size_t DeviceBytes() const noexcept {
    return static_cast<std::size_t>(In()) * Out() * k / 8;
  }
};

/// The Exl3Ref of a bound trellis linear, or null.
[[nodiscard]] inline const Exl3Ref* RefOf(const TensorRef& t) noexcept {
  return static_cast<const Exl3Ref*>(t.format_state);
}

/// What a binding owns: converted payloads (folded norms, permuted rows,
/// Q8_0 projections), tensor names and the trellis descriptors. TensorRefs
/// point into it, so it must outlive the weights.
struct HostTensors {
  std::deque<std::vector<std::uint8_t>> buffers;
  std::deque<std::string> names;
  std::deque<Exl3Ref> refs;
};

/// The checkpoint's trellis n-gram table, read from disk only.
struct NgramRef {
  std::uint32_t file{0};         ///< Checkpoint file of the table.
  std::uint64_t file_offset{0};  ///< First row's offset in that file.
  std::uint64_t rows{0};
  int k{0};                              ///< Bits per symbol.
  std::vector<std::uint16_t> head_bias;  ///< [ple_heads][ple_head_dim] fp16
};

/// Reads the checkpoint's config.json and n-gram tables. `mtp` reads the
/// configuration of its `mtp.*` draft block instead of the trunk.
[[nodiscard]] std::optional<Config> ConfigFromPack(const Pack& pack, bool mtp,
                                                   std::string* error_msg);

/// Binds the trunk; `ngram` receives the n-gram table when the
/// configuration has a PLE layer.
[[nodiscard]] std::optional<ModelWeights> BindTrunk(const Pack& pack,
                                                    HostTensors& host,
                                                    NgramRef* ngram,
                                                    std::string* error_msg);

/// The `mtp.*` block, or nullopt (with an empty error) when it has none.
[[nodiscard]] std::optional<MtpWeights> BindMtp(const Pack& pack,
                                                const Config& trunk,
                                                HostTensors& host,
                                                std::string* error_msg);

}  // namespace gufo::ext::qwen38_flash_next_exl3

#endif  // QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_BIND_HPP_
