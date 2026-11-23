// Flash-Next's weights from an EXL3 checkpoint directory: the trunk, its
// own `mtp.*` draft block, tokenizer.json and the trellis n-gram table.
#ifndef QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_WEIGHT_SOURCE_HPP_
#define QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_WEIGHT_SOURCE_HPP_

#include <cstdint>
#include <memory>

#include "src/models/qwen38_flash_next/weight_source.hpp"

namespace gufo::ext::qwen38_flash_next_exl3 {

/// EXL3 checkpoints stream each routed expert's trellis once per pass and
/// row tile, so their prompt passes double the GGUF's: an 8,022-token prompt
/// prefills faster in one pass. Their device weights are smaller than the
/// Q4 GGUF's, which leaves room for the larger scratch.
inline constexpr std::uint32_t kPrefillChunkTokens = 8192;

[[nodiscard]] std::unique_ptr<::gufo::ext::qwen::flash_next::WeightSource>
CreateExl3WeightSource();

}  // namespace gufo::ext::qwen38_flash_next_exl3

#endif  // QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_WEIGHT_SOURCE_HPP_
