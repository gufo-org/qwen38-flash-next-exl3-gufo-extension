// Flash-Next stages for EXL3 trellis linears (format_state DeviceFormat).
#ifndef QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_STAGES_HPP_
#define QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_STAGES_HPP_

#include <gufo/ext/families/qwen/models/flash_next/stages.hpp>
#include <memory>

namespace gufo::ext::qwen38_flash_next_exl3 {

/// Decode and verification run the trellis GEMVs, grouped per input; prompt
/// passes run trellis matrix-core GEMMs for the routed experts and the head
/// and take Flash-Next's Q8_0 routes for every other projection, expanded
/// once per layer into the equivalent Q8_0 matrices (PromptLayer).
[[nodiscard]] std::unique_ptr<::gufo::ext::qwen::flash_next::Stages>
CreateExl3Stages();

}  // namespace gufo::ext::qwen38_flash_next_exl3

#endif  // QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_STAGES_HPP_
