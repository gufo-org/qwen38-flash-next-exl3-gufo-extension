// The text runner of an EXL3 checkpoint: Flash-Next's model loaded with the
// EXL3 stages and weight source.
#ifndef QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_RUNNER_HPP_
#define QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_RUNNER_HPP_

#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

namespace gufo::ext::qwen38_flash_next_exl3 {

/// The checkpoint carries its own MTP block, tokenizer and configuration:
/// `--speculative mtp` needs no draft model.
[[nodiscard]] std::shared_ptr<TextModelRunner> CreateExl3TextRunner(
    const TextRunnerOptions& options, std::string* error);

}  // namespace gufo::ext::qwen38_flash_next_exl3

#endif  // QWEN38_FLASH_NEXT_EXL3_FLASH_NEXT_RUNNER_HPP_
