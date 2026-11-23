// The three EXL3 checkpoint providers: one per turboderp quantization, told
// apart by the checkpoint's bits per weight. Each serves its checkpoint
// directory through Flash-Next with the EXL3 stages and weight source, and
// drafts from the checkpoint's own MTP block.
#include <exception>
#include <filesystem>
#include <gufo/ext/generated/qwen38_flash_next_exl3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <string_view>
#include <utility>

#include "exl3/pack.hpp"
#include "flash_next/runner.hpp"

namespace {

namespace exl3 = gufo::ext::qwen38_flash_next_exl3;
namespace generated = gufo::ext::generated::extension_qwen38_flash_next_exl3;

/// A Flash-Next checkpoint directory whose quantization_config.bits lies in
/// [min_bits, max_bits).
bool ProbeBits(std::string_view model_root, double min_bits, double max_bits) {
  try {
    const std::filesystem::path root(model_root);
    if (!exl3::IsFlashNextPack(root))
      return false;
    const double bits = exl3::PackBits(root);
    return bits >= min_bits && bits < max_bits;
  } catch (const std::exception&) {
    return false;
  }
}

bool Probe4(std::string_view root) {
  return ProbeBits(root, 3.5, 4.5);
}
bool Probe3(std::string_view root) {
  return ProbeBits(root, 2.5, 3.5);
}
bool Probe2(std::string_view root) {
  return ProbeBits(root, 1.5, 2.5);
}

void Register(gufo::ext::Registry& registry, gufo::ext::ProviderInfo provider,
              bool (*probe)(std::string_view)) {
  gufo::ext::TextProvider text;
  text.id = provider.id;
  text.probe = probe;
  text.create = &exl3::CreateExl3TextRunner;
  gufo::ext::TextSpeculativeDefaults mtp;
  mtp.backend = "mtp";
  mtp.max_draft_tokens = 7;
  mtp.min_draft_tokens = 1;
  text.speculative_defaults.push_back(std::move(mtp));
  registry.AddProvider(std::move(provider));
  registry.AddTextProvider(std::move(text));
}

}  // namespace

void qwen38_flash_next_exl3_4bpw_provider_register(
    gufo::ext::Registry& registry) {
  Register(registry, generated::kProvider_flash_next_exl3_4bpw, &Probe4);
}

void qwen38_flash_next_exl3_3bpw_provider_register(
    gufo::ext::Registry& registry) {
  Register(registry, generated::kProvider_flash_next_exl3_3bpw, &Probe3);
}

void qwen38_flash_next_exl3_2bpw_provider_register(
    gufo::ext::Registry& registry) {
  Register(registry, generated::kProvider_flash_next_exl3_2bpw, &Probe2);
}
