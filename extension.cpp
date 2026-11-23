// Registration of the EXL3 extension. The three checkpoint providers keep
// their final IDs, so `gufo model` can install their weights now, but they
// report themselves unavailable: serving needs the Flash-Next Stages and
// WeightSource interface (see README.md).
#include <gufo/ext/generated/qwen38_flash_next_exl3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <utility>

namespace {

namespace generated = gufo::ext::generated::extension_qwen38_flash_next_exl3;

void RegisterUnavailable(gufo::ext::Registry& registry,
                         gufo::ext::ProviderInfo provider) {
  provider.available = false;
  provider.unavailable_reason =
      "EXL3 serving waits for the Flash-Next Stages and WeightSource "
      "interface; this build carries the format and kernels only";
  registry.AddProvider(std::move(provider));
}

}  // namespace

void qwen38_flash_next_exl3_4bpw_provider_register(
    gufo::ext::Registry& registry) {
  RegisterUnavailable(registry, generated::kProvider_flash_next_exl3_4bpw);
}

void qwen38_flash_next_exl3_3bpw_provider_register(
    gufo::ext::Registry& registry) {
  RegisterUnavailable(registry, generated::kProvider_flash_next_exl3_3bpw);
}

void qwen38_flash_next_exl3_2bpw_provider_register(
    gufo::ext::Registry& registry) {
  RegisterUnavailable(registry, generated::kProvider_flash_next_exl3_2bpw);
}

void qwen38_flash_next_exl3_register(gufo::ext::Registry& registry) {
  (void)registry;
}
