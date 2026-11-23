// Registration of the EXL3 extension. Gufo requires a provider, so this one
// claims the EXL3 checkpoint keys and reports itself unavailable: serving
// needs the Flash-Next Stages and WeightSource interface (see README.md).
#include <gufo/ext/generated/qwen38_flash_next_exl3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <utility>

void qwen38_flash_next_exl3_provider_register(gufo::ext::Registry& registry) {
  gufo::ext::ProviderInfo provider = gufo::ext::generated::
      extension_qwen38_flash_next_exl3::kProvider_flash_next_exl3;
  provider.available = false;
  provider.unavailable_reason =
      "EXL3 serving waits for the Flash-Next Stages and WeightSource "
      "interface; this build carries the format and kernels only";
  registry.AddProvider(std::move(provider));
}

void qwen38_flash_next_exl3_register(gufo::ext::Registry& registry) {
  (void)registry;
}
