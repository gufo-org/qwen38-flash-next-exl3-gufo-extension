// Registration of the EXL3 extension. The three checkpoint providers
// register from the HIP library (flash_next/provider.cpp); this entry has
// nothing of its own to add.
#include <gufo/ext/registry.hpp>

void qwen38_flash_next_exl3_register(gufo::ext::Registry& registry) {
  (void)registry;
}
