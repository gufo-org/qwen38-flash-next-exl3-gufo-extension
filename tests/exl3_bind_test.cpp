// Binds an EXL3 Flash-Next checkpoint on the host and checks the descriptors
// Flash-Next receives: geometry, trellis linears carrying format_state, the
// MTP block and the n-gram table. Prints the geometry and the timings.
// Usage: qwen38_flash_next_exl3_bind_test <bits class: 4, 3 or 2>
// QWEN38_FLASH_NEXT_EXL3_MODEL lists checkpoint directories (separated by
// ':'); the test takes the one of its bits class and skips (exit 77) when
// none is listed.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "exl3/pack.hpp"
#include "flash_next/bind.hpp"

namespace exl3 = gufo::ext::qwen38_flash_next_exl3;

int main(int argc, char** argv) {
  const int bits = argc > 1 ? std::atoi(argv[1]) : 4;
  const char* env = std::getenv("QWEN38_FLASH_NEXT_EXL3_MODEL");
  std::stringstream list(env != nullptr ? env : "");
  std::string dir;
  for (std::string entry; std::getline(list, entry, ':');) {
    if (!entry.empty() && exl3::IsFlashNextPack(entry) &&
        exl3::PackBits(entry) >= bits - 0.5 &&
        exl3::PackBits(entry) < bits + 0.5)
      dir = entry;
  }
  if (dir.empty()) {
    std::printf(
        "SKIP: no %d bpw EXL3 checkpoint in "
        "QWEN38_FLASH_NEXT_EXL3_MODEL\n",
        bits);
    return 77;
  }
  const auto start = std::chrono::steady_clock::now();
  std::string error;
  auto pack = exl3::Pack::Open(dir, &error);
  if (!pack) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  const auto opened = std::chrono::steady_clock::now();
  exl3::HostTensors host;
  exl3::NgramRef ngram;
  auto w = exl3::BindTrunk(*pack, host, &ngram, &error);
  if (!w) {
    std::fprintf(stderr, "bind: %s\n", error.c_str());
    return 1;
  }
  auto mtp = exl3::BindMtp(*pack, w->config, host, &error);
  if (!mtp) {
    std::fprintf(stderr, "mtp: %s\n", error.empty() ? "none" : error.c_str());
    return 1;
  }
  const auto bound = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const exl3::Config& c = w->config;
  const auto* head = exl3::RefOf(w->output);
  const auto* experts = exl3::RefOf(w->layers[0].ffn_gate_exps);
  const auto* dense = exl3::RefOf(w->layers[0].ssm_qkv);
  int failures = 0;
  const auto expect = [&](bool ok, const char* what) {
    if (!ok) {
      std::fprintf(stderr, "FAIL: %s\n", what);
      ++failures;
    }
  };
  expect(c.num_layers == 48 && c.hidden_size == 2560, "trunk geometry");
  expect(head != nullptr && w->output.rows == c.vocab_size, "trellis head");
  expect(experts != nullptr && experts->parts.size() == c.num_experts,
         "trellis expert stacks");
  expect(dense != nullptr && w->layers[0].ssm_qkv.cols == c.hidden_size,
         "trellis dense projections");
  expect(w->ple_table.empty() && c.ple_layer >= 0 && ngram.rows >= c.ple_rows &&
             ngram.k != 0,
         "trellis n-gram table");
  expect(exl3::RefOf(mtp->block.nextn_fc_hidden) != nullptr, "MTP draft block");
  std::printf("tensors %zu identity %s\n", pack->TensorCount(),
              pack->Identity().c_str());
  std::printf("layers %u hidden %u vocab %u ctx %u ple_layer %d rows %llu\n",
              c.num_layers, c.hidden_size, c.vocab_size, c.context_length,
              c.ple_layer, static_cast<unsigned long long>(c.ple_rows));
  std::printf("head K%d expert K%d dense K%d ngram K%d\n",
              head != nullptr ? head->k : 0,
              experts != nullptr ? experts->k : 0,
              dense != nullptr ? dense->k : 0, ngram.k);
  std::size_t host_bytes = 0;
  for (const auto& b : host.buffers)
    host_bytes += b.size();
  std::printf("host conversions %.1f MiB, open %.0f ms, bind %.0f ms\n",
              host_bytes / 1048576.0, ms(start, opened), ms(opened, bound));
  return failures == 0 ? 0 : 1;
}
