// Loads an EXL3 checkpoint through Flash-Next's Model::Load with the EXL3
// stages and weight source, prefills a chat prompt and greedily decodes,
// then checks the answer and that the prompt pass and the decode path agree
// on the next token after the same text.
// Usage: qwen38_flash_next_exl3_generate_test [checkpoint] [tokens] [prompt]
// Without a checkpoint argument it reads QWEN38_FLASH_NEXT_EXL3_MODEL and
// skips (exit 77) when neither names a checkpoint.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "exl3/pack.hpp"
#include "flash_next/stages.hpp"
#include "flash_next/weight_source.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace exl3 = gufo::ext::qwen38_flash_next_exl3;
namespace flash_next = gufo::models::qwen38_flash_next;

int main(int argc, char** argv) {
  const char* env = std::getenv("QWEN38_FLASH_NEXT_EXL3_MODEL");
  const std::string dir = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (dir.empty() || !exl3::IsFlashNextPack(dir)) {
    std::printf("SKIP: no EXL3 checkpoint (QWEN38_FLASH_NEXT_EXL3_MODEL)\n");
    return 77;
  }
  const int steps = argc > 2 ? std::atoi(argv[2]) : 32;
  const bool own_prompt = argc > 3;
  const std::string question =
      own_prompt ? argv[3]
                 : "What is the capital of France? Answer in one sentence.";
  const std::string prompt = "<|im_start|>user\n" + question +
                             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                             "</think>\n\n";
  using clock = std::chrono::steady_clock;
  const auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto t0 = clock::now();
  std::string error;
  flash_next::ModelOptions options;
  options.max_context = 8192;
  auto model = flash_next::Model::Load(dir, options, exl3::CreateExl3Stages(),
                                       exl3::CreateExl3WeightSource(), &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  const auto t1 = clock::now();
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      8192, &error);
  if (!session) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }
  const std::vector<std::int32_t> tokens = model->Tokenize(prompt);
  const auto t2 = clock::now();
  if (!session->Sync(tokens, &error)) {
    std::fprintf(stderr, "prefill: %s\n", error.c_str());
    return 1;
  }
  const auto t3 = clock::now();
  const auto argmax = [](std::span<const float> logits) {
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
  };
  std::vector<std::int32_t> out;
  for (int s = 0; s < steps; ++s) {
    const auto logits = session->Logits();
    if (s == 0) {
      std::vector<int> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](int a, int b) { return logits[a] > logits[b]; });
      std::printf("top5:");
      for (int i = 0; i < 5; ++i)
        std::printf(" %d(%s)=%.3f", order[i],
                    model->TokenText(order[i]).c_str(), logits[order[i]]);
      std::printf("\n");
    }
    const std::int32_t best = argmax(logits);
    out.push_back(best);
    if (model->IsStopToken(best))
      break;
    if (!session->Evaluate(best, &error)) {
      std::fprintf(stderr, "decode: %s\n", error.c_str());
      return 1;
    }
  }
  const auto t4 = clock::now();
  const std::string text = model->Decode(out);
  std::printf("text: %s\n", text.c_str());
  std::printf(
      "load %.0f ms, prefill %zu tok %.1f ms (%.1f tok/s), decode %zu tok "
      "%.1f ms (%.2f tok/s)\n",
      ms(t0, t1), tokens.size(), ms(t2, t3),
      tokens.size() / (ms(t2, t3) / 1000.0), out.size(), ms(t3, t4),
      out.size() / (ms(t3, t4) / 1000.0));
  if (own_prompt)
    return 0;
  int failures = 0;
  if (text.find("Paris") == std::string::npos) {
    std::fprintf(stderr, "FAIL: the answer does not name Paris\n");
    ++failures;
  }
  // The prompt pass (matrix-core and Q8_0 routes) against the decode path
  // (trellis GEMVs), token by token over the same text.
  auto other = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                    8192, &error);
  if (!other) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }
  for (const auto id : tokens) {
    if (!other->Evaluate(id, &error)) {
      std::fprintf(stderr, "decode: %s\n", error.c_str());
      return 1;
    }
  }
  const std::int32_t decoded = argmax(other->Logits());
  std::printf("prefill vs decode next token: %d / %d\n", out.front(), decoded);
  if (decoded != out.front()) {
    std::fprintf(stderr, "FAIL: prompt and decode paths disagree\n");
    ++failures;
  }
  return failures == 0 ? 0 : 1;
}
