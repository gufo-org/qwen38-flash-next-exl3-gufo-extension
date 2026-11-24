// Loads an EXL3 checkpoint through Flash-Next's Model::Load with the EXL3
// stages and weight source and its own MTP block, then checks greedy
// generation:
// - the autoregressive answer, and that the prompt pass and the decode path
//   agree on the next token after the same text;
// - MTP decoding of one session against the autoregressive tokens;
// - two MTP sessions decoded together, whose verification passes are wider
//   than the grouped EXL3 decode route and take the per-projection one.
// Usage: qwen38_flash_next_exl3_generate_test <bits class: 4, 3 or 2>
// QWEN38_FLASH_NEXT_EXL3_MODEL lists checkpoint directories (separated by
// ':'); the test takes the one of its bits class and skips (exit 77) when
// none is listed.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <gufo/ext/families/qwen/control_tokens/control_tokens.hpp>
#include <gufo/ext/families/qwen/models/flash_next/model.hpp>
#include <gufo/ext/sampling.hpp>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "exl3/pack.hpp"
#include "flash_next/stages.hpp"
#include "flash_next/weight_source.hpp"

namespace exl3 = gufo::ext::qwen38_flash_next_exl3;
namespace flash_next = gufo::models::qwen38_flash_next;
using gufo::core::SessionMode;

namespace {

constexpr std::uint32_t kContext = 8192;
constexpr std::size_t kTokens = 48;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

/// The listed checkpoint whose bits per weight round to `bits`.
std::string FindPack(int bits) {
  const char* env = std::getenv("QWEN38_FLASH_NEXT_EXL3_MODEL");
  std::stringstream list(env != nullptr ? env : "");
  for (std::string dir; std::getline(list, dir, ':');) {
    if (!dir.empty() && exl3::IsFlashNextPack(dir) &&
        exl3::PackBits(dir) >= bits - 0.5 && exl3::PackBits(dir) < bits + 0.5)
      return dir;
  }
  return {};
}

std::string ChatPrompt(const std::string& question) {
  using namespace gufo::tokenization;
  return std::string(kImStart) + "user\n" + question + std::string(kImEnd) +
         "\n" + std::string(kImStart) + "assistant\n<think>\n\n</think>\n\n";
}

std::int32_t Argmax(std::span<const float> logits) {
  return static_cast<std::int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

/// Greedy autoregressive tokens after `prompt`, the stop token excluded.
std::vector<std::int32_t> Autoregressive(
    flash_next::Model& model, const std::vector<std::int32_t>& prompt,
    std::string* error) {
  auto session =
      model.CreateSession(SessionMode::kAutoregressive, kContext, error);
  std::vector<std::int32_t> out;
  if (!session || !session->Sync(prompt, error))
    return out;
  while (out.size() < kTokens) {
    const std::int32_t best = Argmax(session->Logits());
    if (model.IsStopToken(best))
      break;
    out.push_back(best);
    if (!session->Evaluate(best, error))
      return {};
  }
  return out;
}

struct Speculative {
  std::unique_ptr<flash_next::Session> session;
  std::optional<gufo::sampling::SamplerState> sampler;
  std::vector<std::int32_t> out;
  bool done{false};
};

bool Start(flash_next::Model& model, const std::vector<std::int32_t>& prompt,
           Speculative* s, std::string* error) {
  s->session = model.CreateSession(SessionMode::kSpeculative, kContext, error);
  if (!s->session || !s->session->Sync(prompt, error))
    return false;
  const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                     prompt.end());
  gufo::sampling::SamplingConfig greedy;
  greedy.temperature = 0.0F;
  s->sampler.emplace(greedy, history);
  return true;
}

/// Appends a step's tokens up to kTokens, the stop token excluded.
void Append(flash_next::Model& model,
            const flash_next::Session::DecodeResult& step, Speculative* s) {
  for (const auto token : step.tokens) {
    if (model.IsStopToken(token) || s->out.size() == kTokens) {
      s->done = true;
      break;
    }
    s->out.push_back(token);
  }
  s->done = s->done || step.stop || s->out.size() == kTokens;
}

/// Greedy MTP decoding of `sessions` together until each has kTokens.
bool DecodeTogether(flash_next::Model& model,
                    std::vector<Speculative*> sessions, std::string* error) {
  for (int guard = 0; guard < 4 * static_cast<int>(kTokens); ++guard) {
    std::vector<flash_next::Session::DecodeRequest> requests;
    std::vector<flash_next::Session::DecodeResult> results(sessions.size());
    std::vector<Speculative*> active;
    for (Speculative* s : sessions) {
      if (s->done)
        continue;
      requests.push_back({.session = s->session.get(),
                          .max_tokens = kTokens - s->out.size(),
                          .sampler = &*s->sampler,
                          .result = &results[active.size()]});
      active.push_back(s);
    }
    if (active.empty())
      return true;
    const bool ok = active.size() == 1
                        ? active[0]->session->DecodeStep(requests[0].max_tokens,
                                                         *active[0]->sampler,
                                                         &results[0], error)
                        : flash_next::Session::DecodeBatch(requests, error);
    if (!ok)
      return false;
    for (std::size_t i = 0; i < active.size(); ++i)
      Append(model, results[i], active[i]);
  }
  *error = "MTP decoding made no progress";
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const int bits = argc > 1 ? std::atoi(argv[1]) : 4;
  const std::string dir = FindPack(bits);
  if (dir.empty()) {
    std::printf(
        "SKIP: no %d bpw EXL3 checkpoint in "
        "QWEN38_FLASH_NEXT_EXL3_MODEL\n",
        bits);
    return 77;
  }
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::string error;
  flash_next::ModelOptions options;
  options.max_context = kContext;
  options.mtp_model_path = dir;  // the checkpoint's own MTP block
  options.decode_concurrency = 2;
  auto model = flash_next::Model::Load(dir, options, exl3::CreateExl3Stages(),
                                       exl3::CreateExl3WeightSource(), &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf(
      "%s loaded in %.0f ms\n", dir.c_str(),
      std::chrono::duration<double, std::milli>(clock::now() - t0).count());
  const auto first = model->Tokenize(
      ChatPrompt("What is the capital of France? Answer in one sentence."));
  const auto second =
      model->Tokenize(ChatPrompt("Count from 1 to 30, separated by commas."));

  // Autoregressive answers, and the prompt pass against the decode path.
  const auto ar_first = Autoregressive(*model, first, &error);
  const auto ar_second = Autoregressive(*model, second, &error);
  if (ar_first.empty() || ar_second.empty()) {
    std::fprintf(stderr, "autoregressive: %s\n", error.c_str());
    return 1;
  }
  const std::string text = model->Decode(ar_first);
  std::printf("text: %s\n", text.c_str());
  Expect(text.find("Paris") != std::string::npos, "the answer names Paris");
  {
    auto stepped =
        model->CreateSession(SessionMode::kAutoregressive, kContext, &error);
    for (const auto id : first) {
      if (!stepped || !stepped->Evaluate(id, &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
    }
    Expect(Argmax(stepped->Logits()) == ar_first.front(),
           "prompt and decode paths agree on the next token");
  }

  // One MTP session: the same greedy tokens, from accepted drafts.
  Speculative solo;
  if (!Start(*model, first, &solo, &error) ||
      !DecodeTogether(*model, {&solo}, &error)) {
    std::fprintf(stderr, "mtp: %s\n", error.c_str());
    return 1;
  }
  const auto& stats = solo.session->Statistics();
  std::printf("mtp: %llu drafted, %llu accepted\n",
              static_cast<unsigned long long>(stats.drafted),
              static_cast<unsigned long long>(stats.accepted));
  Expect(solo.out == ar_first, "MTP tokens equal autoregressive tokens");
  Expect(stats.drafted > 0 && stats.accepted > 0, "MTP accepted drafts");

  // Two MTP sessions verified in one pass: wider than the grouped route.
  Speculative a;
  Speculative b;
  if (!Start(*model, first, &a, &error) || !Start(*model, second, &b, &error) ||
      !DecodeTogether(*model, {&a, &b}, &error)) {
    std::fprintf(stderr, "batched mtp: %s\n", error.c_str());
    return 1;
  }
  Expect(a.out == ar_first, "batched MTP session 1 equals autoregressive");
  Expect(b.out == ar_second, "batched MTP session 2 equals autoregressive");
  return failures == 0 ? 0 : 1;
}
