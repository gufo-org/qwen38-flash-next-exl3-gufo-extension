// EXL3 checkpoint reader on tiny synthetic directories: a valid two-file
// pack, EXL3 detection and config queries, the identity's inputs, and the
// malformed headers Pack::Open must reject (truncation, offsets past the
// file, offsets and shapes whose arithmetic wraps, duplicate tensors, no
// config.json).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "exl3/pack.hpp"

namespace fs = std::filesystem;
namespace fmt = gufo::ext::qwen38_flash_next_exl3;

namespace {

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

constexpr const char* kExl3Config =
    R"({"model_type":"qwen4_exp","quantization_config":)"
    R"({"quant_method":"exl3","bits":4.05},)"
    R"("text_config":{"max_position_embeddings":262144}})";

/// A fresh, empty directory under the system temporary directory.
class TempDir {
public:
  explicit TempDir(const std::string& name) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = fs::temp_directory_path() /
            ("exl3_pack_test_" + name + "_" + std::to_string(stamp));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  [[nodiscard]] const fs::path& path() const { return path_; }

private:
  fs::path path_;
};

void WriteText(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

/// A safetensors file: little-endian header length, JSON header, payload.
void WriteSafetensors(const fs::path& path, const std::string& header,
                      std::size_t payload_bytes) {
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  std::uint8_t len[8];
  for (int i = 0; i < 8; ++i)
    len[i] = static_cast<std::uint8_t>(n >> (8 * i));
  out.write(reinterpret_cast<const char*>(len), sizeof(len));
  out << header;
  const std::vector<char> payload(payload_bytes, '\x5a');
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

std::string Entry(const std::string& name, const std::string& dtype,
                  const std::string& shape, std::uint64_t begin,
                  std::uint64_t end) {
  return "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape +
         ",\"data_offsets\":[" + std::to_string(begin) + "," +
         std::to_string(end) + "]}";
}

/// Opens a pack whose only file has `header` and `payload` bytes of data.
bool OpensWith(const std::string& name, const std::string& header,
               std::size_t payload, std::string* error) {
  TempDir dir(name);
  WriteText(dir.path() / "config.json", kExl3Config);
  WriteSafetensors(dir.path() / "model.safetensors", header, payload);
  return fmt::Pack::Open(dir.path(), error) != nullptr;
}

void ValidPack() {
  TempDir dir("valid");
  WriteText(dir.path() / "config.json", kExl3Config);
  WriteSafetensors(dir.path() / "a.safetensors",
                   "{\"__metadata__\":{\"format\":\"pt\"}," +
                       Entry("w.trellis", "I16", "[2,3]", 0, 12) + "," +
                       Entry("w.suh", "F16", "[4]", 12, 20) + "}",
                   20);
  WriteSafetensors(dir.path() / "b.safetensors",
                   "{" + Entry("lm_head", "BF16", "[2,2]", 0, 8) + "}", 8);

  Expect(fmt::IsFlashNextPack(dir.path()), "EXL3 directory detected");
  Expect(fmt::PackBits(dir.path()) == 4.05, "bits read");
  Expect(fmt::PackContextLength(dir.path()) == 262144,
         "context length read from text_config");

  std::string error;
  const auto pack = fmt::Pack::Open(dir.path(), &error);
  Expect(pack != nullptr, "valid pack opens: " + error);
  if (pack == nullptr)
    return;
  Expect(pack->TensorCount() == 3, "metadata entry is not a tensor");
  Expect(pack->Regions().size() == 2, "one region per file");
  const fmt::PackTensor* trellis = pack->Find("w.trellis");
  Expect(trellis != nullptr && trellis->dtype == fmt::DType::kI16 &&
             trellis->shape == std::vector<std::uint64_t>{2, 3} &&
             trellis->size_bytes == 12 && trellis->file == 0,
         "trellis entry");
  const fmt::PackTensor* suh = pack->Find("w.suh");
  Expect(suh != nullptr && suh->file_offset > trellis->file_offset &&
             suh->file_offset - trellis->file_offset == 12,
         "payload offsets follow the header");
  const fmt::PackTensor* head = pack->Find("lm_head");
  Expect(head != nullptr && head->file == 1 && head->size_bytes == 8,
         "second file indexed");
  if (head != nullptr) {
    const auto* bytes = static_cast<const std::uint8_t*>(head->data);
    Expect(bytes[0] == 0x5a && bytes[7] == 0x5a, "payload mapped");
  }
  Expect(pack->Find("missing") == nullptr, "unknown tensor");
  Expect(pack->Identity().size() == 64, "identity is a sha256 hex digest");

  // The identity covers config.json, the headers and a payload sample whose
  // last block ends each file.
  const std::string identity = pack->Identity();
  WriteSafetensors(dir.path() / "b.safetensors",
                   "{" + Entry("lm_head", "BF16", "[2,2]", 0, 8) + "}", 8);
  {
    std::fstream f(dir.path() / "b.safetensors",
                   std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(-1, std::ios::end);
    f.put('\x01');
  }
  const auto same_headers = fmt::Pack::Open(dir.path(), &error);
  Expect(same_headers != nullptr && same_headers->Identity() != identity,
         "payload change under the same headers changes the identity");
  WriteText(dir.path() / "config.json",
            std::string(kExl3Config)
                .replace(std::string(kExl3Config).find("4.05"), 4, "3.05"));
  const auto other_config = fmt::Pack::Open(dir.path(), &error);
  Expect(other_config != nullptr && other_config->Identity() != identity,
         "config change changes the identity");
}

void Detection() {
  TempDir dir("detect");
  Expect(!fmt::IsFlashNextPack(dir.path()), "no config.json: not a pack");
  Expect(!fmt::IsFlashNextPack(dir.path() / "absent"), "missing directory");
  Expect(fmt::PackBits(dir.path()) == 0.0, "no config.json: no bits");
  Expect(fmt::PackContextLength(dir.path()) == 0,
         "no config.json: no context length");
  WriteText(dir.path() / "config.json",
            R"({"model_type":"qwen4_exp","max_position_embeddings":4096})");
  Expect(!fmt::IsFlashNextPack(dir.path()), "unquantized checkpoint rejected");
  Expect(fmt::PackContextLength(dir.path()) == 4096,
         "context length without text_config");
  WriteText(dir.path() / "config.json",
            R"({"model_type":"qwen3","quantization_config":)"
            R"({"quant_method":"exl3"}})");
  Expect(!fmt::IsFlashNextPack(dir.path()), "other model type rejected");
  WriteText(dir.path() / "config.json", "{not json");
  Expect(!fmt::IsFlashNextPack(dir.path()), "invalid config rejected");
}

void Malformed() {
  std::string error;
  {
    TempDir dir("no_config");
    WriteSafetensors(dir.path() / "model.safetensors",
                     "{" + Entry("w", "F16", "[2]", 0, 4) + "}", 4);
    Expect(fmt::Pack::Open(dir.path(), &error) == nullptr &&
               error.find("config.json") != std::string::npos,
           "missing config.json rejected");
  }
  {
    TempDir dir("no_files");
    WriteText(dir.path() / "config.json", kExl3Config);
    Expect(fmt::Pack::Open(dir.path(), &error) == nullptr,
           "directory without safetensors rejected");
  }
  {
    TempDir dir("short");
    WriteText(dir.path() / "config.json", kExl3Config);
    WriteText(dir.path() / "model.safetensors", "abc");
    Expect(fmt::Pack::Open(dir.path(), &error) == nullptr,
           "file shorter than its header length rejected");
  }
  {
    // Header length past the end of the file.
    TempDir dir("header");
    WriteText(dir.path() / "config.json", kExl3Config);
    WriteSafetensors(dir.path() / "model.safetensors", "{}", 0);
    {
      std::fstream f(dir.path() / "model.safetensors",
                     std::ios::binary | std::ios::in | std::ios::out);
      f.put('\x7f');
    }
    Expect(fmt::Pack::Open(dir.path(), &error) == nullptr,
           "header length past the file rejected");
  }
  Expect(!OpensWith("truncated", "{" + Entry("w", "F16", "[4]", 0, 8) + "}", 6,
                    &error),
         "truncated payload rejected");
  Expect(
      !OpensWith("size", "{" + Entry("w", "F16", "[4]", 0, 6) + "}", 8, &error),
      "payload size differing from the shape rejected");
  Expect(!OpensWith("reversed", "{" + Entry("w", "F16", "[0]", 8, 0) + "}", 8,
                    &error),
         "end before begin rejected");
  // end = 2^64 - 2048 is the largest offset JSON numbers carry exactly; a
  // header over 2048 bytes makes data_start + end wrap to a small value,
  // which passes an unchecked bound.
  Expect(!OpensWith("offset",
                    "{\"__metadata__\":{\"pad\":\"" + std::string(4096, 'p') +
                        "\"}," +
                        Entry("w", "F16", "[1024]", 18446744073709547520ULL,
                              18446744073709549568ULL) +
                        "}",
                    4096, &error),
         "offset whose sum wraps rejected");
  // 2^32 * 2^32 elements wrap to zero bytes, matching begin == end.
  Expect(
      !OpensWith("shape",
                 "{" + Entry("w", "F16", "[4294967296,4294967296]", 0, 0) + "}",
                 8, &error),
      "shape whose byte count wraps rejected");
  Expect(!OpensWith("dtype", "{" + Entry("w", "F8_E4M3", "[4]", 0, 4) + "}", 4,
                    &error),
         "unsupported dtype rejected");
  Expect(!OpensWith("incomplete", R"({"w":{"dtype":"F16","shape":[2]}})", 4,
                    &error),
         "entry without data_offsets rejected");
  {
    TempDir dir("duplicate");
    WriteText(dir.path() / "config.json", kExl3Config);
    WriteSafetensors(dir.path() / "a.safetensors",
                     "{" + Entry("w", "F16", "[2]", 0, 4) + "}", 4);
    WriteSafetensors(dir.path() / "b.safetensors",
                     "{" + Entry("w", "F16", "[2]", 0, 4) + "}", 4);
    Expect(fmt::Pack::Open(dir.path(), &error) == nullptr &&
               error.find("duplicate") != std::string::npos,
           "tensor named in two files rejected");
  }
}

}  // namespace

int main() {
  ValidPack();
  Detection();
  Malformed();
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("exl3 pack: ok\n");
  return 0;
}
