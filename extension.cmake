# Qwen3.8 Flash-Next EXL3: the trellis format, checkpoint reader, host
# reference and HIP linears. The three checkpoint providers are registered
# unavailable until Flash-Next exposes its Stages and WeightSource
# interface; see README.md.

# Host code: format primitives, the safetensors pack view and the scalar
# reference. pack.cpp hashes with <gufo/ext/sha256.hpp>, whose header
# includes <openssl/evp.h>; gufo_ext_core carries OpenSSL::Crypto.
gufo_extension_target(
  NAME gufo_qwen38_flash_next_exl3
  SOURCES
    extension.cpp
    exl3/pack.cpp
    exl3/reference.cpp
  LINK_LIBRARIES
    gufo_ext_core
    gufo_ext_runtime
)

if(TARGET gufo_qwen38_flash_next_exl3)
  target_include_directories(gufo_qwen38_flash_next_exl3 PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# EXL3 linears for gfx11 wave32, built with Flash-Next's HIP options.
gufo_extension_target(
  NAME gufo_qwen38_flash_next_exl3_hip
  SOURCES
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3
    gufo_ext_runtime
    hip::host
)

if(TARGET gufo_qwen38_flash_next_exl3_hip)
  gufo_extension_hip_sources(gufo_qwen38_flash_next_exl3_hip
    SOURCES
      exl3/rocm/exl3.hip.cpp
  )
  target_compile_options(gufo_qwen38_flash_next_exl3_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_qwen38_flash_next_exl3_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _qwen38_flash_next_exl3_test_libs gufo_ext_hip)
endif()

# Format contracts against independent references; no GPU or weights.
gufo_extension_target(
  NAME qwen38_flash_next_exl3_format_test
  SOURCES
    tests/exl3_format_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
)

if(TARGET qwen38_flash_next_exl3_format_test)
  add_test(NAME qwen38_flash_next_exl3.format
    COMMAND qwen38_flash_next_exl3_format_test)
  set_tests_properties(qwen38_flash_next_exl3.format PROPERTIES
    LABELS "qwen38_flash_next_exl3;cpu")
endif()

# Checkpoint reader on tiny synthetic safetensors directories; no GPU.
gufo_extension_target(
  NAME qwen38_flash_next_exl3_pack_test
  SOURCES
    tests/exl3_pack_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
)

if(TARGET qwen38_flash_next_exl3_pack_test)
  add_test(NAME qwen38_flash_next_exl3.pack
    COMMAND qwen38_flash_next_exl3_pack_test)
  set_tests_properties(qwen38_flash_next_exl3.pack PROPERTIES
    LABELS "qwen38_flash_next_exl3;cpu")
endif()

# HIP linears against the host reference. The fused SwiGLU check compares
# with Flash-Next's own Swiglu kernel, so this test links its export.
gufo_extension_target(
  NAME qwen38_flash_next_exl3_ops_test
  SOURCES
    tests/exl3_ops_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3_hip
    gufo_qwen38_flash_next_exl3
    gufo::qwen38_flash_next_hip
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
    hip::host
)

if(TARGET qwen38_flash_next_exl3_ops_test)
  add_test(NAME qwen38_flash_next_exl3.ops
    COMMAND qwen38_flash_next_exl3_ops_test)
  set_tests_properties(qwen38_flash_next_exl3.ops PROPERTIES
    LABELS "qwen38_flash_next_exl3;hip;gpu;gfx1151;slow")
endif()

# Decode and prompt-pass linear timings on random weights; run by hand.
gufo_extension_target(
  NAME qwen38_flash_next_exl3_gemv_bench
  SOURCES
    tests/exl3_gemv_bench.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3_hip
    gufo_qwen38_flash_next_exl3
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
    hip::host
)

install(FILES
    "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
    "${GUFO_EXTENSION_BUILD_ROOT}/THIRD_PARTY_NOTICES.md"
  DESTINATION share/licenses/gufo/extensions/qwen38_flash_next_exl3)
install(DIRECTORY "${GUFO_EXTENSION_BUILD_ROOT}/licenses/"
  DESTINATION share/licenses/gufo/extensions/qwen38_flash_next_exl3/licenses)
