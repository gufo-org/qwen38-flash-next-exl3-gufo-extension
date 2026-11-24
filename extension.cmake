# Qwen3.8 Flash-Next EXL3: the trellis format, checkpoint reader, host
# reference and HIP linears, and the Flash-Next stages and weight source
# that serve the three checkpoints through the qwen38_flash_next extension.

# Host code: format primitives, the safetensors pack view, the scalar
# reference and the binding to Flash-Next's weight descriptors. pack.cpp
# hashes with <gufo/ext/sha256.hpp>, whose header includes <openssl/evp.h>;
# gufo_ext_core carries OpenSSL::Crypto.
gufo_extension_target(
  NAME gufo_qwen38_flash_next_exl3
  SOURCES
    extension.cpp
    exl3/pack.cpp
    exl3/reference.cpp
    flash_next/bind.cpp
  LINK_LIBRARIES
    gufo::qwen38_flash_next
    gufo_ext_core
    gufo_ext_runtime
)

if(TARGET gufo_qwen38_flash_next_exl3)
  target_include_directories(gufo_qwen38_flash_next_exl3 PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# EXL3 linears for gfx11 wave32, built with Flash-Next's HIP options, and
# the providers: the stages and the weight source Flash-Next's runner takes.
gufo_extension_target(
  NAME gufo_qwen38_flash_next_exl3_hip
  SOURCES
    flash_next/provider.cpp
    flash_next/stages.cpp
    flash_next/weight_source.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3
    gufo::qwen38_flash_next_hip
    gufo::qwen_family
    gufo_ext_runtime
    hip::host
)

if(TARGET gufo_qwen38_flash_next_exl3_hip)
  gufo_extension_hip_sources(gufo_qwen38_flash_next_exl3_hip
    SOURCES
      exl3/rocm/exl3.hip.cpp
  )
  target_compile_definitions(gufo_qwen38_flash_next_exl3_hip PRIVATE
    ENGINE_ENABLE_HIP=1)
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
# with an unfused reference kernel of the test's own.
gufo_extension_target(
  NAME qwen38_flash_next_exl3_ops_test
  SOURCES
    tests/exl3_ops_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3_hip
    gufo_qwen38_flash_next_exl3
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
    hip::host
)

if(TARGET qwen38_flash_next_exl3_ops_test)
  # The reference SwiGLU kernel, with the linears' HIP options.
  gufo_extension_hip_sources(qwen38_flash_next_exl3_ops_test
    SOURCES
      tests/exl3_swiglu.hip.cpp
  )
  target_compile_options(qwen38_flash_next_exl3_ops_test PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
  add_test(NAME qwen38_flash_next_exl3.ops
    COMMAND qwen38_flash_next_exl3_ops_test)
  set_tests_properties(qwen38_flash_next_exl3.ops PROPERTIES
    LABELS "qwen38_flash_next_exl3;hip;gpu;gfx1151;slow")
endif()

# Real checkpoints (QWEN38_FLASH_NEXT_EXL3_MODEL lists their directories,
# ':'-separated), one case per bits class: the binding on the host, then
# greedy AR and MTP generation through Flash-Next. Each skips without its
# checkpoint.
gufo_extension_target(
  NAME qwen38_flash_next_exl3_bind_test
  SOURCES
    tests/exl3_bind_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3
    gufo::qwen38_flash_next
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
)

if(TARGET qwen38_flash_next_exl3_bind_test)
  foreach(bits 4 3 2)
    add_test(NAME qwen38_flash_next_exl3.bind_${bits}bpw
      COMMAND qwen38_flash_next_exl3_bind_test ${bits})
    set_tests_properties(qwen38_flash_next_exl3.bind_${bits}bpw PROPERTIES
      LABELS "qwen38_flash_next_exl3;cpu;model"
      SKIP_RETURN_CODE 77)
  endforeach()
endif()

gufo_extension_target(
  NAME qwen38_flash_next_exl3_generate_test
  SOURCES
    tests/exl3_generate_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_exl3_hip
    gufo_qwen38_flash_next_exl3
    gufo::qwen38_flash_next_hip
    gufo_ext_runtime
    ${_qwen38_flash_next_exl3_test_libs}
    hip::host
)

if(TARGET qwen38_flash_next_exl3_generate_test)
  target_compile_definitions(qwen38_flash_next_exl3_generate_test PRIVATE
    ENGINE_ENABLE_HIP=1)
  foreach(bits 4 3 2)
    add_test(NAME qwen38_flash_next_exl3.generate_${bits}bpw
      COMMAND qwen38_flash_next_exl3_generate_test ${bits})
    set_tests_properties(qwen38_flash_next_exl3.generate_${bits}bpw PROPERTIES
      LABELS "qwen38_flash_next_exl3;hip;gpu;gfx1151;model;slow"
      SKIP_RETURN_CODE 77)
  endforeach()
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
