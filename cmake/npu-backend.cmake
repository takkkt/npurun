# Attach the NPU extra buffer type (src/npu) to llama.cpp's ggml-cpu backend.
set(NPU_DIR ${NPURUN_ROOT}/src/npu)

if (NOT TARGET ggml-cpu)
    message(FATAL_ERROR "npurun: ggml-cpu target not found (GGML_BACKEND_DL / CPU variants are not supported)")
endif()

# XRT SDK: headers + import library. The runtime DLL (xrt_coreutil.dll) ships with
# the AMD NPU driver, so nothing is installed or redistributed.
set(NPU_XRT_DIR "${NPURUN_XRT_DIR}")
if (NOT NPU_XRT_DIR)
    set(NPU_XRT_ROOT ${CMAKE_BINARY_DIR}/_deps/xrt_sdk)
    set(NPU_XRT_DIR  ${NPU_XRT_ROOT}/xrt_sdk/xrt)
    if (NOT EXISTS ${NPU_XRT_DIR}/lib/xrt_coreutil.lib)
        set(NPU_XRT_URL "https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip")
        message(STATUS "npurun: downloading the XRT SDK from ${NPU_XRT_URL}")
        file(DOWNLOAD ${NPU_XRT_URL} ${NPU_XRT_ROOT}/xrt_windows_sdk.zip STATUS dl_status SHOW_PROGRESS)
        list(GET dl_status 0 dl_code)
        if (NOT dl_code EQUAL 0)
            message(FATAL_ERROR "npurun: XRT SDK download failed (${dl_status}); "
                                "download it manually and pass -DNPURUN_XRT_DIR=<dir containing include/ and lib/>")
        endif()
        file(ARCHIVE_EXTRACT INPUT ${NPU_XRT_ROOT}/xrt_windows_sdk.zip DESTINATION ${NPU_XRT_ROOT})
    endif()
endif()
if (NOT EXISTS ${NPU_XRT_DIR}/lib/xrt_coreutil.lib)
    message(FATAL_ERROR "npurun: no XRT SDK at ${NPU_XRT_DIR}")
endif()

# Embed the NPU core program so binaries need no extra files.
set(NPU_GEN_DIR    ${CMAKE_BINARY_DIR}/npu)
set(NPU_XCLBIN     ${NPU_DIR}/tgemv.xclbin)
set(NPU_XCLBIN_INC ${NPU_GEN_DIR}/npurun_xclbin.inc)
if (NOT EXISTS ${NPU_XCLBIN_INC} OR ${NPU_XCLBIN} IS_NEWER_THAN ${NPU_XCLBIN_INC})
    file(READ ${NPU_XCLBIN} npu_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," npu_hex "${npu_hex}")
    file(WRITE ${NPU_XCLBIN_INC} "static const unsigned char npurun_xclbin[] = {${npu_hex}};\n")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NPU_XCLBIN})

message(STATUS "npurun: AMD XDNA2 NPU offload enabled (XRT SDK: ${NPU_XRT_DIR})")
target_compile_definitions(ggml-cpu PRIVATE GGML_USE_CPU_NPU GGML_NPU_EMBEDDED_XCLBIN)
target_sources(ggml-cpu PRIVATE ${NPU_DIR}/npu.cpp ${NPU_DIR}/npurun.cpp)
# src/ for the patched ggml-cpu.cpp's #include "npu/npu.h"
target_include_directories(ggml-cpu PRIVATE
    ${NPURUN_ROOT}/src ${NPU_DIR} ${NPU_DIR}/vendor ${NPU_GEN_DIR} ${NPU_XRT_DIR}/include)
target_link_libraries(ggml-cpu PRIVATE ${NPU_XRT_DIR}/lib/xrt_coreutil.lib)
