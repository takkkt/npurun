# npurun: llama.cpp with AMD XDNA 2 NPU offload for ternary GGUF models.
#
# This repository holds only the NPU integration. llama.cpp (PrismML's fork, which
# adds the ternary PQ2_0 / PTQ1_0 formats) is a pinned submodule in
# third_party/llama.cpp; patches/llama.cpp.patch adds the small hooks npurun needs.
cmake_minimum_required(VERSION 3.21)
project(npurun C CXX)

if (NOT WIN32)
    message(FATAL_ERROR "npurun supports Windows only so far")
endif()

set(NPURUN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(LLAMA_DIR   ${NPURUN_ROOT}/third_party/llama.cpp)
set(NPURUN_XRT_DIR "" CACHE PATH "Local XRT SDK (contains include/ and lib/); empty = download")

find_package(Git REQUIRED)

# --- llama.cpp submodule --------------------------------------------------------
if (NOT EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    message(STATUS "npurun: fetching the llama.cpp submodule")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --depth 1 third_party/llama.cpp
                    WORKING_DIRECTORY ${NPURUN_ROOT} RESULT_VARIABLE rc)
    if (NOT rc EQUAL 0)
        message(FATAL_ERROR "npurun: could not fetch third_party/llama.cpp; run: git submodule update --init")
    endif()
endif()

# --- npurun hooks into llama.cpp (idempotent) -----------------------------------
set(NPURUN_PATCH ${NPURUN_ROOT}/patches/llama.cpp.patch)
execute_process(COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${NPURUN_PATCH}
                WORKING_DIRECTORY ${LLAMA_DIR} RESULT_VARIABLE already OUTPUT_QUIET ERROR_QUIET)
if (NOT already EQUAL 0)
    execute_process(COMMAND ${GIT_EXECUTABLE} apply --whitespace=nowarn ${NPURUN_PATCH}
                    WORKING_DIRECTORY ${LLAMA_DIR} RESULT_VARIABLE rc ERROR_VARIABLE err)
    if (NOT rc EQUAL 0)
        message(FATAL_ERROR "npurun: patches/llama.cpp.patch does not apply to third_party/llama.cpp:\n${err}"
                            "Reset the submodule (git submodule update --init --force) and configure again.")
    endif()
    message(STATUS "npurun: applied patches/llama.cpp.patch")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NPURUN_PATCH})

# --- llama.cpp ------------------------------------------------------------------
# As a subproject llama.cpp builds only its libraries by default; we ship all of its
# programs, like its own releases do.
foreach(opt COMMON TOOLS SERVER EXAMPLES APP)
    set(LLAMA_BUILD_${opt} ON CACHE BOOL "" FORCE)
endforeach()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# `--version` reports the npurun version (git tag / commit of this repository)
execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                WORKING_DIRECTORY ${NPURUN_ROOT} OUTPUT_VARIABLE NPURUN_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if (NOT NPURUN_VERSION)
    set(NPURUN_VERSION unknown)
endif()
set(LLAMA_BUILD_COMMIT "npurun-${NPURUN_VERSION}")
message(STATUS "npurun: version ${NPURUN_VERSION}")

add_subdirectory(${LLAMA_DIR} llama.cpp)

# --- NPU backend (compiled into ggml-cpu) and developer tools ---------------------
include(cmake/npu-backend.cmake)
add_subdirectory(tools)
