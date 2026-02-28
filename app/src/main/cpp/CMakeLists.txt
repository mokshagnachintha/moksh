cmake_minimum_required(VERSION 3.22.1)

project(llama_jni)

# Use ccache if available (speeds up CI rebuilds of llama.cpp)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

include(FetchContent)

# ── Disable everything not needed for CPU-only Android inference ───────────
set(BUILD_SHARED_LIBS         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL                OFF CACHE BOOL "" FORCE)
set(LLAMA_NATIVE              OFF CACHE BOOL "" FORCE)
set(GGML_CUDA                 OFF CACHE BOOL "" FORCE)
set(GGML_METAL                OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN               OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP               OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE               OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS          OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES       OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL           OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE            OFF CACHE BOOL "" FORCE)
# ───────────────────────────────────────────────────────────────────────────

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b4963
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(llama_cpp)

add_library(llama_jni SHARED llama_jni.cpp)

find_library(log-lib log)

target_include_directories(llama_jni PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

# ggml is linked transitively through llama — do not add it explicitly
target_link_libraries(llama_jni
    llama
    ${log-lib}
)

# ── ARM64 vectorisation flags ─────────────────────────────────────────────
# dotprod  : 8-bit dot-product — 4× faster Q4 matrix-vector multiply
# i8mm     : 8-bit integer matrix multiply (Snapdragon 865+ / Dimensity 1200+)
# fp16     : native 16-bit float arithmetic
# -O3      : full compiler optimisation (default NDK is -O2)
# Only applied for arm64-v8a; other ABIs are already filtered out.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(llama_jni PRIVATE
        -O3
        -march=armv8.4-a+dotprod+i8mm+fp16
    )
    # Apply the same flags to ggml so the hot kernels benefit too
    target_compile_options(ggml PRIVATE
        -O3
        -march=armv8.4-a+dotprod+i8mm+fp16
    )
endif()
