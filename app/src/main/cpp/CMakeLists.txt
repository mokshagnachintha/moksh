cmake_minimum_required(VERSION 3.22.1)

project(llama_jni)

include(FetchContent)

# ── llama.cpp ──────────────────────────────────────────────────────────────
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b4693
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(llama_cpp)
# ───────────────────────────────────────────────────────────────────────────

add_library(llama_jni SHARED llama_jni.cpp)

find_library(log-lib log)

target_include_directories(llama_jni PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(llama_jni
    llama
    ggml
    ${log-lib}
)
