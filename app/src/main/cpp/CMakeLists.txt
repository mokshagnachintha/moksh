cmake_minimum_required(VERSION 3.22.1)

project(llama_jni)

# We will need to link against the actual llama.cpp library here when added.
# For now, we compile the JNI interface as a shared library.
add_library(llama_jni SHARED
            llama_jni.cpp)

# Native Android logging library
find_library(log-lib log)

target_link_libraries(llama_jni
                      ${log-lib})
