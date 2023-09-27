cmake_minimum_required(VERSION 3.18) # 3.18 or later is required for CUDA_ARCHITECTURES support
project(TropicalGEMM CXX CUDA)

set(CUDA_FILES tropicalgemm_kernels.cu)
set(LIBRARY_NAME kernels.so)

set(MACRO_COMBINATIONS
    PlusMul_FP32
    PlusMul_FP64
    PlusMul_INT32
    PlusMul_INT64
    TropicalAndOr_Bool
    TropicalMaxMul_FP32
    TropicalMaxMul_FP64
    TropicalMaxMul_INT32
    TropicalMaxMul_INT64
    TropicalMaxPlus_FP32
    TropicalMaxPlus_FP64
    TropicalMinPlus_FP32
    TropicalMinPlus_FP64
)

# Set the desired GPU architectures, e.g., 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Set the output directory for the shared libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

foreach(MACRO IN LISTS MACRO_COMBINATIONS)
    string(REPLACE "_" ";" MACRO_SPLIT ${MACRO})
    list(GET MACRO_SPLIT 0 MACRO_1)
    list(GET MACRO_SPLIT 1 MACRO_2)

    set(TARGET_NAME "_${MACRO}")
    add_library(${TARGET_NAME} SHARED ${CUDA_FILES})
    target_compile_definitions(${TARGET_NAME} PRIVATE ${MACRO_1} ${MACRO_2})
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()