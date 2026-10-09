cmake_minimum_required(VERSION 3.12)

project(mobilenet_gtsrb C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# Target 1: Golden Baseline Single Core (No SWIFI)
add_executable(mobilenet_gtsrb_golden_single_core "")
target_include_directories(mobilenet_gtsrb_golden_single_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/firmware)
set_target_properties(mobilenet_gtsrb_golden_single_core PROPERTIES 
  COMPILE_FLAGS "-fno-rtti -fno-exceptions -fno-threadsafe-statics -nostdlib"
)
target_sources(mobilenet_gtsrb_golden_single_core PRIVATE 
  ${CMAKE_CURRENT_LIST_DIR}/firmware/main_golden.cpp
  ${CMAKE_CURRENT_LIST_DIR}/firmware/model_settings.cpp
)
target_link_libraries(mobilenet_gtsrb_golden_single_core pico-tflmicro pico_stdlib)
pico_enable_stdio_usb(mobilenet_gtsrb_golden_single_core 1)
pico_enable_stdio_uart(mobilenet_gtsrb_golden_single_core 1)
pico_add_extra_outputs(mobilenet_gtsrb_golden_single_core)

# Target 2: Batch SWIFI (Expects serial commands)
add_executable(mobilenet_gtsrb_batch "")
target_include_directories(mobilenet_gtsrb_batch PRIVATE ${CMAKE_CURRENT_LIST_DIR}/firmware)
set_target_properties(mobilenet_gtsrb_batch PROPERTIES 
  COMPILE_FLAGS "-fno-rtti -fno-exceptions -fno-threadsafe-statics -nostdlib"
)
target_sources(mobilenet_gtsrb_batch PRIVATE 
  ${CMAKE_CURRENT_LIST_DIR}/firmware/main_batch.cpp
  ${CMAKE_CURRENT_LIST_DIR}/firmware/model_settings.cpp
)
target_link_libraries(mobilenet_gtsrb_batch pico-tflmicro pico_stdlib)
pico_enable_stdio_usb(mobilenet_gtsrb_batch 1)
pico_enable_stdio_uart(mobilenet_gtsrb_batch 1)
pico_add_extra_outputs(mobilenet_gtsrb_batch)
