cmake_minimum_required(VERSION 3.12)

project(mobilenet_gtsrb C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(mobilenet_gtsrb "")

target_include_directories(mobilenet_gtsrb
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  mobilenet_gtsrb
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(mobilenet_gtsrb
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/main_golden.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
)

target_link_libraries(
  mobilenet_gtsrb
  pico-tflmicro
  pico_stdlib
)

pico_enable_stdio_usb(mobilenet_gtsrb 1)
pico_enable_stdio_uart(mobilenet_gtsrb 1)

pico_add_extra_outputs(mobilenet_gtsrb)
