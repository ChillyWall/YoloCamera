cmake_minimum_required(VERSION 3.16)
project(yolo_camera LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

add_executable(yolo_camera src/main.cpp src/yolo_camera.cpp)
target_include_directories(yolo_camera PRIVATE ${OpenCV_INCLUDE_DIRS} include ${TensorRT-YOLO_INCLUDE_DIRS})
target_link_libraries(yolo_camera PRIVATE ${OpenCV_LIBS} trtyolo custom_plugins)
target_compile_definitions(yolo_camera PRIVATE
    CONFIG_FILE="${CMAKE_CURRENT_SOURCE_DIR}/res/config.yml")

install(TARGETS yolo_camera)
