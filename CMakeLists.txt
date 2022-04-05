cmake_minimum_required (VERSION 3.15)

project (pekka)

set(TARGET ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#OpenGL
find_package(OpenGL REQUIRED)

#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(ext/glfw)

add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        ext/glad/src/glad.c
)

target_include_directories(
        ${PROJECT_NAME} PRIVATE
        ext/glad/include
        ext/glfw/include
)

target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL)