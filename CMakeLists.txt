cmake_minimum_required (VERSION 3.18)

project (Helios LANGUAGES CUDA CXX C)

set(TARGET ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#CUDA
find_package(CUDAToolkit REQUIRED)

#OpenGL
find_package(OpenGL REQUIRED)

#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(ext/glfw)
add_subdirectory(ext/glm)

set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_BUILD_SAMPLES OFF)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
add_subdirectory(ext/assimp)

set(
        SOURCES

        #Helios
        src/main.cpp
        src/Ray_tracer.cu
        src/Application.cpp
        src/Shader.cpp
        src/Quad.cpp
        src/Frame.cpp
        src/Camera.cpp
        src/Object_Loader.cpp

        #glad
        ext/glad/src/glad.c

        #imgui
        ext/imgui/imgui.cpp
        ext/imgui/imgui_impl_glfw.cpp
        ext/imgui/imgui_demo.cpp
        ext/imgui/imgui_draw.cpp
        ext/imgui/imgui_tables.cpp
        ext/imgui/imgui_widgets.cpp
        ext/imgui/imgui_impl_opengl3.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS}

        src/

        ext/glad/include
        ext/glfw/include
        ext/imgui/
        ext/glm/
        ext/assimp/include
        )

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --keep # Keep PTX code
        >)

target_link_libraries(${TARGET} glfw OpenGL::GL glm assimp)