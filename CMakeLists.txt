cmake_minimum_required(VERSION 3.4)
project(svddash)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(NumPy REQUIRED)
find_package(OpenMP)

add_subdirectory(thirdparty/pybind11)


pybind11_add_module(svddash src/main.cpp)
set_target_properties(svddash
    PROPERTIES
    CXX_STANDARD 14 # need C++14 for the xtensor stuff
    CXX_STANDARD_REQUIRED ON)
target_include_directories(svddash PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xtl/include" "thirdparty/eigen" ${NUMPY_INCLUDE_DIRS})
target_link_libraries(svddash PUBLIC ${PYTHON_LIBRARY})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(svddash PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
