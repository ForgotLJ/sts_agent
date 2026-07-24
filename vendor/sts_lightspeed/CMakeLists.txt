cmake_minimum_required(VERSION 3.19)
project(
        sts_lightspeed
        VERSION 0.1
        DESCRIPTION "High Performance, rng accurate implementation of Slay the Spire with tools for tree search and ml."
        LANGUAGES CXX)

file (GLOB_RECURSE sts_lightspeed_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
    add_compile_options(/O2 /permissive-)
else()
    add_compile_options(-Wno-shift-count-overflow -O3)
endif()

add_subdirectory(json)
add_subdirectory(pybind11)

add_library(sts_lightspeed_core STATIC ${sts_lightspeed_SOURCES})
set_target_properties(sts_lightspeed_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(sts_lightspeed_core PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(sts_lightspeed_core PUBLIC include json/include)

pybind11_add_module(slaythespire NO_EXTRAS bindings/slaythespire.cpp bindings/bindings-util.cpp bindings/simulator-bridge.cpp)
target_link_libraries(slaythespire PRIVATE sts_lightspeed_core)
target_include_directories(slaythespire PRIVATE bindings)

option(STS_LIGHTSPEED_BUILD_TEST_DRIVER "Build the upstream command-line test driver" ON)
if (STS_LIGHTSPEED_BUILD_TEST_DRIVER)
    add_executable(test apps/test.cpp)
    target_link_libraries(test PRIVATE sts_lightspeed_core)
endif()
