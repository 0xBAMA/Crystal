cmake_minimum_required( VERSION 3.16 )

project( NQADE VERSION 1.2 LANGUAGES C CXX )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED True )
# set( CMAKE_CXX_FLAGS "-std=c++20 -O4 -mavx" )
set( CMAKE_CXX_FLAGS "-std=c++20 -mavx" )

set( CMAKE_REQUIRED_FLAGS -lstdc++fs )

# build type set to debug
set( CMAKE_BUILD_TYPE Debug )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

find_package(ftxui REQUIRED)
add_executable(main main.cpp)
add_library( STB_ImageUtilsWrapper STATIC stb/impl.cc )

target_link_libraries(main
  PUBLIC STB_ImageUtilsWrapper
  PRIVATE ftxui::screen
  PRIVATE ftxui::dom
  PRIVATE ftxui::component
)

