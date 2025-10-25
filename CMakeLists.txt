cmake_minimum_required( VERSION 3.16 )

project( NQADE VERSION 1.2 LANGUAGES C CXX )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED True )
set( CMAKE_CXX_FLAGS "-std=c++20 -O4 -mavx" )
set( CMAKE_REQUIRED_FLAGS -lstdc++fs )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

find_package(ftxui REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main
  PRIVATE ftxui::screen
  PRIVATE ftxui::dom
  PRIVATE ftxui::component
)

