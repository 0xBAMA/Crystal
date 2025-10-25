mkdir build
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
cd build
time make -j72 main