nvcc src/cpu.cu ../src/utils.cpp -o builds/cpu -I ../includes
nvcc src/gpu.cu ../src/utils.cpp -o builds/gpu -I ../includes