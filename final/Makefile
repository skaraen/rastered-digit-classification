GCC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -flto -funroll-loops -ffast-math -lm
GFLAGS = -use_fast_math -gencode
A100 = arch=compute_80,code=sm_80
V100 = arch=compute_70,code=sm_70
RTX = arch=compute_75,code=sm_75
OPENMP = -fopenmp
BLAS = -lopenblas
CUBLAS = -lcublas

cpu_native:
	$(CC) $(CFLAGS) $(OPENMP) model_cpu.c -o model_cpu

cpu_blas:
	$(CC) $(CFLAGS) $(OPENMP) $(BLAS) model_blas.c -o model_blas

gpu_native:
	$(NVCC) $(GFLAGS) $(A100) model_gpu.cu -o a100
	$(NVCC) $(GFLAGS) $(V100) model_gpu.cu -o v100
	$(NVCC) $(GFLAGS) $(RTX) model_gpu.cu -o rtx

gpu_cublas:
	$(NVCC) $(GFLAGS) $(A100) model_cublas.cu -o a100_cublas $(CUBLAS)
	$(NVCC) $(GFLAGS) $(V100) model_cublas -o v100_cublas $(CUBLAS)
	$(NVCC) $(GFLAGS) $(RTX) model_cublas.cu -o rtx_cublas $(CUBLAS)

clean:
	rm -f model_cpu model_blas a100 v100 rtx a100_cublas v100_cublas rtx_cublas