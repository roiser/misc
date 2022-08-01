CUTLASS_HOME=/p/home/jusers/roiser1/juwels/usr/local
CUDA_ARCH=80
EXEC=ampere_tf32_tensorop_gemm
NVCC=nvcc

# nvcc -arch sm_80 -I /p/home/jusers/roiser1/juwels/usr/local/include ampere_tf32_tensorop_gemm.cu

all: $(EXEC)

$(EXEC): $(EXEC).cu

% : %.cu
	$(NVCC) -arch sm_$(CUDA_ARCH) -I $(CUTLASS_HOME)/include -o $@ $<

clean:
	rm -f $(EXEC)
