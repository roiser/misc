CUDA_ARCH = 80
EXEC = main
NVCC = nvcc

SRC = main.cu
HEADERS = dev_array.h hst_matrix.h kernel.h

ARCH = -arch sm_$(CUDA_ARCH)
CFLAGS = $(ARCH)

# nvcc -arch sm_80 -I /p/home/jusers/roiser1/juwels/usr/local/include ampere_tf32_tensorop_gemm.cu

all: $(EXEC)

debug: CFLAGS += -g -G
debug: all

profile: CFLAGS += -pg
profile: debug

$(EXEC): $(SRC) $(HEADERS)
	$(NVCC) $(ARCH) -I . -o $@ $<

clean:
	rm -f $(EXEC)
