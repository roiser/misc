CUDA_ARCH = 80
EXEC = main
NVCC = nvcc

SRC = main.cu kernel.cu
OBJ = $(SRC:.cu=.o)
HEADERS = kernel.h dev_array.h

ARCH = -arch sm_$(CUDA_ARCH)
CFLAGS = $(ARCH)

# nvcc -arch sm_80 -I /p/home/jusers/roiser1/juwels/usr/local/include ampere_tf32_tensorop_gemm.cu

all: $(EXEC)

debug: CFLAGS += -g -G
debug: all

profile: CFLAGS += -pg
profile: debug

$(EXEC): $(OBJ)
	$(NVCC) -o $@ $<

%.o : %.cu $(HEADERS)
	$(NVCC) -c $(CFLAGS) -o $@ $<

clean:
	rm -f $(EXEC) $(OBJ)
