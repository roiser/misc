CUDA_ARCH = 80
ARCH = -arch sm_$(CUDA_ARCH)

INCDIR = ../include
HEADERS = $(INCDIR)/dev_array.h data.h data_3g.h hst_matrix.h kernel.h macros.h

NVCC = nvcc
CFLAGS = $(ARCH) -I $(INCDIR) -lcublas -lnvToolsExt

all: main

debug: CFLAGS += -g -G -DDEBUG
debug: all

profile: CFLAGS += -pg
profile: debug

%: %.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f main

# nvcc -arch sm_80 -I /p/home/jusers/roiser1/juwels/usr/local/include ampere_tf32_tensorop_gemm.cu
