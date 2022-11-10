CUDA_ARCH = 80
ARCH = -arch sm_$(CUDA_ARCH)

INCDIR = ../include
HEADERS = $(INCDIR)/dev_array.h data.h data_3g.h gpu_common.h helpers.h \
          hst_matrix.h kernel.h matmult.h timer.h

NVCC = nvcc
CFLAGS = $(ARCH) -I $(INCDIR) -lcublas -lnvToolsExt

all: main_cx
cx: main_cx
sc: main


debug: CFLAGS += -g -G -DDEBUG
debug: all

profile: CFLAGS += -pg
profile: debug

%: %.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f main main_cx main_full

# nvcc -arch sm_80 -I /p/home/jusers/roiser1/juwels/usr/local/include ampere_tf32_tensorop_gemm.cu
