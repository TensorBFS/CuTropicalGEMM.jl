CUDA_FILES := max_add.cu and_or.cu max_mul.cu min_add.cu gemm.cu
TARGET := TropicalGemmC.so
DEPENDENCIES := $(CUDA_FILES)
NVCC := nvcc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(DEPENDENCIES)
	$(NVCC)	-Xcompiler	-fPIC	-shared	-o	$@	$^

clean:
	rm	-f	$(TARGET)