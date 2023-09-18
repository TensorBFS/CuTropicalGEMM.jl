CUDA_FILES := cuda/max_add.cu cuda/and_or.cu cuda/max_mul.cu cuda/mul_add.cu
TARGET := TropicalGemmC.so
DEPENDENCIES := $(CUDA_FILES)
NVCC := nvcc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(DEPENDENCIES)
	$(NVCC)	-Xcompiler	-fPIC	-shared	-o	$@	$^

clean:
	rm	-f	$(TARGET)