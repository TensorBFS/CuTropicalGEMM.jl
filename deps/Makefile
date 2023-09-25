# CUDA_FILES := cuda/max_add.cu cuda/and_or.cu cuda/max_mul.cu cuda/mul_add.cu
CUDA_FILES := cuda/andor/Bool_andor.cu cuda/maxmul/FP32_maxmul.cu cuda/maxmul/FP64_maxmul.cu cuda/maxmul/INT32_maxmul.cu cuda/maxmul/INT64_maxmul.cu cuda/maxplus/FP32_maxplus.cu cuda/maxplus/FP64_maxplus.cu cuda/minplus/FP32_minplus.cu cuda/minplus/FP64_minplus.cu cuda/muladd/FP32_muladd.cu cuda/muladd/FP64_muladd.cu cuda/muladd/INT32_muladd.cu cuda/muladd/INT64_muladd.cu

TARGET := TropicalGemmC.so
DEPENDENCIES := $(CUDA_FILES)
NVCC := nvcc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(DEPENDENCIES)
	$(NVCC)	-Xcompiler	-fPIC	-shared	-o	$@	$^

clean:
	rm	-f	$(TARGET)