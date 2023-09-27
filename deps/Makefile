CUDA_FILES :=  tropicalgemm_kernels.cu

MACRO_COMBINATIONS :=  PlusMul_FP32 PlusMul_FP64 PlusMul_INT32 PlusMul_INT64  TropicalAndOr_Bool   TropicalMaxMul_FP32   TropicalMaxMul_FP64   TropicalMaxMul_INT32  TropicalMaxMul_INT64 TropicalMaxPlus_FP32 TropicalMaxPlus_FP64

LIBRARY_NAME	:=kernels.so

$(LIBRARY_NAME):	$(foreach	MACROS,	$(MACRO_COMBINATIONS),	$(CUDA_FILES))
	mkdir ./lib
	@for	MACROS	in	$(MACRO_COMBINATIONS);	do	\
	MACRO_1=$$(echo	$$MACROS	|	cut	-d'_'	-f1);	\
	MACRO_2=$$(echo	$$MACROS	|	cut	-d'_'	-f2);	\
	nvcc	-Xcompiler	-fPIC	-shared	-D$$MACRO_1	-D$$MACRO_2	$(CUDA_FILES)	-o	./lib/lib_$$MACROS.so;	\
    done

clean:
	rm	-rf	./lib
