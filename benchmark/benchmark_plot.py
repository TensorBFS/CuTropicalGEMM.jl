import numpy as np
import matplotlib.pyplot as plt

max = 19.5

CublasGemm = np.array([18578.596294, 18877.954689, 18943.901496]) / 1000
GemmKernelsTropical = np.array([10694.937413845337, 10381.497926923561, 10126.264507163212]) / 1000
CuTropicalGEMM_FP32 = np.array([14.035, 13.486, 13.05])
MapReduce = np.array([1.323, 0.657, 0.549])

total_width, n = 0.8, 4
width = total_width / n
size = 3
x = np.arange(size)
x = x - (total_width - width) / 2

plt.figure(dpi = 150)
plt.bar(x, CublasGemm, width=width, label='CublasGemm', color = 'orange')
plt.bar(x + width, CuTropicalGEMM_FP32, width=width, label='CuTropicalGEMM', color = 'g')
plt.bar(x + 2 * width, GemmKernelsTropical, width=width, label='GemmKernelsTropical', color = 'b')
plt.bar(x + 3 * width, MapReduce, width=width, label='CUDA map reduce', color = 'r')

for a,b in zip(x, CublasGemm):  
    plt.text(a, b+0.5, '%.2f' % b, ha='center', va= 'bottom',fontsize=8)  

for a,b in zip(x + width, CuTropicalGEMM_FP32):  
    plt.text(a, b+0.5, '%.2f' % b, ha='center', va= 'bottom',fontsize=8) 

for a,b in zip(x + 2 * width, GemmKernelsTropical):  
    plt.text(a, b+0.5, '%.2f' % b, ha='center', va= 'bottom',fontsize=8) 

for a,b in zip(x + 3 * width, MapReduce):  
    plt.text(a, b+0.5, '%.2f' % b, ha='center', va= 'bottom',fontsize=8) 

plt.xticks([0.0, 1.0, 2.0], [r"$2560 \times 2048 \times 2048$", r"$5120 \times 4096 \times 4096$", r"$10240 \times 8192 \times 8192$"])
plt.legend(ncols = 2)
plt.xlabel("Matrix size")
plt.ylabel("Performance (TFLOPS)")
plt.ylim([0.0, 25.0])

plt.title("A800 with FP32 performance of about 19.5 TFlops")

plt.savefig("benchmark.png")