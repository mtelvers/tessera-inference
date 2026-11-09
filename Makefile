# tessera_encode      CPU build: oneDNN built with ONEDNN_CPU_RUNTIME=SEQ
# tessera_encode_cuda GPU build: CUDA toolkit with cuBLASLt
#
#   make ONEDNN=/path/to/onednn-seq                 # CPU
#   make cuda CUDA=/usr/local/cuda CUDA_ARCH=sm_89  # GPU (sm_89 = L4, sm_80 = A30/A100)

CXX      ?= g++
NVCC     ?= $(CUDA)/bin/nvcc
ONEDNN   ?= /usr/local
CUDA     ?= /usr/local/cuda
CUDA_ARCH ?= sm_89
MARCH    ?= native
CXXFLAGS ?= -O3 -march=$(MARCH)
NVFLAGS  ?= -O3
ALL_CXXFLAGS = $(CXXFLAGS) -std=c++17 -fopenmp
ALL_NVFLAGS  = $(NVFLAGS) -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(CXX)

all: tessera_encode

tessera_encode: src/tessera_encode.cpp src/tessera_cpu.cpp
	$(CXX) $(ALL_CXXFLAGS) -I$(ONEDNN)/include -o $@ $^ \
	  $(LDFLAGS) -L$(ONEDNN)/lib -L$(ONEDNN)/lib64 -ldnnl -Wl,-rpath,$(ONEDNN)/lib -Wl,-rpath,$(ONEDNN)/lib64

cuda: tessera_encode_cuda

tessera_cuda.o: src/tessera_cuda.cu
	$(NVCC) $(ALL_NVFLAGS) -c -o $@ $<

tessera_encode_cuda: src/tessera_encode.cpp tessera_cuda.o
	$(CXX) $(ALL_CXXFLAGS) -o $@ $^ \
	  $(LDFLAGS) -L$(CUDA)/lib64 -L$(CUDA)/lib -lcudart -lcublasLt \
	  -Wl,-rpath,$(CUDA)/lib64 -Wl,-rpath,$(CUDA)/lib

clean:
	rm -f tessera_encode tessera_encode_cuda tessera_cuda.o

.PHONY: all cuda clean
