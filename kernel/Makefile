CC          = clang
CFLAGS      = -O3 -fPIC -Wall -Wextra -DACCELERATE_NEW_LAPACK \
              -march=armv8.6-a+simd+i8mm -I.
LDFLAGS     = -shared -framework Accelerate

SRCS = kv_cache.c ops.c attention.c scratch.c decoder.c w8a32_kernel.c

libdecoder_release.dylib: $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)
	@echo "Built $@"

# Bench FLOPS library — links w8a32_kernel.c for w8a32_linear/quantize
BENCH_SRCS = bench_flops.c w8a32_kernel.c
BENCH_CFLAGS = $(CFLAGS) -march=armv8.6-a+simd+i8mm+fp16

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  BENCH_LIB = libbench_flops.dylib
else
  BENCH_LIB = libbench_flops.so
  BENCH_CFLAGS = $(CFLAGS)
endif

$(BENCH_LIB): $(BENCH_SRCS)
	$(CC) $(BENCH_CFLAGS) $(LDFLAGS) -o $@ $(BENCH_SRCS)
	@echo "Built $@"

.PHONY: all clean bench-flops

all: libdecoder_release.dylib

bench-flops: $(BENCH_LIB)

clean:
	rm -f libdecoder_release.dylib $(BENCH_LIB)
