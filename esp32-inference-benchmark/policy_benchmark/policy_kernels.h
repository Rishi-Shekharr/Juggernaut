#pragma once

#include <Arduino.h>
#include <stddef.h>
#include <stdint.h>

// Firmware-only dense kernel variants. The C path intentionally preserves the
// original single-accumulator operation order for the benchmark baseline.
enum PolicyKernel : uint8_t { POLICY_KERNEL_C = 0, POLICY_KERNEL_DSP = 1,
                              POLICY_KERNEL_UNROLL4 = 2 };

// Keep return types visible before Arduino inserts sketch function prototypes.
struct ProfileStats {
  double mean;
  uint32_t p99;
  uint32_t maximum;
  uint32_t expCalls;
};

#ifndef POLICY_DENSE_IRAM
#define POLICY_DENSE_IRAM 0
#endif

#if POLICY_DENSE_IRAM
#define POLICY_DENSE_ATTR IRAM_ATTR __attribute__((noinline))
#else
#define POLICY_DENSE_ATTR __attribute__((noinline))
#endif

#ifdef __FAST_MATH__
#error "FP32 comparison builds must not enable fast-math"
#endif

