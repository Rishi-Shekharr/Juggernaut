#include <Arduino.h>
#include <esp_chip_info.h>
#include <esp_timer.h>
#include <esp_rom_sys.h>
#include <esp_heap_caps.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "policy_data.h"
#include "policy_int8_weight.h"
#include "policy_kernels.h"
#include "policy_quantized_data.h"
#include <dsps_dotprod.h>

static_assert(sizeof(fc1_w_int8) == 32768 * sizeof(int8_t),
              "layer-1 INT8 matrix must contain 128x256 values");
static_assert(sizeof(fc1_w_scale) == 128 * sizeof(float),
              "layer-1 INT8 scales must contain one float per output channel");
static_assert(kL1QuantInputs == 256 && kL1QuantOutputs == 128,
              "quantized layer dimensions must be 256x128");
static_assert(sizeof(fc1_w) == 128 * 256 * sizeof(float),
              "FP32 layer-1 matrix must contain 128x256 values");

#ifndef ESP_ARDUINO_VERSION_STR
#define ESP_ARDUINO_VERSION_STR "unknown"
#endif

#if defined(CONFIG_BT_ENABLED) && !defined(CONFIG_BT_SPP_DISABLED)
#include <BluetoothSerial.h>
#define POLICY_BENCH_HAS_BT 1
static BluetoothSerial SerialBT;
#else
#define POLICY_BENCH_HAS_BT 0
#endif

#ifndef POLICY_OPT_LEVEL
#define POLICY_OPT_LEVEL 0
#endif
#define POLICY_STRINGIFY_INNER(value) #value
#define POLICY_STRINGIFY(value) POLICY_STRINGIFY_INNER(value)
#ifndef POLICY_BUILD_ID
#define POLICY_BUILD_ID unspecified
#endif
#if POLICY_OPT_LEVEL == 1
#define POLICY_OPT_NAME "Os"
#elif POLICY_OPT_LEVEL == 2
#define POLICY_OPT_NAME "O2"
#else
#define POLICY_OPT_NAME "unknown"
#endif

static constexpr uint32_t kPeriodUs = 20000;
static constexpr uint32_t kDefaultBenchIterations = 2000;
static constexpr uint32_t kMaxBenchIterations = 10000;
static constexpr uint32_t kDefaultRunDurationMs = 60000;
static constexpr uint32_t kMaxRunDurationMs = 120000;
static constexpr size_t kMaxSamples = kMaxBenchIterations;
static constexpr int kLayerCount = 4;
static const int kInCounts[kLayerCount] = {kInputCount, 256, 128, 64};
static const int kOutCounts[kLayerCount] = {256, 128, 64, kOutputCount};
static const bool kActivations[kLayerCount] = {true, true, true, false};

static float g_a0[256];
static float g_a1[128];
static float g_a2[64];
static uint32_t g_samples[kMaxSamples];
static volatile float g_checksumSink = 0.0f;
static PolicyKernel g_kernel = POLICY_KERNEL_C;
alignas(16) static float g_dspInput[257];
alignas(16) static int8_t g_l1QuantInput[256];
static bool g_quantL1Int8 = false;
static uint32_t g_quantFallbackCount = 0;

static const float *const g_flashWeights[kLayerCount] = {fc0_w, fc1_w, fc2_w, fc3_w};
static const float *const g_flashBiases[kLayerCount] = {fc0_b, fc1_b, fc2_b, fc3_b};
static const float *g_liveWeights[kLayerCount] = {fc0_w, fc1_w, fc2_w, fc3_w};
static const float *g_liveBiases[kLayerCount] = {fc0_b, fc1_b, fc2_b, fc3_b};
static float *g_weightCopies[kLayerCount] = {};
static int g_weightsRamLayer = -1;
static size_t g_weightsRamBytes = 0;

static inline float elu(float x) { return x >= 0.0f ? x : expf(x) - 1.0f; }

static POLICY_DENSE_ATTR void denseC(const float *input, float *output,
                                     const float *weights, const float *bias,
                                     int outCount, int inCount, bool activate) {
  for (int o = 0; o < outCount; ++o) {
    const float *row = weights + (size_t)o * inCount;
    float value = bias[o];
    // Preserve the baseline's scalar accumulation and exact operation order.
    for (int i = 0; i < inCount; ++i) value += row[i] * input[i];
    output[o] = activate ? elu(value) : value;
  }
}

static POLICY_DENSE_ATTR void denseUnroll4(const float *input, float *output,
                                           const float *weights, const float *bias,
                                           int outCount, int inCount, bool activate) {
  for (int o = 0; o < outCount; ++o) {
    const float *row = weights + (size_t)o * inCount;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < inCount; i += 4) {
      s0 += row[i] * input[i];
      s1 += row[i + 1] * input[i + 1];
      s2 += row[i + 2] * input[i + 2];
      s3 += row[i + 3] * input[i + 3];
    }
    float value = bias[o] + ((s0 + s1) + (s2 + s3));
    for (; i < inCount; ++i) value += row[i] * input[i];
    output[o] = activate ? elu(value) : value;
  }
}

static __attribute__((noinline)) void denseDSP(const float *input, float *output, const float *weights,
                     const float *bias, int outCount, int inCount, bool activate) {
  // ESP-DSP ae32 may read src2[len], so prepare and pad the input once per layer.
  memcpy(g_dspInput, input, (size_t)inCount * sizeof(float));
  g_dspInput[inCount] = 0.0f;
  for (int o = 0; o < outCount; ++o) {
    float dot = 0.0f;
    (void)dsps_dotprod_f32_ae32(weights + (size_t)o * inCount,
                                g_dspInput, &dot, inCount);
    const float value = bias[o] + dot;
    output[o] = activate ? elu(value) : value;
  }
}

static POLICY_DENSE_ATTR void dense(const float *input, float *output,
                                    const float *weights, const float *bias,
                                    int outCount, int inCount, bool activate) {
  if (g_kernel == POLICY_KERNEL_DSP)
    denseDSP(input, output, weights, bias, outCount, inCount, activate);
  else if (g_kernel == POLICY_KERNEL_UNROLL4)
    denseUnroll4(input, output, weights, bias, outCount, inCount, activate);
  else
    denseC(input, output, weights, bias, outCount, inCount, activate);
}

static POLICY_DENSE_ATTR void denseL1Int8(const float *input, float *output) {
  float maxAbs = 0.0f;
  for (int i = 0; i < 256; ++i) {
    const float value = input[i];
    if (!isfinite(value)) {
      ++g_quantFallbackCount;
      dense(input, output, g_liveWeights[1], g_liveBiases[1], 128, 256, true);
      return;
    }
    const float absValue = fabsf(value);
    if (absValue > maxAbs) maxAbs = absValue;
  }

  const float inputScale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
  if (!isfinite(inputScale) || inputScale <= 0.0f) {
    ++g_quantFallbackCount;
    dense(input, output, g_liveWeights[1], g_liveBiases[1], 128, 256, true);
    return;
  }
  if (maxAbs == 0.0f) {
    memset(g_l1QuantInput, 0, sizeof(g_l1QuantInput));
  } else {
    for (int i = 0; i < 256; ++i) {
      const float rounded = roundf(input[i] / inputScale);  // ties away from zero
      int32_t quantized = (int32_t)rounded;
      if (quantized > 127) quantized = 127;
      if (quantized < -127) quantized = -127;
      g_l1QuantInput[i] = (int8_t)quantized;
    }
  }

  for (int o = 0; o < 128; ++o) {
    const int8_t *row = fc1_w_int8 + (size_t)o * 256;
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < 256; i += 4) {
      s0 += (int32_t)row[i] * (int32_t)g_l1QuantInput[i];
      s1 += (int32_t)row[i + 1] * (int32_t)g_l1QuantInput[i + 1];
      s2 += (int32_t)row[i + 2] * (int32_t)g_l1QuantInput[i + 2];
      s3 += (int32_t)row[i + 3] * (int32_t)g_l1QuantInput[i + 3];
    }
    for (; i < 256; ++i)
      s0 += (int32_t)row[i] * (int32_t)g_l1QuantInput[i];
    const int32_t sum = (s0 + s1) + (s2 + s3);
    const float value = g_liveBiases[1][o] +
                        ((float)sum * inputScale) * fc1_w_scale[o];
    if (!isfinite(value)) {
      ++g_quantFallbackCount;
      dense(input, output, g_liveWeights[1], g_liveBiases[1], 128, 256, true);
      return;
    }
    output[o] = elu(value);
  }
}

// PROFILE-only pass, deliberately outside the timed region and normal inference.
static uint32_t countNegativeEluInputs(const float *input, const float *weights,
                                       const float *bias, int outCount, int inCount) {
  uint32_t count = 0;
  if (g_kernel == POLICY_KERNEL_DSP) {
    memcpy(g_dspInput, input, (size_t)inCount * sizeof(float));
    g_dspInput[inCount] = 0.0f;
    for (int o = 0; o < outCount; ++o) {
      float dot = 0.0f;
      (void)dsps_dotprod_f32_ae32(weights + (size_t)o * inCount,
                                  g_dspInput, &dot, inCount);
      if (bias[o] + dot < 0.0f) ++count;
    }
  } else if (g_kernel == POLICY_KERNEL_UNROLL4) {
    for (int o = 0; o < outCount; ++o) {
      const float *row = weights + (size_t)o * inCount;
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      int i = 0;
      for (; i + 3 < inCount; i += 4) {
        s0 += row[i] * input[i]; s1 += row[i + 1] * input[i + 1];
        s2 += row[i + 2] * input[i + 2]; s3 += row[i + 3] * input[i + 3];
      }
      float value = bias[o] + ((s0 + s1) + (s2 + s3));
      for (; i < inCount; ++i) value += row[i] * input[i];
      if (value < 0.0f) ++count;
    }
  } else {
    for (int o = 0; o < outCount; ++o) {
      const float *row = weights + (size_t)o * inCount;
      float value = bias[o];
      for (int i = 0; i < inCount; ++i) value += row[i] * input[i];
      if (value < 0.0f) ++count;
    }
  }
  return count;
}

// Signed INT8 weights only: FP32 inputs and accumulation, one scale per row.
static POLICY_DENSE_ATTR void denseL0WeightInt8(const float *input, float *output) {
  for (int o = 0; o < 256; ++o) {
    const int8_t *row = fc0_w8 + (size_t)o * 230;
    float sum = 0.0f;
    for (int i = 0; i < 230; ++i) sum += (float)row[i] * input[i];
    output[o] = elu(g_liveBiases[0][o] + fc0_w8_scales[o] * sum);
  }
}

static void runLayer(int layer, const float *input, float *output) {
  if (layer == 0) { denseL0WeightInt8(input, output); return; }
  if (layer == 1 && g_quantL1Int8) {
    denseL1Int8(input, output);
    return;
  }
  dense(input, output, g_liveWeights[layer], g_liveBiases[layer],
        kOutCounts[layer], kInCounts[layer], kActivations[layer]);
}

static void infer(const float *input, float *output) {
  runLayer(0, input, g_a0);
  runLayer(1, g_a0, g_a1);
  runLayer(2, g_a1, g_a2);
  runLayer(3, g_a2, output);
}

static void sortSamples(size_t count) {
  qsort(g_samples, count, sizeof(g_samples[0]),
        [](const void *a, const void *b) -> int {
          const uint32_t x = *static_cast<const uint32_t *>(a);
          const uint32_t y = *static_cast<const uint32_t *>(b);
          return x < y ? -1 : (x > y ? 1 : 0);
        });
}

static uint32_t percentile(size_t count, unsigned pct) {
  if (!count) return 0;
  size_t rank = (count * pct + 99) / 100;
  if (rank == 0) rank = 1;
  return g_samples[rank - 1];
}

static const char *kernelName() {
  return g_kernel == POLICY_KERNEL_DSP ? "DSP" :
         g_kernel == POLICY_KERNEL_UNROLL4 ? "UNROLL4" : "C";
}

static const char *quantModeName() { return g_quantL1Int8 ? "L0_WEIGHT_INT8_PLUS_L1_INT8" : "L0_WEIGHT_INT8"; }

static void printInfo() {
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  const size_t weightBytes = sizeof(fc0_w) + sizeof(fc0_b) + sizeof(fc1_w) +
                             sizeof(fc1_b) + sizeof(fc2_w) + sizeof(fc2_b) +
                             sizeof(fc3_w) + sizeof(fc3_b);
  const uint32_t flashMode = (uint32_t)ESP.getFlashChipMode();
  Serial.printf("{\"type\":\"info\",\"schema_version\":2,\"build_id\":\"%s\",\"arduino_core\":\"%s\",\"chip\":\"%s\",\"cores\":%u,\"cpu_mhz\":%u,\"heap\":%u,\"min_heap\":%u,\"weights_bytes\":%u,\"policy_sha256\":\"%s\",\"policy_checkpoint\":\"%s\",\"kernel\":\"%s\",\"optimization\":\"%s\",\"dense_iram\":%s,\"runtime_flash_mode\":%u,\"flash_header_speed_hz\":%u,\"weights_layer\":%d,\"weights_ram_bytes\":%u,\"quant_mode\":\"%s\",\"quant_layer\":0,\"quant_sha256\":\"%s\",\"quant_scheme\":\"%s\",\"quant_storage_bytes\":%u,\"fallback_count\":%u,\"free_internal_heap\":%u,\"largest_internal_block\":%u}\n",
                POLICY_STRINGIFY(POLICY_BUILD_ID), ESP_ARDUINO_VERSION_STR,
                ESP.getChipModel(), (unsigned)chip.cores,
                (unsigned)getCpuFrequencyMhz(), (unsigned)ESP.getFreeHeap(),
                (unsigned)ESP.getMinFreeHeap(), (unsigned)weightBytes,
                kPolicySha256, kPolicyCheckpoint, kernelName(), POLICY_OPT_NAME,
                POLICY_DENSE_IRAM ? "true" : "false", (unsigned)flashMode,
                (unsigned)ESP.getFlashChipSpeed(), g_weightsRamLayer,
                (unsigned)g_weightsRamBytes,
                quantModeName(), kInt8WeightSha256, "symmetric_per_row_int8_weights_fp32_activations",
                (unsigned)(sizeof(fc0_w8) + sizeof(fc0_w8_scales)),
                (unsigned)g_quantFallbackCount,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
}

static void commandCheck() {
  float output[kOutputCount];
  for (int n = 0; n < kVectorCount; ++n) {
    infer(test_inputs[n], output);
    Serial.printf("{\"type\":\"prediction\",\"index\":%d,\"output\":[", n);
    for (int j = 0; j < kOutputCount; ++j) {
      if (j) Serial.print(',');
      Serial.printf("%.9g", (double)output[j]);
    }
    Serial.println("]}");
  }
  Serial.println("{\"type\":\"check_done\"}");
}

static void commandBench(uint32_t iterations) {
  if (iterations == 0) iterations = kDefaultBenchIterations;
  if (iterations > kMaxBenchIterations) iterations = kMaxBenchIterations;
  float output[kOutputCount];
  double sum = 0.0;
  uint32_t minimum = UINT32_MAX, maximum = 0;
  for (uint32_t i = 0; i < iterations; ++i) {
    const int64_t before = esp_timer_get_time();
    infer(test_inputs[i % kVectorCount], output);
    const int64_t elapsed = esp_timer_get_time() - before;
    const uint32_t us = elapsed > 0 ? (uint32_t)elapsed : 0;
    g_samples[i] = us;
    sum += us;
    if (us < minimum) minimum = us;
    if (us > maximum) maximum = us;
    g_checksumSink += output[i % kOutputCount];
    vTaskDelay(1);
  }
  sortSamples(iterations);
  Serial.printf("{\"type\":\"bench\",\"kernel\":\"%s\",\"quant_mode\":\"%s\",\"fallback_count\":%u,\"iterations\":%u,\"min_us\":%u,\"mean_us\":%.3f,\"p50_us\":%u,\"p95_us\":%u,\"p99_us\":%u,\"max_us\":%u,\"heap\":%u,\"checksum\":%.9g}\n",
                kernelName(), quantModeName(), (unsigned)g_quantFallbackCount,
                (unsigned)iterations, (unsigned)minimum,
                sum / iterations, (unsigned)percentile(iterations, 50),
                (unsigned)percentile(iterations, 95),
                (unsigned)percentile(iterations, 99), (unsigned)maximum,
                (unsigned)ESP.getFreeHeap(), (double)g_checksumSink);
}

static ProfileStats profilePass(int layer, uint32_t iterations) {
  float output[kOutputCount];
  double sum = 0.0;
  uint32_t maximum = 0, expCalls = 0;
  for (uint32_t i = 0; i < iterations; ++i) {
    const float *input = test_inputs[i % kVectorCount];
    if (layer > 0) runLayer(0, input, g_a0);
    if (layer > 1) runLayer(1, g_a0, g_a1);
    if (layer > 2) runLayer(2, g_a1, g_a2);
    const float *layerInput = layer == 0 ? input : layer == 1 ? g_a0 : layer == 2 ? g_a1 : g_a2;
    float *layerOutput = layer == 0 ? g_a0 : layer == 1 ? g_a1 : layer == 2 ? g_a2 : output;
    const int64_t before = esp_timer_get_time();
    runLayer(layer, layerInput, layerOutput);
    const uint32_t elapsed = (uint32_t)(esp_timer_get_time() - before);
    g_samples[i] = elapsed;
    sum += elapsed;
    if (elapsed > maximum) maximum = elapsed;
    if (kActivations[layer])
      expCalls += countNegativeEluInputs(layerInput, g_liveWeights[layer],
                                         g_liveBiases[layer], kOutCounts[layer],
                                         kInCounts[layer]);
    g_checksumSink += layerOutput[i % kOutCounts[layer]];
    vTaskDelay(1);
  }
  sortSamples(iterations);
  ProfileStats stats = {sum / iterations, percentile(iterations, 99), maximum, expCalls};
  return stats;
}

static ProfileStats profileAggregate(uint32_t iterations) {
  float output[kOutputCount];
  double sum = 0.0;
  uint32_t maximum = 0, expCalls = 0;
  for (uint32_t i = 0; i < iterations; ++i) {
    const int64_t before = esp_timer_get_time();
    infer(test_inputs[i % kVectorCount], output);
    const uint32_t elapsed = (uint32_t)(esp_timer_get_time() - before);
    g_samples[i] = elapsed;
    sum += elapsed;
    if (elapsed > maximum) maximum = elapsed;
    const float *profileInputs[3] = {test_inputs[i % kVectorCount], g_a0, g_a1};
    for (int l = 0; l < 3; ++l)
      expCalls += countNegativeEluInputs(profileInputs[l], g_liveWeights[l],
                                         g_liveBiases[l], kOutCounts[l],
                                         kInCounts[l]);
    g_checksumSink += output[i % kOutputCount];
    vTaskDelay(1);
  }
  sortSamples(iterations);
  ProfileStats stats = {sum / iterations, percentile(iterations, 99), maximum, expCalls};
  return stats;
}

static void commandProfile(uint32_t iterations) {
  if (g_quantL1Int8) {
    Serial.println("{\"type\":\"error\",\"message\":\"profile_not_supported_with_quantization\"}");
    return;
  }
  if (!iterations) iterations = 128;
  if (iterations > kMaxBenchIterations) iterations = kMaxBenchIterations;
  ProfileStats layers[kLayerCount];
  for (int l = 0; l < kLayerCount; ++l) layers[l] = profilePass(l, iterations);
  const ProfileStats aggregate = profileAggregate(iterations);
  Serial.printf("{\"type\":\"profile\",\"kernel\":\"%s\",\"iterations\":%u,\"layers\":[",
                kernelName(), (unsigned)iterations);
  for (int l = 0; l < kLayerCount; ++l) {
    if (l) Serial.print(',');
    const uint64_t macs = (uint64_t)kInCounts[l] * (uint64_t)kOutCounts[l];
    Serial.printf("{\"layer\":%d,\"macs\":%llu,\"mean_us\":%.3f,\"p99_us\":%u,\"max_us\":%u,\"negative_elu_exp_calls\":%u}",
                  l, (unsigned long long)macs, layers[l].mean,
                  (unsigned)layers[l].p99, (unsigned)layers[l].maximum,
                  (unsigned)layers[l].expCalls);
  }
  Serial.printf("],\"aggregate\":{\"macs\":%llu,\"mean_us\":%.3f,\"p99_us\":%u,\"max_us\":%u,\"negative_elu_exp_calls\":%u},\"instrumentation\":\"per_layer_and_aggregate_timer_calls; per_layer_passes_recompute_untimed_prefix_layers\"}\n",
                (unsigned long long)((uint64_t)kInCounts[0] * kOutCounts[0] +
                                     (uint64_t)kInCounts[1] * kOutCounts[1] +
                                     (uint64_t)kInCounts[2] * kOutCounts[2] +
                                     (uint64_t)kInCounts[3] * kOutCounts[3]),
                aggregate.mean, (unsigned)aggregate.p99,
                (unsigned)aggregate.maximum, (unsigned)aggregate.expCalls);
}

static bool parsePositiveNumber(const char *text, uint32_t *out);

static void commandWeights(char *arg) {
  if (!arg) { Serial.println("{\"type\":\"error\",\"message\":\"weights_argument_required\"}"); return; }
  if (!strcmp(arg, "FLASH")) {
    if (g_weightsRamLayer >= 0) {
      const int old = g_weightsRamLayer;
      free(g_weightCopies[old]);
      g_weightCopies[old] = nullptr;
      g_liveWeights[old] = g_flashWeights[old];
      g_liveBiases[old] = g_flashBiases[old];
      g_weightsRamLayer = -1;
      g_weightsRamBytes = 0;
    }
    Serial.printf("{\"type\":\"weights\",\"layer\":-1,\"mode\":\"FLASH\",\"ram_bytes\":0,\"free_internal_heap\":%u,\"largest_internal_block\":%u}\n",
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    return;
  }
  if (strncmp(arg, "RAM", 3) || (arg[3] != ' ' && arg[3] != '\t')) {
    Serial.println("{\"type\":\"error\",\"message\":\"weights_must_be_FLASH_or_RAM_layer\"}"); return;
  }
  arg += 3;
  while (*arg == ' ' || *arg == '\t') ++arg;
  char *end = arg + strlen(arg);
  while (end > arg && (end[-1] == ' ' || end[-1] == '\t')) *--end = '\0';
  uint32_t parsedLayer = 0;
  if (!parsePositiveNumber(arg, &parsedLayer) || parsedLayer >= kLayerCount) {
    Serial.println("{\"type\":\"error\",\"message\":\"weights_layer_must_be_0_to_3_or_FLASH\"}"); return;
  }
  const int layer = (int)parsedLayer;
  if (layer == 1 && g_quantL1Int8) {
    Serial.println("{\"type\":\"error\",\"message\":\"disable_quantization_before_ram_layer1\"}");
    return;
  }
  if (layer == g_weightsRamLayer) {
    Serial.printf("{\"type\":\"weights\",\"layer\":%d,\"mode\":\"RAM\",\"ram_bytes\":%u,\"free_internal_heap\":%u,\"largest_internal_block\":%u}\n",
                  layer, (unsigned)g_weightsRamBytes,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    return;
  }
  const size_t floats = (size_t)kInCounts[layer] * kOutCounts[layer] + kOutCounts[layer];
  const size_t bytes = floats * sizeof(float);
  float *copy = (float *)heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!copy) {
    Serial.printf("{\"type\":\"error\",\"message\":\"weights_ram_allocation_failed\",\"layer\":%ld,\"requested_bytes\":%u,\"free_internal_heap\":%u,\"largest_internal_block\":%u}\n",
                  (long)layer, (unsigned)bytes,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    return;
  }
  const int l = layer;
  memcpy(copy, g_liveWeights[l], (size_t)kInCounts[l] * kOutCounts[l] * sizeof(float));
  memcpy(copy + (size_t)kInCounts[l] * kOutCounts[l], g_liveBiases[l], (size_t)kOutCounts[l] * sizeof(float));
  // Allocation and copy succeeded; only now release the previous selection.
  if (g_weightsRamLayer >= 0) {
    const int old = g_weightsRamLayer;
    free(g_weightCopies[old]);
    g_weightCopies[old] = nullptr;
    g_liveWeights[old] = g_flashWeights[old];
    g_liveBiases[old] = g_flashBiases[old];
  }
  g_weightCopies[l] = copy;
  g_liveWeights[l] = copy;
  g_liveBiases[l] = copy + (size_t)kInCounts[l] * kOutCounts[l];
  g_weightsRamLayer = l;
  g_weightsRamBytes = bytes;
  Serial.printf("{\"type\":\"weights\",\"layer\":%d,\"mode\":\"RAM\",\"ram_bytes\":%u,\"free_internal_heap\":%u,\"largest_internal_block\":%u}\n",
                l, (unsigned)bytes,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
}

static volatile bool g_runDone = true;
static uint32_t g_runDurationMs = kDefaultRunDurationMs;
static void waitUntilUs(int64_t deadline) {
  for (;;) {
    const int64_t remaining = deadline - esp_timer_get_time();
    if (remaining <= 0) return;
    if (remaining > 2500) vTaskDelay(pdMS_TO_TICKS((uint32_t)((remaining - 1000) / 1000)));
    else esp_rom_delay_us((uint32_t)remaining);
  }
}

static void runWorker(void *) {
  float output[kOutputCount];
  const int64_t firstRelease = esp_timer_get_time() + kPeriodUs;
  const int64_t endTime = firstRelease + (int64_t)g_runDurationMs * 1000;
  int64_t release = firstRelease;
  uint32_t ticks = 0, misses = 0, skipped = 0;
  uint32_t maxStartLate = 0, maxCompute = 0, maxResponse = 0;
  double responseSum = 0.0;
  while (release < endTime && ticks < kMaxSamples) {
    waitUntilUs(release);
    int64_t start = esp_timer_get_time();
    if (start < release) start = release;
    const uint32_t startLate = (uint32_t)(start - release);
    const int64_t inferStart = esp_timer_get_time();
    infer(test_inputs[ticks % kVectorCount], output);
    const int64_t finished = esp_timer_get_time();
    const uint32_t compute = (uint32_t)(finished - inferStart);
    const uint32_t response = (uint32_t)(finished - release);
    g_samples[ticks] = response;
    g_checksumSink += output[ticks % kOutputCount];
    ++ticks;
    if (startLate > maxStartLate) maxStartLate = startLate;
    if (compute > maxCompute) maxCompute = compute;
    if (response > maxResponse) maxResponse = response;
    if (response > kPeriodUs) ++misses;
    responseSum += response;
    release += kPeriodUs;
    while (release < endTime && release <= finished) { release += kPeriodUs; ++skipped; }
    taskYIELD();
  }
  sortSamples(ticks);
  bool btEnabled = false, btConnected = false;
#if POLICY_BENCH_HAS_BT
  btEnabled = SerialBT.hasClient() || SerialBT.isReady();
  btConnected = SerialBT.hasClient();
#endif
  Serial.printf("{\"type\":\"run\",\"quant_mode\":\"%s\",\"fallback_count\":%u,\"ticks\":%u,\"period_us\":%u,\"duration_ms\":%u,\"misses\":%u,\"skipped_ticks\":%u,\"max_start_lateness_us\":%u,\"max_compute_us\":%u,\"max_response_us\":%u,\"mean_response_us\":%.3f,\"p99_response_us\":%u,\"free_heap\":%u,\"min_heap\":%u,\"bt_enabled\":%s,\"bt_connected\":%s,\"checksum\":%.9g}\n",
                quantModeName(), (unsigned)g_quantFallbackCount,
                (unsigned)ticks, (unsigned)kPeriodUs, (unsigned)g_runDurationMs,
                (unsigned)misses, (unsigned)skipped, (unsigned)maxStartLate,
                (unsigned)maxCompute, (unsigned)maxResponse, ticks ? responseSum / ticks : 0.0,
                (unsigned)percentile(ticks, 99), (unsigned)ESP.getFreeHeap(),
                (unsigned)ESP.getMinFreeHeap(), btEnabled ? "true" : "false",
                btConnected ? "true" : "false", (double)g_checksumSink);
  g_runDone = true;
  vTaskDelete(nullptr);
}

static void drainBluetooth() {
#if POLICY_BENCH_HAS_BT
  while (SerialBT.available()) (void)SerialBT.read();
#endif
}

static void commandRun(uint32_t durationMs) {
  if (durationMs == 0) durationMs = kDefaultRunDurationMs;
  if (durationMs > kMaxRunDurationMs) durationMs = kMaxRunDurationMs;
  if (durationMs < 20) durationMs = 20;
  g_runDurationMs = durationMs;
  g_runDone = false;
  const BaseType_t created = xTaskCreatePinnedToCore(runWorker, "policy-run", 8192, nullptr, 5, nullptr, 1);
  if (created != pdPASS) { g_runDone = true; Serial.println("{\"type\":\"error\",\"message\":\"run_task_create_failed\"}"); return; }
  while (!g_runDone) { drainBluetooth(); vTaskDelay(pdMS_TO_TICKS(5)); }
}

static void commandBluetooth(bool enable) {
  bool enabled = false;
#if POLICY_BENCH_HAS_BT
  if (enable) { if (!SerialBT.isReady()) enabled = SerialBT.begin("LoongPolicyBench"); else enabled = true; }
  else { SerialBT.end(); enabled = false; }
#else
  (void)enable;
#endif
  bool connected = false;
#if POLICY_BENCH_HAS_BT
  connected = enabled && SerialBT.hasClient();
#endif
  Serial.printf("{\"type\":\"bt\",\"enabled\":%s,\"connected\":%s}\n", enabled ? "true" : "false", connected ? "true" : "false");
}

static void commandQuant(const char *arg) {
  if (!arg) {
    Serial.println("{\"type\":\"error\",\"message\":\"quant_argument_required\"}");
    return;
  }
  if (!strcmp(arg, "OFF")) {
    g_quantL1Int8 = false;
  } else if (!strcmp(arg, "L1_INT8")) {
    if (g_weightsRamLayer == 1) {
      Serial.println("{\"type\":\"error\",\"message\":\"return_layer1_weights_to_flash_before_quantization\"}");
      return;
    }
    g_quantL1Int8 = true;
  } else {
    Serial.println("{\"type\":\"error\",\"message\":\"quant_must_be_OFF_or_L1_INT8\"}");
    return;
  }
  Serial.printf("{\"type\":\"quant\",\"mode\":\"%s\",\"layer\":1,\"fallback_count\":%u}\n",
                quantModeName(), (unsigned)g_quantFallbackCount);
}

static void commandQCheck() {
  if (g_weightsRamLayer == 1) {
    Serial.println("{\"type\":\"error\",\"message\":\"qcheck_requires_layer1_weights_in_flash\"}");
    return;
  }
  float fp32[kOutputCount], quantized[kOutputCount];
  const bool oldMode = g_quantL1Int8;
  const uint32_t fallbacksBefore = g_quantFallbackCount;
  for (int n = 0; n < kVectorCount; ++n) {
    g_quantL1Int8 = false;
    infer(test_inputs[n], fp32);
    g_quantL1Int8 = true;
    infer(test_inputs[n], quantized);
    Serial.printf("{\"type\":\"quant_prediction\",\"index\":%d,\"fp32\":[", n);
    for (int j = 0; j < kOutputCount; ++j) {
      if (j) Serial.print(',');
      Serial.printf("%.9g", (double)fp32[j]);
    }
    Serial.print("],\"quantized\":[");
    for (int j = 0; j < kOutputCount; ++j) {
      if (j) Serial.print(',');
      Serial.printf("%.9g", (double)quantized[j]);
    }
    Serial.println("]}");
  }
  g_quantL1Int8 = oldMode;
  Serial.printf("{\"type\":\"qcheck_done\",\"checked_vectors\":%d,\"fallbacks\":%u,\"fallback_count\":%u}\n",
                kVectorCount, (unsigned)(g_quantFallbackCount - fallbacksBefore),
                (unsigned)g_quantFallbackCount);
}

static bool parsePositiveNumber(const char *text, uint32_t *out) {
  if (!text || !*text) return false;
  uint32_t value = 0;
  for (const char *p = text; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
    const uint32_t digit = (uint32_t)(*p - '0');
    if (value > (UINT32_MAX - digit) / 10U) return false;
    value = value * 10U + digit;
  }
  *out = value;
  return true;
}

static void processCommand(char *line) {
  while (*line == ' ' || *line == '\t') ++line;
  if (!*line) return;
  char *arg = strchr(line, ' ');
  if (arg) { *arg++ = '\0'; while (*arg == ' ' || *arg == '\t') ++arg; if (!*arg) arg = nullptr; }
  if (!strcmp(line, "INFO") && !arg) printInfo();
  else if (!strcmp(line, "CHECK") && !arg) commandCheck();
  else if (!strcmp(line, "BENCH") || !strcmp(line, "PROFILE") || !strcmp(line, "RUN")) {
    uint32_t number = 0;
    if (arg && !parsePositiveNumber(arg, &number)) { Serial.println("{\"type\":\"error\",\"message\":\"invalid_numeric_argument\"}"); return; }
    if (!strcmp(line, "BENCH")) commandBench(number);
    else if (!strcmp(line, "PROFILE")) commandProfile(number);
    else commandRun(number);
  }
  else if (!strcmp(line, "KERNEL") && arg) {
    if (!strcmp(arg, "C")) g_kernel = POLICY_KERNEL_C;
    else if (!strcmp(arg, "DSP")) g_kernel = POLICY_KERNEL_DSP;
    else if (!strcmp(arg, "UNROLL4")) g_kernel = POLICY_KERNEL_UNROLL4;
    else { Serial.println("{\"type\":\"error\",\"message\":\"kernel_must_be_C_DSP_or_UNROLL4\"}"); return; }
    Serial.printf("{\"type\":\"kernel\",\"name\":\"%s\"}\n", kernelName());
  }
  else if (!strcmp(line, "QUANT") && arg && !strcmp(arg, "OFF")) commandQuant(arg);
  
  else if (!strcmp(line, "WEIGHTS") && arg) commandWeights(arg);
  else if (!strcmp(line, "BT") && arg && (!strcmp(arg, "ON") || !strcmp(arg, "OFF"))) commandBluetooth(!strcmp(arg, "ON"));
  else if (!strcmp(line, "HELP") && !arg) Serial.println("{\"type\":\"help\",\"commands\":[\"INFO\",\"CHECK\",\"QCHECK\",\"BENCH [N]\",\"PROFILE [N]\",\"RUN [duration_ms]\",\"KERNEL C|DSP|UNROLL4\",\"QUANT OFF|L1_INT8\",\"WEIGHTS FLASH|RAM layer\",\"BT ON|OFF\"]}");
  else Serial.println("{\"type\":\"error\",\"message\":\"invalid_command_or_arguments\"}");
}

void setup() {
  Serial.begin(115200);
  delay(200);
  printInfo();
}

void loop() {
  static char line[128];
  static size_t used = 0;
  while (Serial.available()) {
    const char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') { line[used] = '\0'; processCommand(line); used = 0; }
    else if (used + 1 < sizeof(line)) line[used++] = c;
    else { used = 0; Serial.println("{\"type\":\"error\",\"message\":\"command_too_long\"}"); }
  }
  delay(1);
}
