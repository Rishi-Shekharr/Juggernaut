# ESP32 inference benchmark

This sketch benchmarks the walking policy with INT8 first-layer weights and FP32 remaining layers. It uses synthetic inputs. The measured ESP32-D0WD-V3 run averaged 10.60 ms for 1,000 inference calls, and completed 1,500 scheduled inference cycles in 30 seconds without a missed 20 ms deadline. This timing is for inference only; it excludes sensor reads and motor output.

The policy and quantized weights are included in the headers under `policy_benchmark/`. Install Arduino CLI, the ESP32 board package and ESP-DSP, then open or compile `policy_benchmark/policy_benchmark.ino`. The PowerShell build helper accepts `-ArduinoCli` to specify the executable and writes its build products locally. See the sketch for serial commands and benchmark details.
