# FASTR Benchmark Report

## Methodology

All benchmarks use identical:
- Model architecture
- Tensor sizes and batch dimensions
- Data generation (same random seed)
- Warmup iterations before measurement
- CUDA synchronization after every iteration
- Measurement via wall-clock timing (not async launch timing)

## Hardware

| GPU | Memory | Interface | Driver |
|---|---|---|---|
| RTX 4060 | 8 GB | PCIe 4.0 x8 (motherboard) | 572.xx |
| RTX 5060 Ti | 16 GB | OCuLink (M.2 slot, PCIe 4.0 x4 equiv.) | 572.xx |

## Training Benchmark

**Model:** 64 → 128 → 10 MLP  
**Operations per iteration:** forward (2× matmul + add + layernorm + relu + mse_loss) + backward (2× matmul grad + bias grad + relu grad + mse grad) + 4× SGD update  
**Batch size:** 32  
**Optimizer:** SGD, learning rate 0.001  
**Warmup:** 100 iterations  
**Measured:** 2,000 iterations  

### Results

| GPU | Time (2000 iters) | Iter/s | ms/iter |
|---|---|---|---|
| RTX 4060 | 0.333 s | **6,009** | 0.17 |
| RTX 5060 Ti | 0.458 s | **4,363** | 0.23 |

The RTX 4060 achieves higher iteration throughput despite lower compute because the training loop is PCIe-bound (host→device data upload per iteration). The 4060's direct motherboard PCIe 4.0 x8 connection (~16 GB/s) outperforms the 5060 Ti's OCuLink interface (~8 GB/s).

## Matmul Benchmark

**Size:** 2048 × 2048, FP32  
**Warmup:** 10 iterations  
**Measured:** 100 iterations  

| GPU | GFLOPS | Latency (ms) |
|---|---|---|
| RTX 4060 | 999 | 17.2 |
| RTX 5060 Ti | **1,549** | 11.1 |

For compute-bound workloads, the 5060 Ti provides 55% higher throughput. cuBLAS (used by PyTorch) achieves approximately 4–8× higher matmul throughput through hand-tuned assembly kernels. FASTR's FP32 matmul is a simple tiled kernel — optimization potential exists.

## Interpretation

**Where FASTR excels:**
- Low-overhead training loops (sub-millisecond per iteration)
- Host→device transfer overhead elimination through memory pooling
- Predictable single-GPU execution

**Where cuBLAS/cuDNN remain superior:**
- Raw matmul throughput (hand-tuned assembly vs generic PTX)
- Convolution operations
- Mixed-precision training with automatic loss scaling

**Limitations of current kernels:**
- FP32 matmul is a basic tiled implementation — throughput ceiling is ~1.5 TFLOPS on 5060 Ti
- No kernel fusion (each op is a separate kernel launch)
- No automatic mixed precision

## Reproducing

```bash
# FASTR
cargo run --release -p fastr-bench

# Requires: Rust 1.70+, NVIDIA GPU, nvcuda.dll
```
