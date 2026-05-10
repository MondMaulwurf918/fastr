# ⚡ FASTR

**A GPU-accelerated tensor runtime for Rust.** Talk to your NVIDIA GPU directly — no CUDA toolkit, no Python, no external dependencies at runtime.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://rust-lang.org)

---

## What is FASTR?

FASTR is a tensor computation library that lets you write GPU-accelerated code in pure Rust. It handles device memory, kernel dispatch, and tensor operations so you can focus on your models — not CUDA boilerplate.

```
Your Rust code  ──→  FASTR  ──→  nvcuda.dll  ──→  GPU
```

No CUDA toolkit. No Python. No FFI headaches.

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fastr-core = "0.3"
```

**Requirements:**
- NVIDIA GPU with driver 535 or newer
- Rust 1.70 or newer
- Windows (Linux support planned)

---

## Quick Start

```rust
use std::sync::Arc;
use fastr_core::device::Gpu;
use fastr_core::alloc::Allocator;
use fastr_core::tensor::Tensor;
use fastr_core::eager::{self, KernelCache};

fn main() -> Result<(), String> {
    // Connect to GPU 0 (primary GPU).
    let gpu = Arc::new(Gpu::init(0)?);

    // Create a 2 GB memory pool.
    let mut alloc = Allocator::new(gpu.clone(), 2.0)?;

    // Cache for compiled GPU kernels.
    let mut cache = KernelCache::new(&gpu);

    // Allocate three tensors of 1000 f32 elements each.
    let a = Tensor::<f32>::zeros(&mut alloc, &[1000])?;
    let b = Tensor::<f32>::zeros(&mut alloc, &[1000])?;
    let c = Tensor::<f32>::zeros(&mut alloc, &[1000])?;

    // Run element-wise addition: c = a + b.
    eager::add(&gpu, &mut cache, &a, &b, &c)?;

    // Run ReLU in-place: c = relu(c).
    eager::relu(&gpu, &mut cache, &c, &c)?;

    // Wait for GPU to finish.
    gpu.synchronize()?;

    Ok(())
}
```

---

## Features at a Glance

| Category | What's included |
|---|---|
| **Device** | GPU init, memory allocation (device + pinned), host↔device copy, synchronization |
| **Memory** | Pool allocator — one GPU allocation, many tensor slices. No per-tensor `cudaMalloc` |
| **Tensors** | Multi-dimensional arrays with shapes, strides, zero-copy views, and reshape |
| **Element-wise** | `add`, `mul`, `relu` |
| **Matmul** | FP32 tiled matrix multiply, FP16 tensor core matrix multiply |
| **Reductions** | Sum (multi-pass, exact) |
| **Neural network** | LayerNorm, Softmax, Dropout, MSE loss, Cross-entropy loss |
| **Backward pass** | Gradients for relu, matmul, bias, MSE |
| **Optimizers** | SGD, Adam (with momentum and variance buffers) |
| **Weight init** | Xavier/Glorot uniform |
| **Multi-GPU** | Automatic work splitting based on per-device benchmarks |
| **Profiling** | Per-GPU capability detection (compute, bandwidth) |

---

## Performance

Real measurements on the FASTR training benchmark (64→128→10 MLP, batch=32, SGD, 2000 iterations after warmup):

| GPU | Training Throughput | Per-Iter Latency | FP32 Matmul (2048²) |
|---|---|---|---|
| RTX 4060 (8 GB) | **6,009 iter/s** | 0.17 ms | 999 GFLOPS |
| RTX 5060 Ti (16 GB) | 4,363 iter/s | 0.23 ms | **1,549 GFLOPS** |

The RTX 4060 achieves higher training throughput because the loop is PCIe-bound — its direct motherboard connection (PCIe 4.0 x8) beats the 5060 Ti's OCuLink interface for host↔device transfers. For pure compute (matmul), the 5060 Ti pulls ahead with 55% higher throughput.

Full methodology and raw numbers: **[BENCHMARK.md](BENCHMARK.md)**

---

## Training a Neural Network

Complete training loop — forward, backward, update — all on GPU:

```rust
use std::sync::Arc;
use fastr_core::device::Gpu;
use fastr_core::alloc::Allocator;
use fastr_core::tensor::Tensor;
use fastr_core::eager::{self, KernelCache};

fn main() -> Result<(), String> {
    let gpu = Arc::new(Gpu::init(0)?);
    let mut alloc = Allocator::new(gpu.clone(), 2.0)?;
    let mut cache = KernelCache::new(&gpu);

    // ---- Model: 64 → 128 → 10 MLP ----
    let w1 = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
    let b1 = Tensor::<f32>::zeros(&mut alloc, &[1, 128])?;
    let w2 = Tensor::<f32>::zeros(&mut alloc, &[128, 10])?;
    let b2 = Tensor::<f32>::zeros(&mut alloc, &[1, 10])?;

    // ---- Weight initialization ----
    eager::xavier_uniform(&gpu, &mut cache, &w1, 64, 128, 42)?;
    eager::xavier_uniform(&gpu, &mut cache, &w2, 128, 10, 43)?;

    // ---- Activation & gradient buffers ----
    let z1 = Tensor::<f32>::zeros(&mut alloc, &[32, 128])?;
    let a1 = Tensor::<f32>::zeros(&mut alloc, &[32, 128])?;
    let z2 = Tensor::<f32>::zeros(&mut alloc, &[32, 10])?;
    let dw1 = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
    let db1 = Tensor::<f32>::zeros(&mut alloc, &[1, 128])?;
    let dw2 = Tensor::<f32>::zeros(&mut alloc, &[128, 10])?;
    let db2 = Tensor::<f32>::zeros(&mut alloc, &[1, 10])?;
    let dz2 = Tensor::<f32>::zeros(&mut alloc, &[32, 10])?;
    let da1 = Tensor::<f32>::zeros(&mut alloc, &[32, 128])?;
    let dz1 = Tensor::<f32>::zeros(&mut alloc, &[32, 128])?;

    // ---- Data & loss buffers ----
    let x = Tensor::<f32>::zeros(&mut alloc, &[32, 64])?;
    let y = Tensor::<f32>::zeros(&mut alloc, &[32, 10])?;
    let loss = Tensor::<f32>::zeros(&mut alloc, &[1])?;

    // ---- Training loop ----
    for _ in 0..1000 {
        // Upload a batch of data from host to GPU.
        gpu.copy_to(x.ptr, bytemuck::cast_slice(&my_batch_x))?;
        gpu.copy_to(y.ptr, bytemuck::cast_slice(&my_batch_y))?;

        // ---- Forward pass ----
        eager::matmul(&gpu, &mut cache, &x, &w1, &z1, 32, 128, 64)?;
        eager::add(&gpu, &mut cache, &z1, &b1, &z1)?;
        eager::relu(&gpu, &mut cache, &z1, &a1)?;
        eager::matmul(&gpu, &mut cache, &a1, &w2, &z2, 32, 10, 128)?;
        eager::add(&gpu, &mut cache, &z2, &b2, &z2)?;
        eager::mse_loss(&gpu, &mut cache, &z2, &y, &loss)?;

        // ---- Backward pass ----
        eager::mse_backward(&gpu, &mut cache, &z2, &y, &dz2)?;
        eager::matmul_backward_dB(&gpu, &mut cache, &a1, &dz2, &dw2, 32, 10, 128)?;
        eager::bias_grad(&gpu, &mut cache, &dz2, &db2, 32, 10)?;
        eager::matmul_backward_dA(&gpu, &mut cache, &dz2, &w2, &da1, 32, 10, 128)?;
        eager::relu_backward(&gpu, &mut cache, &z1, &da1, &dz1)?;
        eager::matmul_backward_dB(&gpu, &mut cache, &x, &dz1, &dw1, 32, 128, 64)?;
        eager::bias_grad(&gpu, &mut cache, &dz1, &db1, 32, 128)?;

        // ---- Weight update (SGD) ----
        eager::sgd_update(&gpu, &mut cache, &w1, &dw1, 0.001)?;
        eager::sgd_update(&gpu, &mut cache, &b1, &db1, 0.001)?;
        eager::sgd_update(&gpu, &mut cache, &w2, &dw2, 0.001)?;
        eager::sgd_update(&gpu, &mut cache, &b2, &db2, 0.001)?;
    }

    gpu.synchronize()?;
    Ok(())
}
```

---

## Multi-GPU

FASTR automatically detects all GPUs in the system and computes optimal work-split ratios based on per-device benchmarks:

```rust
use fastr_core::multigpu::MultiGpu;
use fastr_core::profiler::TaskType;

let mut mgpu = MultiGpu::init(2)?;

// Get data-parallel split ratios.
let ratios = mgpu.split_ratios(TaskType::Fp32Matmul);
// Example: [0.39, 0.61] — GPU 0 gets 39%, GPU 1 gets 61%

// Access individual GPUs:
let gpu0_memory_used = mgpu.allocators[0].pool_used();
```

---

## Project Structure

```
fastr/
├── README.md               ← You are here
├── BENCHMARK.md            ← Benchmark methodology and results
├── DOCS.md                 ← Full API reference (every function)
├── LICENSE                 ← MIT
├── Cargo.toml              ← Crate manifest
├── build.rs                ← Downloads pre-compiled library
├── fastr-core/             ← Core library (device, ops, allocator)
├── fastr-bench/            ← Benchmark suite
├── benchmarks/             ← PyTorch reference benchmarks
├── scripts/                ← Release build scripts
└── internal/               ← GPU kernel sources (private)
```

---

## Documentation

- **[DOCS.md](DOCS.md)** — Complete API reference for every module and function
- **[BENCHMARK.md](BENCHMARK.md)** — Benchmark methodology and measured results
- **[GitHub Releases](https://github.com/MondMaulwurf918/fastr/releases)** — Pre-compiled bi
nary downloads

---

## License

MIT — use it, modify it, ship it. Attribution appreciated.
