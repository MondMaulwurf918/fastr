# FASTR

GPU-accelerated tensor runtime for Rust. Direct GPU execution with minimal host overhead.

## Quick Start

```rust
use fastr_core::device::Gpu;
use fastr_core::alloc::Allocator;
use fastr_core::tensor::Tensor;
use fastr_core::eager::{self, KernelCache};

let gpu = Arc::new(Gpu::init(0)?);
let mut alloc = Allocator::new(gpu.clone(), 2.0)?;
let mut cache = KernelCache::new(&gpu);

let a = Tensor::<f32>::zeros(&mut alloc, &[1000])?;
let b = Tensor::<f32>::zeros(&mut alloc, &[1000])?;
let c = Tensor::<f32>::zeros(&mut alloc, &[1000])?;

eager::add(&gpu, &mut cache, &a, &b, &c)?;
eager::relu(&gpu, &mut cache, &a, &c)?;
gpu.synchronize()?;
```

## Features

- GPU memory pool with pre-allocated device memory
- Multi-dimensional tensor abstraction with views and reshape
- Element-wise operations (add, mul, relu)
- Matrix multiplication (FP32 tiled, FP16 tensor core)
- Two-pass reduction (sum)
- Neural network operations: layernorm, softmax, dropout, mse loss, cross-entropy
- Backward pass: relu, matmul gradients, bias gradients, mse gradient
- Optimizers: SGD, Adam
- Xavier/Glorot weight initialization
- Multi-GPU work distribution
- Device capability auto-detection

## Performance

Measured on 64→128→10 MLP training loop (forward + backward + update), batch=32, SGD:

| GPU | Iter/s | Per-iter | FP32 Matmul (2048²) |
|---|---|---|---|
| RTX 4060 | 6,009 | 0.17 ms | 999 GFLOPS |
| RTX 5060 Ti | 4,363 | 0.23 ms | 1,549 GFLOPS |

Detailed methodology in [BENCHMARK.md](BENCHMARK.md).

## Training Example

```rust
let gpu = Arc::new(Gpu::init(0)?);
let mut alloc = Allocator::new(gpu.clone(), 2.0)?;
let mut cache = KernelCache::new(&gpu);

// 64 -> 128 -> 10 MLP
let w1 = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
let b1 = Tensor::<f32>::zeros(&mut alloc, &[1, 128])?;
let w2 = Tensor::<f32>::zeros(&mut alloc, &[128, 10])?;
let b2 = Tensor::<f32>::zeros(&mut alloc, &[1, 10])?;

eager::xavier_uniform(&gpu, &mut cache, &w1, 64, 128, 42)?;
eager::xavier_uniform(&gpu, &mut cache, &w2, 128, 10, 43)?;

// Forward + loss
eager::matmul(&gpu, &mut cache, &x, &w1, &z1, 32, 128, 64)?;
eager::add(&gpu, &mut cache, &z1, &b1, &z1)?;
eager::relu(&gpu, &mut cache, &z1, &a1)?;
eager::matmul(&gpu, &mut cache, &a1, &w2, &z2, 32, 10, 128)?;
eager::add(&gpu, &mut cache, &z2, &b2, &z2)?;
eager::mse_loss(&gpu, &mut cache, &z2, &y, &loss)?;

// Backward
eager::mse_backward(&gpu, &mut cache, &z2, &y, &dz2)?;
eager::matmul_backward_dB(&gpu, &mut cache, &a1, &dz2, &dw2, 32, 10, 128)?;
eager::matmul_backward_dA(&gpu, &mut cache, &dz2, &w2, &da1, 32, 10, 128)?;

// Update
eager::sgd_update(&gpu, &mut cache, &w2, &dw2, 0.001)?;
```

## Multi-GPU

```rust
use fastr_core::multigpu::MultiGpu;

let mut mgpu = MultiGpu::init(2)?;
let ratios = mgpu.split_ratios();
// Work automatically distributed based on device capability.
```

## API Reference

### Device (`fastr_core::device`)

```rust
let gpu = Gpu::init(device_index)?;
gpu.copy_to(gpu_ptr, host_slice)?;
gpu.copy_from(gpu_ptr, host_slice)?;
gpu.synchronize()?;
gpu.alloc_pinned(bytes) -> (*mut u8, usize);
```

### Memory Pool (`fastr_core::alloc`)

```rust
let mut alloc = Allocator::new(gpu, pool_gb)?;
let t = Tensor::<f32>::zeros(&mut alloc, &[rows, cols])?;
```

### Tensors (`fastr_core::tensor`)

```rust
t.numel()        // element count
t.bytes()        // memory size in bytes
t.view(...)      // zero-copy slice
t.reshape(...)   // zero-copy reshape
```

### Operations (`fastr_core::eager`)

| Category | Functions |
|---|---|
| Element-wise | `add`, `mul`, `relu` |
| Matmul | `matmul`, `matmul_f16` |
| Reduction | `sum` |
| Normalization | `layernorm`, `softmax` |
| Loss | `mse_loss`, `cross_entropy_loss` |
| Regularization | `dropout` |
| Backward | `mse_backward`, `relu_backward`, `matmul_backward_dA`, `matmul_backward_dB`, `bias_grad` |
| Optimization | `sgd_update`, `adam_update` |
| Init | `xavier_uniform` |

## Requirements

- NVIDIA GPU (driver 535+)
- Rust 1.70+
- No CUDA toolkit required at runtime

## License

MIT
