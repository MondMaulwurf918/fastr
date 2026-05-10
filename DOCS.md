# FASTR API Reference

Complete documentation for every public type and function.

---

## Module: device — GPU Device

Handles GPU initialization, memory operations, and kernel execution.

### Gpu

Initialize with `Gpu::init(device_index)` and share via `Arc<Gpu>`.

| Method | Description |
|---|---|
| `gpu.copy_to(dst, src)` | Copy host data to GPU. Use `bytemuck::cast_slice` for typed data |
| `gpu.copy_from(src, dst)` | Copy GPU data to host. Pre-allocate dst buffer |
| `gpu.synchronize()` | Block until all GPU work completes |
| `gpu.alloc_raw(bytes)` | Raw GPU allocation. Prefer the Allocator pool |
| `gpu.alloc_pinned(bytes)` | Page-locked host memory for fast DMA |
| `gpu.launch_kernel(...)` | Low-level kernel launch. Use eager module instead |
| `device_name(idx)` | Query GPU name |

```rust
let gpu = Arc::new(Gpu::init(0)?);
gpu.copy_to(tensor.ptr, bytemuck::cast_slice(&data))?;
gpu.synchronize()?;
```

---

## Module: alloc — Memory Pool

Pre-allocates one GPU memory block at startup. All tensor allocations use slices of this block.

### Allocator

| Method | Description |
|---|---|
| `Allocator::new(gpu, total_gb)` | Create a memory pool. Size in gigabytes |
| `alloc.alloc(size)` | Allocate bytes from the pool. Used by Tensor::zeros |
| `alloc.free(allocation)` | Return memory to the pool. Merges adjacent free blocks |
| `alloc.pool_used()` | Current pool usage in GB |

```rust
let mut alloc = Allocator::new(gpu.clone(), 2.0)?;
let t = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
```

### Allocation

| Field | Type | Description |
|---|---|---|
| `ptr` | DevicePtr | GPU memory address |
| `offset` | usize | Byte offset into pool |
| `size` | usize | Size in bytes |

---

## Module: tensor — Tensors

Multi-dimensional GPU arrays with shape and stride metadata.

### Tensor<T: Dtype>

| Field | Type | Description |
|---|---|---|
| `ptr` | DevicePtr | GPU memory address |
| `shape` | Vec<usize> | Dimensions |
| `strides` | Vec<usize> | Step per dimension |
| `numel` | usize | Total element count |

| Method | Description |
|---|---|
| `Tensor::<T>::zeros(allocator, shape)` | Allocate zero-filled tensor. T = f32, u16, i32 |
| `tensor.numel()` | Total number of elements |
| `tensor.bytes()` | Total memory in bytes |
| `tensor.view(offset, shape, strides)` | Zero-copy slice |
| `tensor.reshape(shape)` | Zero-copy reshape (same numel required) |

```rust
let w = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
let f16 = Tensor::<u16>::zeros(&mut alloc, &[256])?;
let view = full.view(0, &[16, 128], &[128, 1]);
let flat = matrix.reshape(&[matrix.numel()])?;
```

---

## Module: dtype — Data Types

| Type | Description |
|---|---|
| `f32` | 32-bit float |
| `u16` | 16-bit unsigned (FP16 storage) |
| `i32` | 32-bit signed integer |

| Function | Description |
|---|---|
| `f32_to_f16(values)` | Convert f32 slice to FP16 bits (Vec<u16>) |
| `f16_to_f32_bits(bits)` | Convert single FP16 bit pattern to f32 |

```rust
let f16_data = fastr_core::dtype::f32_to_f16(&f32_data);
gpu.copy_to(f16_tensor.ptr, bytemuck::cast_slice(&f16_data))?;
```

---

## Module: eager — Kernel Dispatch

All tensor operations. Every function launches a GPU kernel immediately.

### KernelCache

Caches compiled PTX modules. Pass `&mut KernelCache` to every operation.

```rust
let mut cache = KernelCache::new(&gpu);
```

### Element-wise Operations

| Function | Signature | Description |
|---|---|---|
| `add` | `(gpu, cache, a, b, out)` | out[i] = a[i] + b[i] |
| `mul` | `(gpu, cache, a, b, out)` | out[i] = a[i] * b[i] |
| `relu` | `(gpu, cache, input, output)` | output[i] = max(0, input[i]) |

All tensors must have the same number of elements. `relu` supports in-place (same input/output).

```rust
eager::add(&gpu, &mut cache, &a, &b, &c)?;
eager::relu(&gpu, &mut cache, &x, &x)?;  // in-place
```

### Reduction

| Function | Signature | Description |
|---|---|---|
| `sum` | `(gpu, cache, input, output, partials)` | Sum all elements into scalar output [1] |

`partials` is a workspace tensor of size ~n/256.

```rust
let result = Tensor::<f32>::zeros(&mut alloc, &[1])?;
let workspace = Tensor::<f32>::zeros(&mut alloc, &[40])?;
eager::sum(&gpu, &mut cache, &data, &result, &workspace)?;
```

### Matrix Multiplication

| Function | Description |
|---|---|
| `matmul(gpu, cache, a, b, c, m, n, k)` | FP32: C = A @ B  (A: m x k, B: k x n, C: m x n) |
| `matmul_f16(gpu, cache, a, b, c, m, n, k)` | FP16 tensor core. A,B = Tensor<u16>, C = Tensor<f32> |

```rust
// FP32: 64 x 128 @ 128 x 10 = 64 x 10
eager::matmul(&gpu, &mut cache, &x, &w, &z, 64, 10, 128)?;

// FP16 tensor core: requires RTX 20-series+
gpu.copy_to(a16.ptr, bytemuck::cast_slice(&f32_to_f16(&host_a)))?;
eager::matmul_f16(&gpu, &mut cache, &a16, &b16, &c32, 2048, 2048, 2048)?;
```

### Neural Network — Forward

| Function | Description |
|---|---|
| `layernorm(gpu, cache, input, output, gamma, beta, rows, cols)` | Layer norm per row. Gamma/beta = [cols] |
| `softmax(gpu, cache, input, output, rows, cols)` | Softmax per row |
| `dropout(gpu, cache, input, output, mask, p, seed)` | Random dropout with probability p |

```rust
// LayerNorm: init gamma to 1, beta to 0
eager::layernorm(&gpu, &mut cache, &z1, &a1, &gn, &bt, 32, 128)?;
// Softmax
eager::softmax(&gpu, &mut cache, &logits, &probs, batch, classes)?;
// Dropout (mask is a workspace tensor of same shape)
eager::dropout(&gpu, &mut cache, &a1, &a1_d, &mask, 0.1, 42)?;
```

### Loss Functions

| Function | Description |
|---|---|
| `mse_loss(gpu, cache, pred, target, loss)` | MSE: mean((pred - target)^2). loss = [1] |
| `cross_entropy_loss(gpu, cache, probs, targets, loss, batch, classes)` | Cross-entropy. Use after softmax. targets = one-hot |

### Backward Pass (Gradients)

Pattern for A @ B = C backward:
- dC comes from loss
- dA = dC @ B^T (`matmul_backward_dA`)
- dB = A^T @ dC (`matmul_backward_dB`)

| Function | Description |
|---|---|
| `mse_backward(gpu, cache, pred, target, grad)` | dL/dpred = 2(pred-target)/N |
| `relu_backward(gpu, cache, input, grad_out, grad_in)` | Pass-through where input > 0 |
| `matmul_backward_dA(gpu, cache, dC, B, dA, M, N, K)` | dA = dC @ B^T |
| `matmul_backward_dB(gpu, cache, A, dC, dB, M, N, K)` | dB = A^T @ dC |
| `bias_grad(gpu, cache, grad, bias_grad, batch, dim)` | Sum gradients across batch |

```rust
// dz2:[32,10]  w2:[128,10]  da1:[32,128]
eager::matmul_backward_dA(&gpu, &mut cache, &dz2, &w2, &da1, 32, 10, 128)?;
// a1:[32,128]  dz2:[32,10]  dw2:[128,10]
eager::matmul_backward_dB(&gpu, &mut cache, &a1, &dz2, &dw2, 32, 10, 128)?;
// Sum gradients for bias
eager::bias_grad(&gpu, &mut cache, &dz2, &db2, 32, 10)?;
```

### Optimizers

| Function | Description |
|---|---|
| `sgd_update(gpu, cache, weight, grad, lr)` | SGD: weight -= lr * grad |
| `adam_update(gpu, cache, w, m, v, g, lr, b1, b2, eps, b1t, b2t)` | Adam with momentum buffers |

Adam requires per-weight momentum (m) and velocity (v) tensors of the same shape.
b1t = 1 - b1^step, b2t = 1 - b2^step (bias correction terms).

```rust
// SGD
eager::sgd_update(&gpu, &mut cache, &w1, &dw1, 0.001)?;

// Adam
let m = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
let v = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
let bt1 = 1.0 - 0.9_f32.powi(step);
let bt2 = 1.0 - 0.999_f32.powi(step);
eager::adam_update(&gpu, &mut cache, &w, &m, &v, &g, 0.001, 0.9, 0.999, 1e-8, bt1, bt2)?;
```

### Weight Initialization

| Function | Description |
|---|---|
| `xavier_uniform(gpu, cache, weight, fan_in, fan_out, seed)` | Glorot/Xavier uniform |

```rust
eager::xavier_uniform(&gpu, &mut cache, &w1, 64, 128, 42)?;
```

---

## Module: profiler — Device Profiling

### SystemProfile

| Method | Description |
|---|---|
| `SystemProfile::benchmark(max_gpus)` | Benchmark each GPU (compute + bandwidth) |
| `profile.split_ratio(task)` | Optimal work-split ratios for multi-GPU |

### TaskType

```rust
pub enum TaskType { Fp32Matmul, Fp16Matmul, ElementWise, Bandwidth }
```

---

## Module: multigpu — Multi-GPU

### MultiGpu

| Method / Field | Description |
|---|---|
| `MultiGpu::init(max_gpus)` | Initialize and profile all GPUs |
| `mgpu.
split_ratios(task)` | Get per-GPU work fractions |
| `mgpu.gpu_count()` | Number of GPUs |
| `mgpu.gpus` | Vec<Arc<Gpu>> |
| `mgpu.allocators` | Vec<Allocator> (one per GPU) |
| `mgpu.caches` | Vec<KernelCache> (one per GPU) |

```rust
let mut mgpu = MultiGpu::init(2)?;
let ratios = mgpu.split_ratios(TaskType::Fp32Matmul);
```

---

## Module: graph -- Execution Graphs

Record and replay kernel launch sequences.

```rust
let stream = Stream::new()?;
let capture = Stream::begin_capture(&stream)?;
// ... launch kernels ...
let exec = Stream::end_capture(capture, &stream)?;
exec.launch()?;
```

---

## Complete Training Loop

64-128-10 MLP, forward + backward + SGD, all on GPU.

```rust
use fastr_core::device::Gpu;
use fastr_core::alloc::Allocator;
use fastr_core::tensor::Tensor;
use fastr_core::eager::{self, KernelCache};

fn main() -> Result<(), String> {
    let gpu = Arc::new(Gpu::init(0)?);
    let mut alloc = Allocator::new(gpu.clone(), 2.0)?;
    let mut cache = KernelCache::new(&gpu);
    let (I, H, O, B, LR) = (64i32, 128i32, 10i32, 32usize, 0.001f32);

    let w1 = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
    let b1 = Tensor::<f32>::zeros(&mut alloc, &[1, 128])?;
    let w2 = Tensor::<f32>::zeros(&mut alloc, &[128, 10])?;
    let b2 = Tensor::<f32>::zeros(&mut alloc, &[1, 10])?;
    eager::xavier_uniform(&gpu, &mut cache, &w1, 64, 128, 0)?;
    eager::xavier_uniform(&gpu, &mut cache, &w2, 128, 10, 1)?;

    let z1 = Tensor::<f32>::zeros(&mut alloc, &[B, 128])?;
    let a1 = Tensor::<f32>::zeros(&mut alloc, &[B, 128])?;
    let z2 = Tensor::<f32>::zeros(&mut alloc, &[B, 10])?;
    let dw1 = Tensor::<f32>::zeros(&mut alloc, &[64, 128])?;
    let db1 = Tensor::<f32>::zeros(&mut alloc, &[1, 128])?;
    let dw2 = Tensor::<f32>::zeros(&mut alloc, &[128, 10])?;
    let db2 = Tensor::<f32>::zeros(&mut alloc, &[1, 10])?;
    let dz2 = Tensor::<f32>::zeros(&mut alloc, &[B, 10])?;
    let da1 = Tensor::<f32>::zeros(&mut alloc, &[B, 128])?;
    let dz1 = Tensor::<f32>::zeros(&mut alloc, &[B, 128])?;
    let x = Tensor::<f32>::zeros(&mut alloc, &[B, 64])?;
    let y = Tensor::<f32>::zeros(&mut alloc, &[B, 10])?;

    for _ in 0..1000 {
        gpu.copy_to(x.ptr, bytemuck::cast_slice(&batch_x))?;
        gpu.copy_to(y.ptr, bytemuck::cast_slice(&batch_y))?;

        eager::matmul(&gpu, &mut cache, &x, &w1, &z1, 32, H, I)?;
        eager::add(&gpu, &mut cache, &z1, &b1, &z1)?;
        eager::relu(&gpu, &mut cache, &z1, &a1)?;
        eager::matmul(&gpu, &mut cache, &a1, &w2, &z2, 32, O, H)?;
        eager::add(&gpu, &mut cache, &z2, &b2, &z2)?;
        eager::mse_loss(&gpu, &mut cache, &z2, &y, &z2)?;

        eager::mse_backward(&gpu, &mut cache, &z2, &y, &dz2)?;
        eager::matmul_backward_dB(&gpu, &mut cache, &a1, &dz2, &dw2, 32, O, H)?;
        eager::bias_grad(&gpu, &mut cache, &dz2, &db2, 32, O)?;
        eager::matmul_backward_dA(&gpu, &mut cache, &dz2, &w2, &da1, 32, O, H)?;
        eager::relu_backward(&gpu, &mut cache, &z1, &da1, &dz1)?;
        eager::matmul_backward_dB(&gpu, &mut cache, &x, &dz1, &dw1, 32, H, I)?;
        eager::bias_grad(&gpu, &mut cache, &dz1, &db1, 32, H)?;

        eager::sgd_update(&gpu, &mut cache, &w1, &dw1, LR)?;
        eager::sgd_update(&gpu, &mut cache, &b1, &db1, LR)?;
        eager::sgd_update(&gpu, &mut cache, &w2, &dw2, LR)?;
        eager::sgd_update(&gpu, &mut cache, &b2, &db2, LR)?;
    }
    gpu.synchronize()?;
    Ok(())
}
```
