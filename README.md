# Candle-GCU
[![discord server](https://dcbadge.vercel.app/api/server/hugging-face-879548962464493619)](https://discord.com/channels/879548962464493619/1136218819447238726)
[![Latest version](https://img.shields.io/crates/v/candle-core.svg)](https://crates.io/crates/candle-core)
[![Documentation](https://docs.rs/candle-core/badge.svg)](https://docs.rs/candle-core)
![License](https://img.shields.io/crates/l/candle-core.svg)

Usability conquers the world! Candle-GPU is a compact high performance inference framework specifically encubating for LLMs to achieving maximal ease of use, compatibility and minimal efforts in developing and maintainness. Technically, Candle-GPU builds upon the open-source project Candle rised by Huggingface, which developed in Rust with a focus on performance of GPU support. By leveraging the efforts of accommodate tons of LLMs with the Candle framework, we focuses on how to achieve zero-cost extensions and abstractions to integrate GCU backend into Candle framework, and consequently adapts to all supported LLMs for high performance execution.

According to the roadmap of Candle community and project plan, we can expect the following highlights delievered by this project:
- **Supporting hundreds of LLMs**: huggingface is the de-facto standard and most popular organisation to present LLM facilities to developpers all over the world, the transformers in python has proven its ease of use and portability across hundreds of SOTA LLM models
- **Ease of use**: Minimal dev hassles for LLMs, we can write a LLAMA+LLAMA2 with approx. 400 LOC only.
- **Easy to develop and maintain**: Candle design also simplifies the efforts of AOT operation developping by extracting the sharing operations with micro-kernel mechanism. This design philosophy follows the classical BLIS project and many sota researches such as AMOS, CUTLASS, etc.
- **Advanced features filling up the roadmap**: The short future can be expected by the community roadmap, which highlights the satellites projects such as candle-transformers (an alternative product to huggingface's signature product: transformers); candle-accelerate (auto parallel framework) may supports the multi-dev and multi-node distributed training also with ease.

## Designed Workflow for supporting Enflame GCU
Candle + GCU Backend -> Ubridge -> UHHI -> GCU Runtime (http://git.enflame.cn/sw/caps)

![]() <img src="resources/cangle-gcu.png"  width="600">

## Develop Status

Currently, candle-gcu supports following models in candle-transformers. Notably, this progress couples with the community works

__TODO: update status of the following template__
| LLM Model | Model Intro Link | Supporting GPU | Supporting GCU |
|--|--|--|--|
| #1 | TBD |✅|×|
| #2 | TBD |✅|×|
| #3 | TBD |✅|×|
| #4 | TBD |✅|×| 
| #5 | TBD |✅|✅|
| #6 | TBD |✅|✅|
| #7 | TBA |?|?|
| #8 | TBA |?|?|
| #9 | TBA |?|?|
| #10 | TBA |✅|✅|
| #11 | TBA |✅|?|


## TODO
__Write corresponding GCU kernerls (written in TopsCC)__

## Sample (LLaMa2 Inference)
Download LLaMa2 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains the following files:

config.json             model-00001-of-00002.safetensors  pytorch_model-00001-of-00002.bin  special_tokens_map.json  tokenizer.model
convert.py              model-00002-of-00002.safetensors  pytorch_model-00002-of-00002.bin  tokenizer_config.json    tosafetensor.py
generation_config.json  pytorch_model.bin.index.json      tokenizer.json

Run the following command:

``` shell
cargo run --example llama --features gcu -- --local-weights THE_WEIGHT_FOLDER --prompt "Please give me 200 words about deep learning."
```

**The inference result is not correct because I haven't write all kernels. Currently, the entire workflow can be computed on GCU (i.e., all weights, inputs and outputs buffers were created on GCU). There are 9 types of GCU kernels need to be implemented, i.e., affine, binary, cast, conv, matmul (under testing), fill, indexing, reduce, and unary (finished). The referenceing CUDA kernels can be found in candle-kernels.**

## Get started

### Sample GCU Backend Impl for Candle

For Unary OP

```rust
impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el_count as u32);
        let ds = dev.htod_copy([dims, layout.stride()].concat()).w()?; //data layout buffer
        let src = &src.slice(layout.start_offset()..); //input slice
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), ubridge::UNARY)?; //load GCU kernel
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(el_count).w()?; //output buffer
        let params = (el_count, dims.len(), &ds, src, &out); //launch kernel params
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?; //kernel launch
        Ok(out)
    }
}
```

### Sample usage of ubridge

GCU Alloc Function: device alloc (Candle) -> alloc (ubridge) -> DeviceBuffer uninitialized (UHHI) -> CAPS/TopsAPI

``` rust
    pub fn alloc<T: DeviceCopy>(
        self: &Arc<Self>,
        len: usize,
    ) -> DeviceResult<GcuSlice<T>> {
        let device_ptr = if self.is_async {
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream.unwrap())? }
        } else {
            unsafe { DeviceBuffer::uninitialized(len)? }
        };
        Ok(GcuSlice {
            buffer: device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
        })
    }
```

CPU Input Buffers -> GCU Compute -> CPU Result Buffers

``` rust
match DeviceExecutor::get_gcu_executor(0) {
    Some(gcu_executor) => {

        // let rawptr = lhs.as_ptr().cast::<f32>();
        // let ltensor = DeviceTensor::from_pointer(rawptr, m * k, vec![b, m, k]).unwrap();
        let ltensor = DeviceTensor::from_vec_shape(&vec![1.0f32; b*m*k], vec![b, m, k]).unwrap();
        // let rawptr = rhs.as_ptr().cast::<f32>();
        // let rtensor = DeviceTensor::from_pointer(rawptr, k * n, vec![b, k, n]).unwrap();
        let rtensor = DeviceTensor::from_vec_shape(&vec![1.0f32; b*k*n], vec![b, k, n]).unwrap();
        let mut dst: Vec<f32> = Vec::with_capacity(b * m * n);
        unsafe { dst.set_len(b * m * n); }
        match gcu_executor.transposed_matmul_owned(&ltensor, &rtensor, true) {
            Ok(tensor) => {
                match tensor.to_cpu(&mut dst) {
                    Ok(_) => {
                        let ret = cast_ref::<Vec<f32>, Vec<T>>(&dst).unwrap();
                        return Ok(ret.to_owned());
                    }
                    _=> { panic!("Unable to copy results back to cpu!");}
                }
            }
            _=> {}
        }
    }
    _=> {  }
}
```

### Sample usage of UHHI

Example of UHAL/UHHI for neural network forward pass (on NVidia GPU & Enflame GCU)

Enflame GCU: Install Enflame Driver 2.4.1+ and CAPS (TopsCC, TopsRuntime)

``` rust
use cust_core::DeviceCopy;
use std::collections::HashMap;

//Import UHAL for common computing interface
use uhal::launch;
use uhal::error::{DeviceResult};
use uhal::{DriverLibraryTrait};
use uhal::module::{ModuleTrait};
use uhal::memory::{DeviceBufferTrait};
use uhal::stream::{StreamTrait, StreamFlags};

//Tops backend
#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops::module::TopsModule as Module;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;
#[cfg(feature = "cuda_backend")]
use cuda::stream::CuStream as Stream;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

//Load kernel module
fn load_module<'a>(name : &str) -> DeviceResult<Module>{
    #[cfg(feature = "tops_backend")]
    let ptx = format!("{}/kernels/{}.o", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("{}/kernels/{}.ptx", env!("CARGO_MANIFEST_DIR"), name).to_string();

    Module::from_file(&ptx)
}

//Neural network layer definition
struct Layer<'a, T: DeviceCopy> {
    op : &'a str,
    weight : Option<DeviceBuffer<T>>,
    input_size : (usize, usize),
    output_size : (usize, usize),
    out_ref : Option<&'a DeviceBuffer<T>>
}

//A 6-layer neural network forward pass
//Unified interface (UHAL) for CUDA and Tops backend
#[allow(non_snake_case)]
fn network_test() -> DeviceResult<()> {
    let _device = Api::quick_init(0)?;

    //The entire workflow computed on this stream without copy back & forth between GPU/GCU memory and host memory.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const N : usize = 16;
    const K : usize = 3;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.01f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.02f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "relu", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.5f32; K * K])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is convolution kernel for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "convolution", weight: Some(DeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N, N), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None},  //out (N - K + 1) x (N - K + 1)
        
        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)

        Layer::<f32> {op : "matmul", weight: None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, // no weight in the last layer
        Layer::<f32> {op : "gelu", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)
    ];

    //Buffers on device (GPU/GCU), initialized with values
    let mut matA = DeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = DeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = DeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = DeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    //For activation type mapping
    let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);

    //Reference to output
    let mut out_ref : Option<&DeviceBuffer<f32>> = None;
    let mut out_size : Option<(usize, usize)> = None;

    //Forward computing
    for layer in layers {
        if ["relu", "gelu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation";
            match load_module(function_name) {
                Ok(module) => {
                    let kernel = module.get_function(&function_name)?;
                    unsafe {
                        //Slightly difference calling parameter for GCU and GPU.
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            (layer.input_size.0 * layer.input_size.1) as i32,
                            map_act[layer.op] as i32
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            layer.output_size.0,
                            map_act[layer.op]
                        ));

                        result?;
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => { panic!("Failed to load kernel!"); }
            }
        } else if layer.op == "matmul" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;

                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            layer.output_size.0
                        ));

                        result?;
                    }
                    std::mem::swap(&mut matA, &mut matOut);
                    match layer.weight {
                        Some(w) => { matB = w;}
                        _ => { 
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => { panic!("\nFailed to load kernel (matmul)!"); }
            }
        } else if layer.op == "convolution" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[K as i32, K as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

                    unsafe {
                        
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr(),
                            channelInfo.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            layer.input_size.0 as i32, layer.input_size.1 as i32,
                            K as i32,
                            K as i32
                        ));

                        result?;
                    }

                    std::mem::swap(&mut matA, &mut matConvOut);
                    match layer.weight {
                        Some(w) => { matB = w;}
                        _ => { 
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);

                }
                _ => { panic!("\nFailed to load kernel (convolution)!"); }
            }
        } else {
            panic!("Operation {} not supported!", layer.op); 
        }
    }

    // Wait asynchronous kernels to finish.
    stream.synchronize()?;

    //Obtain results and print
    match out_ref {
        Some(out) => {
            let mut out_host = vec![0.0f32; out.len()];
            out.copy_to(&mut out_host[0..out.len()])?;
            match out_size {
                Some(sz) => {
                    let W = sz.0;
                    let H = sz.1;
                    println!("\n\nResults of forward pass******************");
                    for x in 0..H {
                        for y in 0..W {
                            print!("{:.5} ", out_host[x * W + y]);
                        }
                        println!("{}", "");
                    }
                }
                _ => { panic!("Unable to obtain compute result!") }
            }

        }
        _ => { panic!("Unable to obtain compute result!")}
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}
```

### Sample of UnaryOp for cangle-gcu

```C++
namespace tops {
template <typename T>
__device__ __host__ __forceinline__ constexpr int hvlength() {
  return 128 / sizeof(T);
}

} // namespace tops

__device__ __forceinline__
auto get_index() {
    std::size_t blockIndex = blockIdx.z*(gridDim.x*gridDim.y)
        + blockIdx.y*gridDim.x + blockIdx.x;
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return blockIndex*(blockDim.x*blockDim.y*blockDim.z) + threadIndex;
}

#define UNARY_OP(TYPE, VT, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    TYPE *inp, \
    TYPE *out) \
{ \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    std::size_t idx = get_index(); \
    constexpr std::size_t num_len = tops::hvlength<VT>(); \
    __valigned__ TYPE buffer1[num_len]; \
    tops::mdspan buf1(tops::Private, &buffer1, num_len); \
    tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
    tops::memcpy(ctx, buf1, src1); \
    const auto &x = tops::vload<VT>(buffer1);  \
    tops::mdspan dst(tops::Global, out + idx *num_len, num_len); \
    tops::vstore(FUNC, buffer1);  \
    tops::memcpy(ctx, dst, buf1); \
} \


template<typename T>
__device__ __forceinline__ T elu_fwd(T x, T alpha) {
  if (x > static_cast<T>(0)) {
    return x;
  }
  return alpha * (tops::exp<T>(x) - static_cast<T>(1));
}

//UnaryOp with additional parameter
#define UNARY_OP1(TYPE, VT, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    TYPE param, \
    TYPE *inp, \
    TYPE *out) \
{ \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    std::size_t idx = get_index(); \
    constexpr std::size_t num_len = tops::hvlength<VT>(); \
    __valigned__ TYPE buffer1[num_len]; \
    tops::mdspan buf1(tops::Private, &buffer1, num_len); \
    __valigned__ TYPE buffer2[num_len]; \
    tops::mdspan buf2(tops::Private, &buffer2, num_len); \
    tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
    tops::memcpy(ctx, buf1, src1); \
    const auto &x = tops::vload<VT>(buffer1);  \
    tops::mdspan dst(tops::Global, out + idx *num_len, num_len); \
    for (int i = 0; i < num_len; i++) { \
        buffer2[i] = FUNC; \
    } \
    tops::memcpy(ctx, dst, buf2); \
} \



UNARY_COPY_OP(tops::bfloat, vbfloat, ucopy_bf16)
UNARY_OP(tops::bfloat, vbfloat, uneg_bf16, tops::vneg<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, uexp_bf16, tops::vexp<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, ulog_bf16, tops::vlog<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, usin_bf16, tops::vsin<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, ucos_bf16, tops::vcos<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, uabs_bf16, tops::vabs<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, usqr_bf16, tops::vmul<vbfloat>(x, x))
UNARY_OP(tops::bfloat, vbfloat, usqrt_bf16, tops::vsqrt<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, ugelu_bf16, tops::vgelu<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, urelu_bf16, tops::vmax<vbfloat>(x, tops::vzero<vbfloat>())) 
UNARY_OP1(tops::bfloat, vbfloat, uelu_bf16, elu_fwd(x[i], param))



UNARY_COPY_OP(tops::half, vhalf, ucopy_f16)
UNARY_OP(tops::half, vhalf, uneg_f16, tops::vneg<vhalf>(x))
UNARY_OP(tops::half, vhalf, uexp_f16, tops::vexp<vhalf>(x))
UNARY_OP(tops::half, vhalf, ulog_f16, tops::vlog<vhalf>(x))
UNARY_OP(tops::half, vhalf, usin_f16, tops::vsin<vhalf>(x))
UNARY_OP(tops::half, vhalf, ucos_f16, tops::vcos<vhalf>(x))
UNARY_OP(tops::half, vhalf, uabs_f16, tops::vabs<vhalf>(x))
UNARY_OP(tops::half, vhalf, usqr_f16, tops::vmul<vhalf>(x, x))
UNARY_OP(tops::half, vhalf, usqrt_f16, tops::vsqrt<vhalf>(x))
UNARY_OP(tops::half, vhalf, ugelu_f16, tops::vgelu<vhalf>(x))
UNARY_OP(tops::half, vhalf, urelu_f16, tops::vmax<vhalf>(x, tops::vzero<vhalf>()))
UNARY_OP1(tops::half, vhalf, uelu_f16, elu_fwd(x[i], param))

UNARY_COPY_OP(int8_t, vchar, ucopy_i8)
UNARY_COPY_OP(uint8_t, vuchar, ucopy_u8)
UNARY_COPY_OP(int32_t, vint, ucopy_i32)
UNARY_COPY_OP(uint32_t, vuint, ucopy_u32)

UNARY_COPY_OP(float, vfloat, ucopy_f32)

UNARY_OP(float, vfloat, uneg_f32, tops::vneg<vfloat>(x))
UNARY_OP(float, vfloat, uexp_f32, tops::vexp<vfloat>(x))
UNARY_OP(float, vfloat, ulog_f32, tops::vlog<vfloat>(x))
UNARY_OP(float, vfloat, usin_f32, tops::vsin<vfloat>(x))
UNARY_OP(float, vfloat, ucos_f32, tops::vcos<vfloat>(x))
UNARY_OP(float, vfloat, uabs_f32, tops::vabs<vfloat>(x))
UNARY_OP(float, vfloat, usqr_f32, tops::vmul<vfloat>(x, x))
UNARY_OP(float, vfloat, usqrt_f32, tops::vsqrt<vfloat>(x))
UNARY_OP(float, vfloat, ugelu_f32, tops::vgelu<vfloat>(x))
UNARY_OP(float, vfloat, urelu_f32, tops::vmax<vfloat>(x, tops::vzero<vfloat>()))
```