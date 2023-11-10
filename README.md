# Candle-GCU
[![discord server](https://dcbadge.vercel.app/api/server/hugging-face-879548962464493619)](https://discord.com/channels/879548962464493619/1136218819447238726)
[![Latest version](https://img.shields.io/crates/v/candle-core.svg)](https://crates.io/crates/candle-core)
[![Documentation](https://docs.rs/candle-core/badge.svg)](https://docs.rs/candle-core)
![License](https://img.shields.io/crates/l/candle-core.svg)

Candle-GCU is designed as a compact, high-performance deep learning framework and is meticulously incubated for Language Model Models (LLMs) to prioritize ease of use, compatibility, and the minimization of development and maintenance efforts. Our technical approach is grounded in the foundation laid by the open-source Candle project, spearheaded by Huggingface. Notably, Candle-GCU is developed in Rust, with a deliberate focus on leveraging Enflame GCU capabilities for optimal performance.

At its core, Candle-GCU seeks to synergize with the broader LLM landscape by seamlessly accommodating a multitude of LLMs within the Candle framework. Central to our vision is the pursuit of zero-cost extensions and abstracts, facilitating the seamless integration of the GCU backend into the Candle framework. This adaptation is envisioned to empower all supported LLMs to achieve high-performance execution standards.

In alignment with the roadmap set forth by the Candle community and our project plan, several noteworthy deliverables are anticipated:

1. **Supporting Hundreds of LLMs**: Huggingface stands as the de facto standard and the foremost global repository for LLM resources. Their Transformers library, written in Python, has firmly established itself as a testament to ease of use and portability across an extensive array of state-of-the-art LLM models.

2. **Streamlined Usability**: Candle-GCU aims to minimize development complexities for LLMs. Remarkably, it enables the creation of a LLAMA+LLAMA2 model with a mere 400 lines of code, underscoring its commitment to user-friendly development practices.

3. **Development and Maintenance Simplification**: The Candle design philosophy strategically simplifies the complexities associated with Ahead-of-Time (AOT) operation development. This simplification is achieved through the judicious extraction of shared operations using a micro-kernel mechanism. Drawing inspiration from classical projects like BLIS and contemporary research endeavors such as AMOS and CUTLASS, this approach ensures efficient and maintainable code.

4. **Advanced Roadmap Features**: Looking ahead, the community roadmap envisions a host of advanced features. These include satellite projects like 'candle-transformers,' which serves as a noteworthy alternative to Huggingface's 'Transformers' library. Additionally, 'candle-accelerate,' an auto-parallel framework, is poised to facilitate multi-device and multi-node distributed training with exceptional ease.


## Designed Workflow for supporting Enflame GCU
Candle + GCU Backend -> Ubridge -> UHHI -> GCU Runtime (http://git.enflame.cn/sw/caps)

![]() <img src="resources/cangle-gcu.png"  width="600">

## Develop Status

Currently, candle-gcu supports following models in candle-transformers. Notably, this progress couples with the community works

__TODO: update status of the following template__
| LLM Model ID | LLM Model | Supporting GPU | Supporting PAVO | Supporting Dorado |
|--|--|--|--|--|
| #1 | LLAMA |✅|✅|x|
| #2 | LLAMA2 |✅|✅|x|
| #3 | BigCode |✅|✅|x|
| #4 | TBD |✅|✅|x| 
| #5 | TBD |✅|✅|x|
| #6 | TBD |✅|✅|x|
| #7 | TBA |✅|✅|x|
| #8 | TBA |✅|✅|x|
| #9 | TBA |✅|✅|x|
| #10 | TBA |✅|✅|x|
| #11 | TBA |✅|✅|x|

## Installation of dependencies 
To bootstrap this project, you should run follow cmd first to fetch all the submodules from its source repos:

Install GCU driver (2.7.1+), TopsCC and Runtime

```shell
sudo enflame-x86_64-gcc-2.4.7.run

#Install topscc into /user/lib (must)
sudo topscc_0.7.0-1_amd64.run /usr/lib

sudo dpkg -i topsruntime_2.4.7-1_amd64.deb
```

Install Rust and Cargo

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Update submodules (candle-gcu, ubridge, UHHI)

```shell
git submodule update --init --recursive
```

If you want to download the pretrained weights (for LLaMa2) from gitlab LFS (predownloaded by us), you can just run
```
// check this utility
git lfs fetch --all
```

## $\textcolor{green}{\text{TODO}}$
__Write the following unfinished GCU kernerls defined in Candle (written in TopsCC, refer to candle-kernels)__

**Unary** ✅: $\textcolor{red}{\text{copy}}$, neg, recip, $\textcolor{red}{\text{exp}}$, log, $\textcolor{red}{\text{sin}}$, $\textcolor{red}{\text{cos}}$, abs, $\textcolor{red}{\text{sqr}}$, $\textcolor{red}{\text{sqrt}}$, gelu, relu, elu

**Ternary**: $\textcolor{red}{\text{where}}$

**Reduce**: sum, fast_min, fast_max, fast_argmin, fast_argmax, $\textcolor{red}{\text{fast\_sum}}$

**Indexing**: $\textcolor{red}{\text{is}}$, gather, ia, sa

**Fill** ✅: fill

**Conv**: conv1d, conv2d, conv_transpose2d, avg_pool2d, max_pool2d, unsample_nearest2d

**Cast**✅: $\textcolor{red}{\text{cast}}$

**Binary** ✅: $\textcolor{red}{\text{add, div, mul, sub,}}$ minimum, maximum, ne, lt, le, gt, ge

**Affine** ✅: $\textcolor{red}{\text{affine}}$

**Matmul/Dot** ✅: $\textcolor{red}{\text{matmul}}$/dot

$\textcolor{green}{\text{Note}}$: $\textcolor{red}{\text{micro-kernels in red for large language models}}$, e.g., llama, chatglm, falcon, etc.

✅: Initial implementation done.

## Sample (LLaMa2 Inference)
Download LLaMa2 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains the following files:

config.json             model-00001-of-00002.safetensors  pytorch_model-00001-of-00002.bin  special_tokens_map.json  tokenizer.model
convert.py              model-00002-of-00002.safetensors  pytorch_model-00002-of-00002.bin  tokenizer_config.json    tosafetensor.py
generation_config.json  pytorch_model.bin.index.json      tokenizer.json

Run the following command:

``` shell
cd candle-gcu
cargo run --example llama --features gcu -- --local-weights THE_WEIGHT_FOLDER --prompt "Please give me 200 words about deep learning."
```

**The inference result is not correct because I haven't write all kernels. Currently, the entire workflow can be computed on GCU (i.e., all weights, inputs and outputs buffers were created on GCU). There are 9 types of GCU kernels need to be implemented, i.e., affine, binary, cast, conv, matmul (under testing), fill, indexing, reduce, and unary (finished). The referenceing CUDA kernels can be found in candle-kernels.**

## End-to-end debuging candle-gcu models + CAPS + GCU kernels (Rust/C++)
Candle-gcu enables end-to-end debuging for Rust and C++ code in a single environment (VS code).

1) Install LLDB debugger (the installer option will prompt during your first view of this project);
   
2) Install clangd for C++ code completion;
   
3) Install rust-analyzer for Rust code completion;
   
4) Build your CAPS project as debug mode;

5) In "UHHI/tops_backend/tops_raw/build.rs", revise the linked library to your CAPS build path;

6) In "candle-gcu/.cargo/config.toml", revise rpath search path to your CAPS build path;

7) Build your debug version of candle-gcu by executing "cargo build --example llama --features gcu";

8) Revise the .vscode/launch.json and set your own weight path (refer resources/weight_path_settings.png);

9)  Navigate to "candle-gcu/candle-examples/examples/llama/main.rs", there is a "debug" option in the main function, click the "debug";

10) Add breakpoints in candle-gcu (candle model or candle-core/gcu-backend/ubridge, etc.) and CAPS source code, e.g., launchkernel.

![]() <img src="resources/cross-language-debug.png"  width="900">

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

### Sample of UnaryOp kernel for cangle-gcu

``` c++
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

### Sample of Dot/Matmul kernel (TopsCC + Intrinsics) for cangle-gcu

``` c++
//m can be any size, k is divisible by tile_size
template <typename T, typename VT, FP dot_intrinsic>
__device__ void dot(
  T *lhs,
  T *rhs,
  T *out,
  int m,
  int k,
  int n) {
  constexpr int vlen = tops::hvlength<VT>();
  constexpr int tile_size = 1 * vlen;

  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;

  int lstride = tile_size;
  if (m < tile_size) { //for small m
    lstride = m;
  }

  int blockIndex = threadId / (n / tile_size);
  int threadIndex = threadId % (n / tile_size);
//   printf("blockIndex %d, threadIndex %d", blockIndex, threadIndex);

  __valigned__ T lhs_l1[lstride * tile_size];
  __valigned__ T rhs_l1[tile_size * tile_size];

  __valigned__ T out_l1[lstride * tile_size];
  __valigned__ T temp[lstride * tile_size];

  tops::mdspan out_l1_(out_l1, lstride, tile_size);

  tops::mdspan srcl_l1(lhs_l1, lstride, tile_size);
  tops::mdspan srcr_l1(rhs_l1, tile_size, tile_size);

  tops::mdspan srcl_l3(lhs, m, k);
  tops::mdspan srcr_l3(rhs, k, n);
  tops::mdspan dst_l3(out, m, n);

  tops_dte_ctx_t ctx;   //L1-L3
  tops::dte_scope s(ctx);

  int idx_y = blockIndex * lstride;
  int idx_x = threadIndex * tile_size;

  if (idx_y < m) { //parallel
    int offsets_l[] = {idx_y, 0};
    if (idx_x < n) { //parallel
      tops::memset<T>(ctx, out_l1_, T(0)); //accumulation buffer
      tops::mdspan dst_l1(out_l1, lstride, tile_size);
      int offsets_r[] = {0, idx_x};

      for (int i = 0; i < k/tile_size; i++) { //k must be divisible by tile_size
        offsets_l[1] = i * tile_size;
        tops::slice(ctx, srcl_l1, srcl_l3, offsets_l); //slicing the left operand
        offsets_r[0] = i * tile_size;
        tops::slice(ctx, srcr_l1, srcr_l3, offsets_r);  //slicing the right operand
        // //dot_no_transpose
        auto lhs_address = (__attribute__((address_space(5))) T *)(lhs_l1);
        auto rhs_address = (__attribute__((address_space(5))) T *)(rhs_l1);
        auto out_address = (__attribute__((address_space(5))) T *)(temp);

        //call intrinsic core (two pieces of buffers, compute on L1)
        dot_intrinsic(reinterpret_cast<long long>(lhs_address),
                       reinterpret_cast<long long>(rhs_address),
                       reinterpret_cast<long long>(out_address),
                       lstride, //lstride can be any size <= tile_size
                       tile_size,
                       tile_size,
                       0,
                       1);

        for (auto i = 0; i < lstride * tile_size; i++) { //result accumulation
          out_l1[i] += temp[i];
        }
      }
      //L1->L3
      int offsets_o[] = {idx_y, idx_x};
      tops::deslice(ctx, dst_l3, dst_l1, offsets_o); //back to output buffer
    } 
  } 
}


extern "C" __global__ void dotllm_f16(const size_t m, const size_t k, const size_t n, tops::half *matA, tops::half *matB, tops::half* out)
{
    dot<tops::half, vhalf, kernel_dot_m_le256_fp16>(matA, matB, out, m, k, n);

}

```