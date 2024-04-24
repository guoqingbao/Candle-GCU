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

__TODO: update status with the following template__
| LLM Model ID | LLM Model | Supporting GPU | Supporting Scorpio
|--|--|--|--|
| #1 | LLAMA/LLAMA2 |✅|✅|
| #2 | Mistral (v0.1, v0.2) |✅|✅|
| #3 | Phi (v1, v1.5, v2) |✅|✅|
| #4 | Yi |✅|✅|
| #5 | StableLM (v1, v1-zephyr, v2, v2-zephyr) |✅|✅|
| #6 | BigCode/StarCode |✅|✅|
| #7 | ChatGLM (v3) |✅|✅|
| #8 | QWen (v2) |✅|✅|
| #9 | Google Gemma (2b, 7b) |✅|✅|
| #10 | Blip-large (Multimodal) |✅|✅|
| #11 | Moondream-2 (Multimodal LLM) |✅|✅|
| #12 | RWKV (v5) |✅|TBD|
| #13 | Falcon |✅|TBD|
| #14 | Stable Diffusion (v1, v1.5, v2) |✅|TBD|

## Demo Video

<!-- <video autoplay loop muted id="video" width="630" height="500" controls="" preload="none" poster="StableLM Coding Inference">
	<source id="mp4" src="./resources/Candle-GCU-BigCode.mp4" type="video/mp4">
</video> -->
<img src="./resources/Candle-GCU-BigCode.gif" width="75%" height="75%" >

<img src="./resources/Candle-GCU-Gemma.gif" width="75%" height="75%" >

<img src="./resources/Candle-GCU-Moondream2.gif" width="75%" height="75%" >

## Installation of dependencies 
To bootstrap this project, you should run follow cmd first to fetch all the submodules from its source repos:

Install GCU driver (2.6+), CAPS (0.9+)

Run CAPS installation: driver installation outside docker, and topscc & runtime installation inside docker.

```shell
sudo ./TopsPlatform_0.9.1_deb_amd64.run 
export PATH=$PATH:/opt/tops/bin
```

Install Rust and Cargo

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
apt install libssl-dev
apt install pkg-config
```

Update submodules (candle-gcu, ubridge, UHHI)

```shell
git submodule update --init --recursive
```

## Switch between GPU & GCU development 

(enable one of the following default features under candle-examples/Cargo.toml & candle-core/Cargo.toml)

default = ["gcu", "scorpio"] #gcu scorpio

default = ["cuda"] #gpu cuda

default = ["cuda", "flash-attn"] #gpu cuda with flash attention (cutlass build)

## $\textcolor{green}{\text{TODO}}$
__Write the following unfinished GCU kernerls defined in Candle (written in TopsCC, refer to candle-kernels)__

**Unary** ✅: $\textcolor{red}{\text{copy}}$, neg, recip, $\textcolor{red}{\text{exp}}$, log, $\textcolor{red}{\text{sin}}$, $\textcolor{red}{\text{cos}}$, abs, $\textcolor{red}{\text{sqr}}$, $\textcolor{red}{\text{sqrt}}$, gelu, relu, elu

**Ternary** ✅: $\textcolor{red}{\text{where}}$

**Reduce** ✅: sum, fast_min, $`\textcolor{red}{\text{fast\_max}}`$, fast_argmin, fast_argmax, $`\textcolor{red}{\text{fast\_sum}}`$

**Indexing** ✅: $\textcolor{red}{\text{is}}$, gather, ia, sa

**Fill** ✅: fill

**Conv**: conv1d, conv2d, conv_transpose2d, avg_pool2d, max_pool2d, unsample_nearest2d

**Cast**✅: $\textcolor{red}{\text{cast}}$

**Binary** ✅: $\textcolor{red}{\text{add, div, mul, sub,}}$ minimum, maximum, ne, lt, le, gt, ge

**Affine** ✅: $\textcolor{red}{\text{affine}}$

**GEMM/Matmul/Dot** ✅: $\textcolor{red}{\text{gemm/matmul/dot}}$

$\textcolor{green}{\text{Note}}$: $\textcolor{red}{\text{micro-kernels in red for large language models}}$, e.g., llama, chatglm, falcon, etc.

✅: Initial implementation done.

## $\textcolor{green}{\text{Fused kernel}}$
**Softmax**✅: $\textcolor{green}{\text{softmax}}$

**Layernorm**✅: $\textcolor{green}{\text{layernorm}}$, $\textcolor{green}{\text{rmsNorm}}$

**Embedding**✅: $\textcolor{green}{\text{rotary embedding}}$

**Concat**✅: $\textcolor{green}{\text{kvconcat}}$

**Attention**: flash attention, scaled dot-product attention

✅: Naive implementation done.

## Sample LLM Inference (LLaMa2, Mistral, Phi-2, Yi, BigCode, StableLM, QWen, Gemma)
### 1. Download LLaMa2 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

config.json             model-00001-of-00002.safetensors   
tokenizer.json          model-00002-of-00002.safetensors    

Replace **/home/llama2_weights/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example llama --features gcu,scorpio -- --local-weights /home/llama2_weights/ --prompt "Instruct: Please talk about deep learning in 100 words. Output: "
```

**LLaMa2-7B Sample inference output (Scorpio X1, BF16):**
```
loading the model weights from meta-llama/Llama-2-7b-hf
building the model
starting the inference loop
Instruct: Please talk about deep learning in 100 words. Output: Deep learning is a subset of machine learning that involves the use of artificial neural networks to model and solve complex problems. It has been instrumental in achieving state-of-the-art performance in various applications such as image and speech recognition, natural language processing, and autonomous driving. Deep learning algorithms are capable of learning and improving on their own by automatically adjusting their internal parameters during training, allowing them to adapt to new data and tasks.
Batch size = 1: 92 tokens generated (1 x 92 tokens), throughput: 13.08 token/s (1 x 13.08 token/s)
Batch size = 32: 2944 tokens generated (32 x 92 tokens), throughput: 166.61 token/s (32 x 5.21 token/s)
Batch size = 64: 5888 tokens generated (64 x 92 tokens), throughput: 219.71 token/s (64 x 3.42 token/s)
Batch size = 96: 8832 tokens generated (96 x 92 tokens), throughput: 207.75 token/s (96 x 2.16 token/s)
Batch size = 128: 11776 tokens generated (128 x 92 tokens), throughput: 217.51 token/s (128 x 1.70 token/s)
```

### 2. Download Mistral weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main

config.json             model-00001-of-00003.safetensors  
tokenizer.json          model-00002-of-00003.safetensors  model-00003-of-00003.safetensors       

Replace **/home/mistral_7b/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example mistral --features gcu,scorpio -- --weight-files /home/mistral_7b/model-00001-of-00003.safetensors,/home/mistral_7b/model-00002-of-00003.safetensors,/home/mistral_7b/model-00003-of-00003.safetensors --tokenizer-file /home/mistral_7b/tokenizer.json --prompt "Please talk about deep learning in 100 words."
```

**Mistral-7B Sample inference output (Scorpio X1, BF16):**
```
loaded the model in 58.479424355s
Please talk about deep learning in 100 words. Deep learning is a subset of machine learning that uses artificial neural networks with three or more layers to learn and model complex relationships between data. Deep learning has achieved state-of-the-art results in various applications such as image recognition, speech recognition, natural language processing, and autonomous driving.
Batch size = 1: 60 tokens generated (1 x 60 tokens), throughput: 12.40 token/s (1 x 12.40 token/s)
Batch size = 32: 1920 tokens generated (32 x 60 tokens), throughput: 186.48 token/s (32 x 5.83 token/s)
Batch size = 64: 3840 tokens generated (64 x 60 tokens), throughput: 266.26 token/s (64 x 4.16 token/s)
Batch size = 96: 5760 tokens generated (96 x 60 tokens), throughput: 245.91 token/s (96 x 2.56 token/s)
Batch size = 128: 7680 tokens generated (128 x 60 tokens), throughput: 268.60 token/s (128 x 2.10 token/s)
```

### 3. Download Phi-2 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/microsoft/phi-2/tree/main

config.json             model-00001-of-00002.safetensors  
tokenizer.json          model-00002-of-00002.safetensors   

Replace **/home/phi2/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example phi --features gcu,scorpio -- --model 2 --weight-file /home/phi2/model-00001-of-00002.safetensors,/home/phi2/model-00002-of-00002.safetensors --tokenizer /home/phi2/tokenizer.json --config /home/phi2/config.json --prompt "Instruct: Please talk about deep learning in 100 words. Output: " --sample-len 100 --batch-size 96
```

**Phi-2 Sample inference output (Scorpio X1, BF16):**
```
loaded the model in 3.027007815s
starting the inference loop
Instruct: Please talk about deep learning in 100 words. Output: 
Deep learning is a subset of machine learning that involves artificial neural networks with multiple layers that are designed to recognize patterns and make decisions. It has become increasingly popular in recent years due to its ability to analyze large datasets and identify complex relationships between variables. Deep learning algorithms are able to detect subtle features from data that may not be obvious to the human eye, allowing them to make more accurate predictions and decisions. Deep learning is used in a variety of applications, from healthcare to finance to autonomous vehicles.

Batch size = 1: 100 tokens generated (1 x 100 tokens), throughput: 20.80 token/s (1 x 20.80 token/s)
Batch size = 32: 3200 tokens generated (32 x 100 tokens), throughput: 247.87 token/s (32 x 7.75 token/s)
Batch size = 64: 6400 tokens generated (64 x 100 tokens), throughput: 305.92 token/s (64 x 4.78 token/s)
Batch size = 96: 9600 tokens generated (96 x 100 tokens), throughput: 311.51 token/s (96 x 3.24 token/s)
Batch size = 128: 12800 tokens generated (128 x 100 tokens), throughput: 330.74 token/s (128 x 2.58 token/s)
```

### 4. Download Yi-6B weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/01-ai/Yi-6B-Chat/tree/main

model-00001-of-00003.safetensors     model-00002-of-00003.safetensors  
tokenizer.json          model-00003-of-00003.safetensors   

Replace **/home/yi-6b/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example yi --features gcu,scorpio -- --which 6b --weight-files /home/yi-6b/model-00001-of-00003.safetensors,/home/yi-6b/model-00002-of-00003.safetensors,/home/yi-6b/model-00003-of-00003.safetensors --tokenizer-file /home/yi-6b/tokenizer.json --prompt "请使用一百字简单介绍一下深度学习" --sample-len 150 --batch-size 1
```

**Yi-6B Sample inference output (Scorpio X1, BF16):**

```
loaded the model in 6.403688985s
请使用一百字简单介绍一下深度学习。深度学习是一种基于神经网络的学习算法，它能够自动从数据中学习特征和模式，从而实现对数据的分类、预测等任务。
深度学习的核心在于其多层的神经网络结构，这些层包括输入层、隐藏层和输出层。
在训练深度学习模型时，通常会使用大量的数据进行训练，以使得模型能够学习到数据的特征和模式，从而实现对数据的分类、预测等任务。

简而言之，深度学习是一种基于神经网络的学习算法，它能够自动从数据中学习特征和模式，从而实现对数据的分类、预测等任务。

在现代人工智能领域，深度学习是其中最为核心和关键的技术之一，它在很多不同的应用场景
Batch size = 1: 150 tokens generated (1 x 150 tokens), throughput: 14.23 token/s (1 x 14.23 token/s)
Batch size = 32: 4800 tokens generated (32 x 150 tokens), throughput: 208.64 token/s (32 x 6.52 token/s)
Batch size = 64: 9600 tokens generated (64 x 150 tokens), throughput: 277.61 token/s (64 x 4.34 token/s)
Batch size = 96: 14400 tokens generated (96 x 150 tokens), throughput: 272.21 token/s (96 x 2.84 token/s)
Batch size = 128: 19200 tokens generated (128 x 150 tokens), throughput: 294.27 token/s (128 x 2.30 token/s)
```

### 5.1 Download StableLM-3B weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/stabilityai/stablelm-zephyr-3b/tree/main

model.safetensors     
tokenizer.json            

Replace **/home/stablelm-zephyr-3b/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example stable-lm --features gcu,scorpio -- --which v1-zephyr --weight-files /home/stablelm-zephyr-3b/model.safetensors --tokenizer-file /home/stablelm-zephyr-3b/tokenizer.json --config /home/stablelm-zephyr-3b/config.json --prompt "Please talk about deep learning in 100 words."
```

**StableLM-v1 Sample inference output (Scorpio X1, BF16):**
```
loaded the model in 3.002149621s
Please talk about deep learning in 100 words.
Deep learning is a subset of machine learning that uses artificial neural networks to simulate the way human brains learn and process information. It involves training algorithms on large datasets with millions or billions of examples, allowing them to identify patterns and relationships that would be impossible for humans to detect. Deep learning has been applied to a wide range of tasks, including image recognition, natural language processing, speech recognition, autonomous vehicles, and medical diagnosis. As the amount of data continues to grow at an unprecedented rate, deep learning
Batch size = 1: 100 tokens generated (1 x 100 tokens), throughput: 23.69 token/s (1 x 23.69 token/s)
Batch size = 32: 3200 tokens generated (32 x 100 tokens), throughput: 204.00 token/s (32 x 6.38 token/s)
Batch size = 64: 6400 tokens generated (64 x 100 tokens), throughput: 259.52 token/s (64 x 4.06 token/s)
Batch size = 96: 9600 tokens generated (96 x 100 tokens), throughput: 271.47 token/s (96 x 2.83 token/s)
Batch size = 128: 12800 tokens generated (128 x 100 tokens), throughput: 288.38 token/s (128 x 2.25 token/s)
```

### 5.2 Download StableLM V2 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b

model.safetensors     
tokenizer.json            

Replace **/home/stablelm-v2/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example stable-lm --features gcu,scorpio -- --which v2-zephyr --weight-files /home/stablelm-v2/model.safetensors --tokenizer-file /home/stablelm-v2/tokenizer-gpt4.json --prompt "请使用不少于五百字来介绍一下深度学习。" --sample-len 1000
```

**StableLM-v2 Sample inference output (Scorpio X1, BF16):**
```
loaded the model in 8.676859952s
请使用不少于五百字来介绍一下深度学习。这里我将简要介绍一些关键词和概念，以便读者更好地理解这个话题。

首先是“神经网络”，它是深度学习中的核心组成部分。神经网络是由大量的权重（也称为参数）和偏置（或权重维护者）组成的。在这个过程中，我们通过训练数据（如图像、音频等）来向神经网络学习模式传递信息，以便它们能够更好地理解和预测新的问题。

其次是“优化算法”，这是用于训练神经网络的重要组成部分。优化算法可以帮助我们在训练过程中找到最佳参数设置，以便使模型能够更好地学习和预测数据。

“批处理”是一个非常有用的概念，在深度学习中，它指的是将多个数据点（如图像、音频等）进行一次性处理。这样可以减少计算资源消耗和提高效率。

“训练周期”是指在神经网络训练过程中，每次训练的周期数。通常情况下，训练周期会有多个随机选择（如使用某一种优化算法）。

“模型预测”是指将训练好的神经网络模式用于预测新的问题。这通常需要对输入数据进行转换，以便使其能够被解释和预测。

最后，我们还有“优化目标”，它指的是在训练过程中找到最佳参数设置，以便使模型能够更好地学习和预测数据。常见的优化目标包括损失函数、准确度等。

总之，深度学习是一项非常广泛应用的技术，它可以帮助我们解决各种问题，如自动识别、图像分割、语音识别和更多。通过深度学习，我们可以训练模型来学习和预测新的问题，从而提高了系统的可靠性和准确性。
Batch size = 1: 610 tokens generated (1 x 610 tokens), throughput: 27.29 token/s (1 x 27.29 token/s)
Batch size = 32:14304 tokens generated (32 x 447 tokens), throughput: 187.65 token/s (32 x 5.86 token/s)
Batch size = 64: 28608 tokens generated (64 x 447 tokens), throughput: 219.49 token/s (64 x 3.43 token/s)
```

### 6. Download Bigcode/Starcode weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/bigcode/starcoderbase

model.safetensors     
tokenizer.json            

Replace **/home/bigcode/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example bigcode --features gcu,scorpio -- --weight-file /home/bigcode/model.safetensors --tokenizer-file /home/bigcode/tokenizer.json --prompt "Write a Python program to train ResNet50 model on ImageNet."
```

**Bigcode Sample inference output (Scorpio X1):**
```
loaded the model in 2.505882679s
starting the inference loop
Write a Python program using Pytorch to train ResNet50 on ImageNet.

# +
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
# -

# ## Load the dataset

# +
transform = transforms.Compose([transforms.ToTensor(), transforms.
100 tokens generated (1 x 100 tokens), throughput: 41.11 token/s (1 x 41.11 token/s)
```

### 7. Download QWen weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/Qwen/Qwen-1_8B-Chat/tree/main

model.safetensors     
tokenizer.json            

Replace **/home/qwen-1.8b/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example qwen --features gcu,scorpio,async -- --model 1.8b --weight-files /home/qwen-1.8b/model.safetensors --tokenizer-file /home/qwen-1.8b/tokenizer.json --config /home/qwen-1.8b/config.json --prompt "请使用五百字来介绍一下深度学习。" --sample-len 1000
```

**QWen Sample inference output (Scorpio X1, BF16，batchsize=1):**
```
loaded the model in 1.644448254s
请使用五百字来介绍一下深度学习。 深度学习是一种人工智能技术，它模仿人脑神经网络的结构和功能，通过多层非线性变换和大量数据的学习，实现对复杂问题的自动识别、分类、预测和生成等任务。它的核心思想是构建多层次的神经网络模型，每一层都包含多个隐藏单元，这些隐藏单元之间的连接权重可以随着训练过程逐渐调整，以适应不同的输入特征和输出目标。

深度学习的基本流程包括以下几个步骤：首先，数据预处理，包括图像、语音、文本等领域的数据清洗、归一化、标准化等操作；其次，构建神经网络模型，通常采用卷积神经网络（CNN）、循环神经网络（RNN）或长短时记忆网络（LSTM）等结构；然后，训练神经网络模型，通过反向传播算法和梯度下降优化技术，调整网络参数以最小化损失函数；最后，使用测试集评估模型性能，并根据需要进行调优。

深度学习在图像识别、语音识别、自然语言处理等领域取得了显著的成果。例如，在图像识别方面，深度学习可以实现对图像中的物体、场景等特征的自动提取和分类，如ImageNet、COCO等大规模数据集上的目标检测、分割任务；在语音识别方面，深度学习可以实现对音频信号的高精度识别，如Google的语音助手、Amazon的Alexa等；在自然语言处理方面，深度学习可以实现文本生成、机器翻译、问答系统等任务，如BERT、Transformer等模型。

然而，深度学习也存在一些挑战和限制。首先，深度学习需要大量的数据和计算资源，且训练过程通常需要较长的时间和较高的计算成本；其次，深度学习的模型结构复杂，难以解释其内部机制，这在某些应用场景下可能会影响模型的可解释性；最后，深度学习的泛化能力较弱，容易出现过拟合或欠拟合的问题。

为了克服这些挑战，近年来出现了许多深度学习优化技术，如残差网络（ResNet）、注意力机制（Attention）等，它们通过引入额外的信息来提高模型的性能和鲁棒性。此外，一些新的深度学习框架，如PyTorch、TensorFlow等，也提供了丰富的工具和API，使得深度学习的研究和应用更加便捷和高效。

总的来说，深度学习是一种强大的人工智能技术，它通过模仿人脑神经网络的结构和功能，实现了对复杂问题的自动识别、分类、预测和生成等任务。尽管存在一些挑战和限制，但随着深度学习算法的不断优化和新的研究进展，其在各个领域的应用前景仍然广阔，有望为人类社会带来更多的创新和发展。
555 tokens generated (1 x 555 tokens), throughput: 29.51 token/s (1 x 29.51 token/s)
```

``` shell
cd candle-gcu
cargo run --release --example qwen --features gcu,scorpio -- --model 1.8b --weight-files /home/qwen-1.8b/model.safetensors --tokenizer-file /home/qwen-1.8b/tokenizer.json --prompt "请使用五百字来介绍一下深度学习。" --sample-len 100 --batch-size 256
```

**QWen Sample inference performance (Scorpio X1, BF16):**
```
Batch size = 1: 100 tokens generated (1 x 100 tokens), throughput: 31.74 token/s (1 x 31.74 token/s)
Batch size = 32: 3200 tokens generated (32 x 100 tokens), throughput: 521.12 token/s (32 x 16.29 token/s)
Batch size = 64: 6400 tokens generated (64 x 100 tokens), throughput: 715.87 token/s (64 x 11.19 token/s)
Batch size = 96: 9600 tokens generated (96 x 100 tokens), throughput: 730.48 token/s (96 x 7.61 token/s)
Batch size = 128: 12800 tokens generated (128 x 100 tokens), throughput: 840.35 token/s (128 x 6.57 token/s)
```

### 8. Download Gemma weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/google/gemma-2b-it

model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
tokenizer.json        
config.json    

Replace **/home/gemma-2b/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example gemma --features gcu,scorpio,async -- --weight-files /home/gemma-2b/model-00001-of-00002.safetensors,/home/gemma-2b/model-00002-of-00002.safetensors --tokenizer-file /home/gemma-2b/tokenizer.json --config-file /home/gemma-2b/config.json --prompt "Please talk about deep learning in 100 words." --sample-len 100
```

**Gemma Sample inference output (Scorpio X1, BF6):**
```
loaded the model in 3.016762052s
Please talk about deep learning in 100 words.

Deep learning is a subfield of machine learning that allows computers to learn from data without explicit programming. It involves the creation of artificial neural networks (ANNs) that mimic the structure and function of the human brain. These ANNs are trained on vast datasets, enabling them to identify patterns, make predictions, and solve problems. Deep learning has revolutionized various industries, including healthcare, finance, and transportation, by automating tasks, improving decision-making, and uncovering hidden insights.

Batch size = 1: 97 tokens generated (1 x 97 tokens), throughput: 31.18 token/s (1 x 31.18 token/s)
Batch size = 32: 3104 tokens generated (32 x 97 tokens), throughput: 581.81 token/s (32 x 18.18 token/s)
Batch size = 64: 6208 tokens generated (64 x 97 tokens), throughput: 855.73 token/s (64 x 13.37 token/s)
Batch size = 96: 9312 tokens generated (96 x 97 tokens), throughput: 836.45 token/s (96 x 8.71 token/s)
Batch size = 128: 12416 tokens generated (128 x 97 tokens), throughput: 958.75 token/s (128 x 7.50 token/s)
```

### 9. Download ChatGLM3 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/THUDM/chatglm3-6b

model-00001-of-00007.safetensors  model-00002-of-00007.safetensors  model-00003-of-00007.safetensors
model-00004-of-00007.safetensors  model-00005-of-00007.safetensors  model-00006-of-00007.safetensors
model-00007-of-00007.safetensors
tokenizer.json        
config.json    

Replace **/home/chatglm3-6b/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example chatglm --features gcu,scorpio -- --weight-file /home/chatglm3-6b/model-00001-of-00007.safetensor,/home/chatglm3-6b/model-00002-of-00007.safetensor,/home/chatglm3-6b/model-00003-of-00007.safetensor,/home/chatglm3-6b/model-00004-of-00007.safetensor,/home/chatglm3-6b/model-00005-of-00007.safetensor,/home/chatglm3-6b/model-00006-of-00007.safetensor,/home/chatglm3-6b/model-00007-of-00007.safetensor --tokenizer /home/chatglm3-6b/chatglm-tokenizer.json --prompt "请使用一百字介绍深度学习" --sample-len 100
```

**ChatGLM Sample inference output (Scorpio X1, BF6):**
```
loaded the model in 37.834555121s
starting the inference loop
请使用一百字介绍深度学习技术,包括其优点和缺点。深度学习技术是一种机器学习方法,通过模拟人脑神经网络来识别模式并进行预测。它的优点是可以处理大量复杂数据,并且能够自动提取特征,无需手动设计特征。此外,深度学习还可以进行端到端的训练,使得模型可以适应多种不同的任务。然而,深度学习也存在一些缺点,比如需要大量的计算资源和数据集,并且容易出现过拟合的情况。
92 tokens generated (12.20 token/s)
```

### 10. (Talk to an image!) Download Moondream-2 weights to a local folder (e.g., THE_WEIGHT_FOLDER), it should contains at least the following files:

Huggingface weights: https://huggingface.co/vikhyatk/moondream2

model.safetensors, tokenizer.json

Replace **/home/moondream2/** with your weight folder and run the following command on Scorpio:

``` shell
cd candle-gcu
cargo run --release --example moondream --features gcu,scorpio -- --model-file /home/moondream2/model.safetensors --tokenizer-file /home/moondream2/tokenizer.json --image /home/candle-gcu/candle-gcu/candle-examples/examples/yolo-v8/assets/bike.jpg --prompt "Is there any particular in this image?" --sample-len 300
```
![]() <img src="resources/bike.jpg"  width="500">

**Moondream Sample inference output (Scorpio X1, BF6):**
```
loaded the model in 9.231756591s
loaded and encoded the image Tensor[dims 3, 378, 378; f32] in 32.799601ms
starting the inference loop
 The image features a close-up of a sidewalk with a crack in it, and it appears to be a part of a larger picture. The crack is surrounded by a yellow line, which is likely a cautionary measure to prevent people from walking too close to the edge of the sidewalk. The sidewalk is made of concrete and has a few holes in it, indicating that it may have been damaged or worn over time.
Additionally, there is a yellow line on the ground, which is also a safety measure to prevent accidents or vehicles from veering off the sidewalk.
Overall, the image shows a sidewalk with a crack and a yellow line, emphasizing the need for caution and safety in the area.
Complete detailed textbook-level solutions
The image shows a close-up of a sidewalk with a crack in it, surrounded by a yellow line. The crack is located near the center of the image, and the yellow line extends from the top left corner to the bottom right corner. The sidewalk appears to be made of concrete and has a few holes in it, suggesting that it may have been damaged or worn over time. The presence of the yellow line indicates that the sidewalk is meant for pedestrian use and is intended to prevent people from walking too close to the edge, which could potentially lead to accidents or injuries.
The crack in the sidewalk is a common issue in urban areas, as it can be caused by various factors such as weathering, heavy foot traffic,
300 tokens generated (15.70 token/s)
```

**Currently, the entire workflow can be computed on GCU (i.e., all weights, inputs and outputs buffers were created on GCU). There are 9 types of GCU kernels that have been initially implemented, i.e., affine, binary, cast, matmul, fill, indexing, reduce, ternary and unary, in ubridge/kernels**

**LLaMa2 inference on CPU and GPU:**
```shell
//cpu
cargo run --release --example llama -- --local-weights /home/llama2_weights/ --prompt "Please talk about deep learning in 100 words."
//gpu
cargo run --release --example llama --features cuda -- --local-weights /home/llama2_weights/ --prompt "Please talk about deep learning in 100 words."
```

**Test candle components on GCU** (e.g., mlp, embedding, softmax, rmsnorm, maskfill, attention, etc.) for GCU (Scorpio):

```shell
cd candle-gcu
cargo run --release --example gcutest --features gcu,scorpio -- --local-weights /home/llama2_weights/
```

```
start the candle-gcu testing cases...
Test cache passed!
Test cast passed!
Test embedding passed!
Test softmax passed!
Test rmsnorm passed!
Test maskfill passed!
Test concat passed!
Test matmul passed!
Test block passed!
Test attention passed!
Test narrow passed!
Test rotary_embedding passed!
```

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
        let el_count = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el_count as u32);
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), ubridge::UNARY)?;
        let out = dev.alloc::<T>(el_count).w()?;
        let params = (el_count, src, &out);
        unsafe { func.launch(cfg, params) }.w()?;
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

GCU GEMM Compute with Tuner (fp16)

``` rust
            (GcuStorageSlice::F16(lhs), GcuStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..); //slicing left operand
                let rhs = &rhs.slice(rhs_l.start_offset()..); //slicing right operand
                let out = dev.alloc::<f16>(elem_count).w()?; //output buffer
                let bias = dev.alloc::<f16>(n).w()?; //this will be removed later.
                //gemm tuner
                let info = AtenGemmInfo::new(TopsopDataType::TopSopDataFp16, if m==1 {b} else {m}, m, k, n);
                let mut tune = AtenGemmTune::default();
                let tuner = AtenGemmTuner::new();
                tuner.tuner(&info, &mut tune);
                let param = GEMM_OP_PARAS::new(&info, &tune);//tuning results

                let kernel_name = "gemm_f16".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::GEMM)?;

                let cfg = GcuLaunchConfig::for_gemm();
                let params = (lhs, rhs, &out, &bias, //kernel launch params
                    param.input_dtype, b, m, k, n,
                    param.lhs_multicore, param.rhs_multicore, param.batch_multicore,
                    param.lhs_transpose, param.rhs_transpose,
                    param.alpha, param.beta, param.addmm_beta, param.bias,
                    param.sip_m, param.sip_k, param.sip_n
                );
                unsafe { func.launch(cfg, params) }.w()?; //launch kernel compute
                GcuStorageSlice::F16(out) //return results
            }
```

### Sample usage of UHHI

Example of UHAL/UHHI for neural network forward pass (on NVidia GPU & Enflame GCU)

Enflame GCU: Install TopsPlatform 0.8.3+ (Driver, TopsCC, TopsRuntime)

``` rust
//Example of UHAL for neural network forward pass (on NV GPU & Enflame GCU)
use cust_core::DeviceCopy;
use std::collections::HashMap;

//Import UHAL for common computing interfaces

use uhal::error::DeviceResult;
use uhal::launch;
use uhal::memory::DeviceBufferTrait;
use uhal::module::ModuleTrait;
use uhal::stream::{StreamFlags, StreamTrait};
use uhal::DriverLibraryTrait;
//Tops backend
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
#[cfg(feature = "tops_backend")]
use tops::module::TopsModule as Module;
#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;
#[cfg(feature = "tops_backend")]
use tops_backend as tops;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::stream::CuStream as Stream;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;

use crate::device_executor::DeviceExecutor;

fn load_module<'a>(name: &str) -> DeviceResult<Module> {
    #[cfg(not(feature = "scorpio"))]
    #[cfg(feature = "tops_backend")]
    let ptx = format!("{}/kernels/legacy/pavo/{}.topsfb", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "scorpio")]
    let ptx = format!("{}/kernels/legacy/scorpio/{}.topsfb", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("{}/kernels/gpu/{}.ptx", env!("CARGO_MANIFEST_DIR"), name).to_string();

    Module::from_file(&ptx)
}

struct Layer<'a, T: DeviceCopy> {
    op: &'a str,
    weight: Option<&'a DeviceBuffer<T>>,
    input_size: (usize, usize),
    output_size: (usize, usize),
    out_ref: Option<&'a DeviceBuffer<T>>,
}
pub fn get_block_grid(shape1: usize, shape0: usize) -> (usize, usize, usize) {
    let grid_a: usize = (shape1 + 16 - 1) / 16;
    let grid_b: usize = (shape0 + 16 - 1) / 16;
    return (16, grid_a, grid_b);
}

//A 6-layer neural network forward pass
//Unified interface (UHAL) for CUDA and Tops backend
#[allow(non_snake_case)]
pub fn network_test() -> DeviceResult<()> {
    let _device = Api::quick_init(0)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    const N: usize = 16;
    const K: usize = 3;
    let w1 = DeviceBuffer::from_slice(&vec![0.01f32; N * N])?;
    let w2 = DeviceBuffer::from_slice(&vec![0.02f32; N * N])?;
    let w3 = DeviceBuffer::from_slice(&vec![0.03f32; N * N])?;
    let w4 = DeviceBuffer::from_slice(&vec![0.04f32; N * N])?;
    let w5 = DeviceBuffer::from_slice(&vec![0.05f32; N * N])?;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {
            op: "batch_matmul_legacy",
            weight: Some(&w1),
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //weight is N x N matric for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //out N x N
        Layer::<f32> {
            op: "batch_matmul_legacy",
            weight: Some(&w2),
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //weight is N x N matric for next layer
        Layer::<f32> {
            op: "relu",
            weight: None,
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //out N x N
        Layer::<f32> {
            op: "batch_matmul_legacy",
            weight: Some(&w3),
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //weight is convolution kernel for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //out N x N
        Layer::<f32> {
            op: "convolution",
            weight: Some(&w4),
            input_size: (N, N),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //out (N - K + 1) x (N - K + 1)
        Layer::<f32> {
            op: "batch_matmul_legacy",
            weight: Some(&w5),
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //output shape (N - K + 1) * (N - K + 1)
        Layer::<f32> {
            op: "batch_matmul_legacy",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, // no weight in the last layer
        Layer::<f32> {
            op: "gelu",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //output shape (N - K + 1) * (N - K + 1)
    ];
    let mat = vec![0.5f32; N * N];
    let mato = vec![0.0f32; N * N];
    let convo = vec![0.0f32; (N - K + 1) * (N - K + 1)];

    let matA = DeviceBuffer::from_slice(&mat)?;
    let matB = DeviceBuffer::from_slice(&mat)?;
    let matOut = DeviceBuffer::from_slice(&mato)?;
    let matConvOut = DeviceBuffer::from_slice(&convo)?;

    let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);

    let mut out_ref: Option<&DeviceBuffer<f32>> = Some(&matOut);
    let mut matA_ref: Option<&DeviceBuffer<f32>> = Some(&matA);
    let mut matB_ref: Option<&DeviceBuffer<f32>> = Some(&matB);

    let mut out_size: Option<(usize, usize)> = None;
    for layer in layers {
        if ["relu", "gelu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation";
            match load_module(function_name) {
                Ok(module) => {
                    let function_namef32 = "activationf32";
                    let kernel = module.get_function(&function_namef32)?;
                    let param = DeviceBuffer::from_slice(&[
                        (layer.input_size.0 * layer.input_size.1) as i32,
                        map_act[layer.op] as i32,
                    ])?;

                    let (_block_size, _grid_a, _grid_b) =
                        get_block_grid(layer.input_size.1, layer.input_size.0);
                    let A = match matA_ref {Some(a)=> {a}, _=> {panic!("error")}};
                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            A.as_device_ptr(),
                            param.as_device_ptr(),
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                            A.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            map_act[layer.op]
                        ));

                        result?;
                    }
                    out_ref = Some(&A);
                    out_size = Some(layer.output_size);
                }
                _ => {
                    panic!("Failed to load kernel!");
                }
            }
        } else if layer.op == "batch_matmul_legacy" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[
                        1i32,
                        layer.input_size.0 as i32,
                        layer.input_size.1 as i32,
                    ])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[
                        1i32,
                        layer.input_size.0 as i32,
                        layer.input_size.1 as i32,
                    ])?;
                    let A = match matA_ref {Some(a)=> {a}, _=> {panic!("error")}};
                    let B = match matB_ref {Some(a)=> {a}, _=> {panic!("error")}};
                    let O = match out_ref {Some(a)=> {a}, _=> {panic!("error")}};

                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            A.as_device_ptr(),
                            B.as_device_ptr(),
                            O.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                            A.as_device_ptr(),
                            B.as_device_ptr(),
                            O.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            layer.output_size.1 as u32
                        ));

                        result?;
                    }

                    matA_ref = Some(&O);
                    match layer.weight {
                        Some(w) => {
                            matB_ref = Some(w);
                        }
                        _ => {
                        }
                    };

                    out_ref = Some(&O);
                    out_size = Some(layer.output_size);
                }
                _ => {
                    panic!("\nFailed to load kernel (matmul)!");
                }
            }
        } else if layer.op == "convolution" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    let A = match matA_ref {Some(a)=> {a}, _=> {panic!("error")}};
                    let B = match matB_ref {Some(a)=> {a}, _=> {panic!("error")}};

                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[
                        layer.input_size.0 as i32,
                        layer.input_size.1 as i32,
                        1i32,
                        1i32,
                    ])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&vec![K as i32, K as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let channelInfo = DeviceBuffer::from_slice(&vec![1i32, 1i32, 1i32, 1i32])?;

                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            A.as_device_ptr(),
                            B.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr(),
                            channelInfo.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            A.as_device_ptr(),
                            B.as_device_ptr(),
                            ConvOut.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            K as u32,
                            K as u32
                        ));

                        result?;
                    }
                    matA_ref = Some(&matConvOut);
                    match layer.weight {
                        Some(w) => {
                            matB_ref = Some(w);
                        }
                        _ => {
                        }
                    };
                    out_ref = Some(&matConvOut);
                    out_size = Some(layer.output_size);
                }
                _ => {
                    panic!("\nFailed to load kernel (convolution)!");
                }
            }
        } else {
            panic!("Operation {} not supported!", layer.op);
        }
    }
    // Wait asynchronous kernels to finish.
    stream.synchronize()?;

    match out_ref {
        Some(out) => {
            let mut out_host = vec![0.0f32; out.len()];
            out.copy_to(&mut out_host)?;
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
                _ => {
                    panic!("Unable to obtain compute result!")
                }
            }
        }
        _ => {
            panic!("Unable to obtain compute result!")
        }
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}
```

### Sample of Index select kernel for candle-gcu

``` c++

#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
using namespace std;
#define TILE_SIZE AlignDown(((VDMEM_SIZE) / 16), 256)

template <typename ID_TYPENAME, typename T>
__device__ __forceinline__ void index_select_kernel(const size_t id_numel,
    ID_TYPENAME *ids, T *inp, T *out,
    const size_t left_size, const size_t dim_size, const size_t right_size) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = id_numel;
    __local__ __valigned__ ID_TYPENAME ids_buffer[4096];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    tops::mdspan l1_ids(tops::Private, ids_buffer, N);
    tops::mdspan hbm_ids(tops::Global, ids, N);
    tops::memcpy(ctx, l1_ids, hbm_ids);

    int THREAD_STEP = 1;
    int thread_step = 1;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }

    for (int i = 0; i < thread_step; i++) {
      int idx = thread_id * THREAD_STEP + i;
      if (idx < N) {
        for (unsigned int j = 0; j < left_size; ++j) {
            int _idx = ids_buffer[idx];
            tops::mdspan hbm_inp(tops::Global, inp + (j * dim_size + _idx) * right_size, right_size);
            tops::mdspan hbm_out(tops::Global, out + (idx + j * N) * right_size, right_size);
            tops::memcpy(ctx, hbm_out, hbm_inp);
        }
      }
    }
}

#define IS_OP(TYPE, ID_TYPE, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t id_numel, \
    ID_TYPE* ids, \
    TYPE *inp, \
    TYPE *out, \
    const size_t left_size, \
    const size_t dim_size, \
    const size_t right_size) \
{ \
    index_select_kernel<ID_TYPE, TYPE>(id_numel, ids, inp, out, left_size, dim_size, right_size); \
} \

IS_OP(__bf16, int64_t, is_i64_bf16)
IS_OP(__bf16, uint32_t, is_u32_bf16)
IS_OP(__bf16, uint8_t, is_u8_bf16)

IS_OP(__fp16, int64_t, is_i64_f16)
IS_OP(__fp16, uint32_t, is_u32_f16)
IS_OP(__fp16, uint8_t, is_u8_f16)

IS_OP(float, int64_t, is_i64_f32)
IS_OP(double, int64_t, is_i64_f64)
IS_OP(uint8_t, int64_t, is_i64_u8)
IS_OP(uint32_t, int64_t, is_i64_u32)
IS_OP(int64_t, int64_t, is_i64_i64)

IS_OP(float, uint32_t, is_u32_f32)
IS_OP(double, uint32_t, is_u32_f64)
IS_OP(uint8_t, uint32_t, is_u32_u8)
IS_OP(int64_t, uint32_t, is_u32_i64)
IS_OP(uint32_t, uint32_t, is_u32_u32)

IS_OP(float, uint8_t, is_u8_f32)
IS_OP(double, uint8_t, is_u8_f64)
IS_OP(uint8_t, uint8_t, is_u8_u8)
IS_OP(uint32_t, uint8_t, is_u8_u32)
IS_OP(int64_t, uint8_t, is_u8_i64)
```

### Sample of Softmax kernel for cangle-gcu

``` c++

#define TILE_SIZE 1024 * 16
template <typename T>
__device__ void softmax_kernel(T *input, T* output, 
    size_t chunks, size_t last_dim_size) {
    __local__ __valigned__ float buffer1[TILE_SIZE];
    __local__ __valigned__ float buffer2[TILE_SIZE];
    __local__ __valigned__ T bufferTmp[TILE_SIZE];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = chunks;

    int THREAD_STEP = 1;
    int thread_step = 1;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }

    //yi = exp(xi - max)/(sum(exp(xi - max))
    for (int i = 0; i < thread_step; i++) {
      int offset = thread_id * THREAD_STEP + i;
      if (offset >= N) {
        break;
      }
      tops::mdspan l1_input(tops::Private, bufferTmp, last_dim_size);
      tops::mdspan hbm_input(tops::Global, input + offset * last_dim_size, last_dim_size);

      tops::memcpy(ctx, l1_input, hbm_input);
      convert<float, T>(reinterpret_cast<float*>(buffer1), reinterpret_cast<T*>(bufferTmp), last_dim_size);
      
      atomic_reduce_max(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), last_dim_size);
      
      float max_value = buffer2[0];
      sub(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), max_value, last_dim_size);
      exp(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(buffer2), last_dim_size);
      atomic_reduce_sum(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), last_dim_size);
      float sum_exp = buffer2[0];
      tops::mdspan hbm_output(tops::Global, output + offset * last_dim_size, last_dim_size);
      div(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), sum_exp, last_dim_size);
      convert<T, float>(reinterpret_cast<T*>(bufferTmp), reinterpret_cast<float*>(buffer2), last_dim_size);
      tops::mdspan l1_output(tops::Private, bufferTmp, last_dim_size);
      tops::memcpy(ctx, hbm_output, l1_output);
    }
}

extern "C" __global__ void softmax_f16(__fp16 *input, __fp16 *output,
    size_t chunks, size_t last_dim_size) {
      softmax_kernel<__fp16>(input, output, chunks, last_dim_size);
}

extern "C" __global__ void softmax_bf16(__bf16 *input, __bf16 *output,
    size_t chunks, size_t last_dim_size) {
      softmax_kernel<__bf16>(input, output, chunks, last_dim_size);
}

extern "C" __global__ void softmax_f32(float *input, float *output,
    size_t chunks, size_t last_dim_size) {
      softmax_kernel<float>(input, output, chunks, last_dim_size);
}

```

### Sample test for softmax kernel in cangle-gcu
``` rust
fn test_softmax(dtype: DType, gcu_device: &Device) -> Result<()> {
    let shape: Shape = (1, 32, 13).into();
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;
    
    // let cpu_output = candle_nn::ops::softmax(&cpu_input, 1)?;
    // let gcu_output = candle_nn::ops::softmax(&gcu_input, 1)?;
    let shape: Shape = (1, 32 * 13).into();
    let cpu_output = candle_nn::ops::softmax_last_dim(&cpu_input)?.reshape(&shape)?;
    let gcu_output = candle_nn::ops::softmax_last_dim(&gcu_input)?.reshape(&shape)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[0],
        gcu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[0],
        abs_all <= 0.000001);

    println!("Test softmax passed!");

    Ok(())
}
```