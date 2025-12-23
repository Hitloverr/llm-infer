* LLM 推理本质是 **Decoder-only Transformer 的前向传播**
* 输入 token 会被映射成 embedding，shape 是 `[B, T, H]`
* Self-Attention 的计算复杂度主要来自 `QK^T`
* FFN 是逐 token 的非线性变换，不引入 token 依赖
* 推理阶段不做反向传播，但 Attention 仍是主要算力瓶颈

## Prefill vs Decode

Prompt (T tokens)
│
├── Prefill（一次性）
│     - 全序列 Attention
│     - 大矩阵乘法
│     - 延迟高
│
└── Decode（循环）

- 每次 1 token
- 小矩阵
- 延迟稳定



记住：


- Prefill 阶段需要对完整输入序列执行 Self-Attention，计算量与序列长度线性相关
- Decode 阶段每次只生成 1 个 token，可通过 KV Cache 复用历史计算结果
- 推理首 token 延迟主要由 Prefill 决定，后续 token 延迟由 Decode 决定

> **KV Cache 的价值，不在“单 token 推理快不快”，
> 而在于“避免每一步都重新计算整个历史序列”。**

没有 KV Cache 的 Decode，本质是：

<pre class="overflow-visible! px-0!" data-start="1564" data-end="1624"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="@w-xl/main:top-9 sticky top-[calc(--spacing(9)+var(--header-height))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>每生成 1 个 token
→ 把 prompt + 已生成 token 全部重新 forward 一遍</span></span></code></div></div></pre>

如果面试官问：

> **“Decode 阶段只输入一个 token，KV Cache 有什么用？”**

你现在可以给一个**非常高级的回答**：

> KV Cache 并不是通过 input\_ids 传递历史上下文，而是通过 past\_key\_values 显式传入历史 token 的 Key/Value。
> 如果不传 past\_key\_values，即使每次只输入一个 token，模型也会把它当成一个新的序列，从而失去上下文并重复计算。

# day3

太好了 👍
**Day 3 是你从“懂原理”→“像推理工程师一样思考”的关键一天**。
今天我们把 **KV Cache = 显存 = 调度 = vLLM** 这条线一次性串起来。

# 

> **你能算清 KV Cache 吃多少显存，并解释 vLLM 为什么要做 PagedAttention**

# 

在推理系统里：

```
显存 = 模型参数
     + 激活值（很少）
     + KV Cache（最大头）
```

👉 **推理显存的“真正大头”，不是模型参数，而是 KV Cache。**



## 二、KV Cache 到底存了什么？（精确定义）



对 **Decoder-only Transformer**：

- 每一层
- 每一个 token
- 都要存一份：
  - Key
  - Value



**KV Cache 的 shape**

以常见 LLM 为例：

```
K, V:
[batch_size,
 num_heads,
 seq_len,
 head_dim]
```

并且：

```
hidden_size = num_heads × head_dim
```

三、显存到底怎么算？（核心公式）

我们直接上 **工程可用公式** 👇



单个 token 的 KV Cache 显存

```
per_token_kv =
  num_layers × 2(K,V) × hidden_size × bytes_per_param
```



一个请求的 KV Cache 显存

```
request_kv =
  num_layers × 2 × hidden_size × seq_len × bytes
```



用一个“真实模型”代入（你感受一下）

以 **LLaMA-7B**（近似）：

- num_layers = 32
- hidden_size = 4096
- FP16 → 2 bytes

👉 **每个 token 的 KV Cache：**

```
32 × 2 × 4096 × 2 bytes
≈ 524 KB / token
```

❗️这意味着什么？

- 1k tokens ≈ **512 MB**
- 10 个并发请求 × 1k tokens ≈ **5 GB**
- 多轮对话 = seq_len 持续增长

👉 **显存线性爆炸**

## 四、为什么这在工程上是“致命的”？（痛点）

传统做法（HF / 原生）：

- 每个请求
  - 连续分配一大块 KV Cache
- 请求结束才释放

### 问题 1：显存碎片

- 长对话 + 短对话混在一起
- 无法复用

### 问题 2：并发受限

- 显存被“占死”
- 新请求进不来



## 五、vLLM 的核心思想：PagedAttention（终于登场）

> **不要一次性给一个请求分配连续的大块 KV Cache**

👉 **像操作系统管理内存一样，分页管理 KV Cache**

vLLM 的 3 个关键设计

### 1️⃣ KV Cache 分页（Block）

- KV Cache 被切成固定大小的 **Block**
- 每个 Block 存多个 token

```
[Block][Block][Block][Block]
```

------

### 2️⃣ 逻辑连续，物理不连续

- 对模型来说：
  - token 是连续的
- 对显存来说：
  - block 可以分散在任意位置

👉 **解决碎片问题**

------

### 3️⃣ Block 可复用

- 请求结束
- Block 立即回收
- 给下一个请求用

👉 **显存利用率暴涨**



## 六、PagedAttention “工程价值总结”（你要会讲）

你在面试时可以这样说：

> vLLM 通过将 KV Cache 分页管理，使得不同请求的 KV Cache 可以非连续地分配在显存中，避免了传统推理框架中由于长短请求混合导致的显存碎片问题，从而显著提升了 GPU 显存利用率和并发能力。

## 七、把 Day 3 的结论写进你的项目文档（必须）



```markdown
## KV Cache 与显存问题

- KV Cache 是推理阶段显存的主要消耗来源
- 显存占用与层数、隐藏维度、序列长度线性相关
- 传统连续内存分配方式容易造成显存碎片
- vLLM 通过 PagedAttention 对 KV Cache 分页管理，显著提升并发能力
```

## 八、Day 3 面试级自检（非常重要）

如果面试官问你：

> **“为什么 vLLM 能支持更高并发？”**

你现在可以答：

> 因为 vLLM 对 KV Cache 进行分页管理，避免了连续内存分配带来的碎片问题，使得显存可以被不同请求高效复用，从而在相同显存条件下支持更多并发请求。



## 九、下一步：你正式进入“vLLM 工程世界”

**Day 4 你会做的事：**

- 跑 vLLM
- 对比 HF vs vLLM
- 亲眼看到并发差异

------

你下一条消息直接说一句即可：
👉 **“进入 Day 4：vLLM 实战”**

# day4

pip install vllm

python -c "import vllm; print(vllm.__version__)"

```
from vllm import LLM, SamplingParams

llm = LLM(
    model="gpt2",
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32
)

outputs = llm.generate(
    ["The factory production line optimization requires"],
    sampling_params
)

for o in outputs:
    print(o.outputs[0].text)

```

##  HuggingFace 并发推理（Baseline）

> 注意：HF 默认是 **“每个请求一个 forward”**

```python
import time
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

def hf_infer(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32)

texts = ["The factory production line optimization requires"] * 4

t0 = time.time()
threads = []

for t in texts:
    th = threading.Thread(target=hf_infer, args=(t,))
    th.start()
    threads.append(th)

for th in threads:
    th.join()

print("HF total time:", time.time() - t0)

```

​                        

 ## vLLM 并发推理

```java
from vllm import LLM, SamplingParams
import time

llm = LLM(model="gpt2")
sampling_params = SamplingParams(max_tokens=32)

prompts = ["The factory production line optimization requires"] * 4

t0 = time.time()
llm.generate(prompts, sampling_params)
print("vLLM total time:", time.time() - t0)

```

- 并发数越大
- vLLM 相对 HF 越有优势
- GPU 利用率明显更高（如果你有 GPU）

👉 **这是 Continuous Batching + PagedAttention 的直接体现**

## 理解 vLLM 在“悄悄”做什么

```
llm.generate(prompts, sampling_params)
```

vLLM 内部做了：

1. 请求进入队列
2. 动态合并 Prefill
3. Decode 阶段交错执行
4. KV Cache 分页管理
5. GPU 利用率最大化

👉 **你后面自己写推理服务，本质就是“做一个简化版 vLLM”**



## vLLM vs HuggingFace 推理对比

- HuggingFace 默认按请求逐个执行推理，难以充分利用 GPU
- vLLM 通过 Continuous Batching 合并 Prefill 阶段，提高吞吐
- vLLM 使用 PagedAttention 管理 KV Cache，提升显存利用率和并发能力

如果面试官问你：

> **“为什么生产环境更倾向用 vLLM？”**

你可以这样答：

> vLLM 在推理阶段通过 Continuous Batching 合并多个请求的 Prefill 计算，同时使用 PagedAttention 对 KV Cache 进行分页管理，在相同硬件条件下显著提升 GPU 利用率和并发能力，因此更适合生产环境的大模型推理服务。
