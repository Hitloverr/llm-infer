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
