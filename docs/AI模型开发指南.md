# AI 模型从零开发完整指南

> 版本：v1.0  
> 日期：2026-11-19  
> 目标：使用 Rust 语言从零构建一款 AI 模型（含底层引擎、框架层、模型架构三层完整实现）

---

# 目录

- [第一章 核心概念分层总览](#第一章-核心概念分层总览)
- [第二章 AI模型底层（数学与计算基础层）](#第二章-ai模型底层数学与计算基础层)
  - [2.1 张量：AI 的原子数据结构](#21-张量ai-的原子数据结构)
  - [2.2 矩阵运算：一切计算的基石](#22-矩阵运算一切计算的基石)
  - [2.3 激活函数：引入非线性的关键](#23-激活函数引入非线性的关键)
  - [2.4 损失函数：衡量模型好坏的标尺](#24-损失函数衡量模型好坏的标尺)
  - [2.5 梯度下降与优化算法](#25-梯度下降与优化算法)
  - [2.6 自动微分：反向传播的工程实现](#26-自动微分反向传播的工程实现)
- [第三章 AI模型框架（工程实现层）](#第三章-ai模型框架工程实现层)
  - [3.1 张量库设计](#31-张量库设计)
  - [3.2 自动微分引擎](#32-自动微分引擎)
  - [3.3 神经网络模块系统（nn）](#33-神经网络模块系统nn)
  - [3.4 注意力机制与 Transformer 架构](#34-注意力机制与-transformer-架构)
  - [3.5 训练引擎与数据流水线](#35-训练引擎与数据流水线)
  - [3.6 模型持久化与序列化](#36-模型持久化与序列化)
- [第四章 AI模型执行原理](#第四章-ai模型执行原理)
  - [4.1 前向传播（Forward Pass）](#41-前向传播forward-pass)
  - [4.2 损失计算](#42-损失计算)
  - [4.3 反向传播（Backward Pass）](#43-反向传播backward-pass)
  - [4.4 参数更新](#44-参数更新)
  - [4.5 推理阶段（Inference）](#45-推理阶段inference)
- [第五章 Rust 生态 AI 框架现状](#第五章-rust-生态-ai-框架现状)
- [第六章 从零开发路线图](#第六章-从零开发路线图)

---

# 第一章 核心概念分层总览

在正式开始之前，必须明确区分三个常被混淆的概念层级。这是整个文档的核心逻辑框架。

## 1.1 三层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                   AI 模型（Model）                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  业务层：Transformer / CNN / RNN / MLP                     │  │
│  │  由框架提供的模块组合而成的具体网络结构                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↑ 依赖                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              AI 框架（Framework）                           │  │
│  │  工程层：张量库 / 自动微分 / 神经网络模块 / 优化器 / 训练循环 │  │
│  │  为模型定义和训练提供完整 API 和运行时                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↑ 依赖                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              AI 底层（Foundation）                          │  │
│  │  数学层：矩阵运算 / 激活函数 / 损失函数 / 链式法则           │  │
│  │  一切 AI 计算的数学原理和原子操作                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 1.2 三层职责边界

| 层级 | 回答的问题 | 核心内容 | 实现产物 |
|------|-----------|---------|---------|
| **底层 Foundation** | "AI 在数学上是什么？" | 张量定义、矩阵乘法、激活函数公式、损失函数推导、链式法则 | `tensor`、`ops` crate |
| **框架 Framework** | "如何高效实现 AI 计算？" | 内存管理、设备抽象、自动微分引擎、模块系统、优化器 | `autograd`、`nn`、`optim` crate |
| **模型 Model** | "具体网络结构是什么？" | Transformer、MLP、CNN 等架构组合 | `model` crate |

## 1.3 为什么要严格区分

- **底层 ≠ 框架**：底层是纯数学，框架是工程。你可以自己手写一个矩阵乘法（底层），但要让它支持自动微分、GPU 加速、动态形状，那就是框架的工作了。
- **框架 ≠ 模型**：PyTorch 是框架，GPT-3 是模型。Burn 是框架，用 Burn 写的 Transformer 网络是模型。
- **开发顺序**：自底向上 — 先有底层运算，才能构建框架；先有框架，才能定义模型。

---

# 第二章 AI模型底层（数学与计算基础层）

> 本章回答 **"AI 在数学上到底是什么"** 这个根本问题。所有现代 AI 框架的核心不过是把这些数学运算高效地实现在硬件上。

## 2.1 张量：AI 的原子数据结构

### 2.1.1 什么是张量

张量（Tensor）是标量、向量、矩阵到更高维度数组的统一抽象：

| 维度 | 名称 | 数学表示 | Rust 类型类比 | 实际例子 |
|------|------|---------|--------------|---------|
| 0 维 | 标量 Scalar | $a \in \mathbb{R}$ | `f32` / `f64` | 单个损失值、学习率 |
| 1 维 | 向量 Vector | $\mathbf{x} \in \mathbb{R}^n$ | `Vec<f32>` | 词嵌入、特征向量 |
| 2 维 | 矩阵 Matrix | $W \in \mathbb{R}^{m \times n}$ | `Vec<Vec<f32>>` | 权重矩阵、图片灰度 |
| 3 维 | 三阶张量 | $T \in \mathbb{R}^{d_1 \times d_2 \times d_3}$ | — | RGB 图片（高×宽×通道） |
| 4 维 | 四阶张量 | $T \in \mathbb{R}^{b \times c \times h \times w}$ | — | 批量图片（批次×通道×高×宽） |

### 2.1.2 张量的关键属性

```
张量 = 数据缓冲区 + 形状(shape) + 数据类型(dtype) + 设备(device)

数据缓冲区: 连续内存块，存储实际数值 [0.1, 0.5, -0.3, ...]
形状 shape:  每个维度的大小 (3, 224, 224) 表示3通道224x224图片
数据类型:    f32(默认) / f64(高精度) / f16(半精度) / bf16(脑浮点) / i64(索引)
设备 device: CPU / CUDA GPU / Metal GPU / WebGPU
步长 stride: 每个维度相邻元素在内存中的间距，决定内存布局
```

### 2.1.3 内存布局：行优先 vs 列优先

```
矩阵 W[2,3] 的内存存储：

行优先（C-contiguous）：[[a, b, c], [d, e, f]]  →  [a, b, c, d, e, f]
  第 i 行第 j 列的偏移 = i * 3 + j

列优先（F-contiguous）：[[a, b, c], [d, e, f]]  →  [a, d, b, e, c, f]
  第 i 行第 j 列的偏移 = i + j * 2

BLAS（基础线性代数子程序）默认列优先
NumPy/PyTorch 默认行优先（但底层调用 BLAS 时会转换）
```

### 2.1.4 张量的广播机制（Broadcasting）

广播是让不同形状的张量进行逐元素运算的规则：

```
规则：从最右维向左，逐维比较
  - 两维相等 → 兼容
  - 其中一维为 1 → 该维可以"复制"扩展
  - 否则 → 报错

示例：
(3, 4) + (4,)   → (3, 4) ✓   向量(4,)被复制到3行
(3, 4) + (1, 4) → (3, 4) ✓   同上行优先
(3, 4) + (3,)   → ✗          最后一维 4 ≠ 3，且都不是1
(2, 1, 3) * (4, 3) → (2, 4, 3) ✓
```

## 2.2 矩阵运算：一切计算的基石

### 2.2.1 矩阵乘法（MatMul）

矩阵乘法是神经网络中最重要的运算：

$$\mathbf{Y} = \mathbf{X} \cdot \mathbf{W} + \mathbf{b}$$

其中：
- $\mathbf{X} \in \mathbb{R}^{batch \times input\_dim}$ 输入
- $\mathbf{W} \in \mathbb{R}^{input\_dim \times output\_dim}$ 权重
- $\mathbf{b} \in \mathbb{R}^{output\_dim}$ 偏置
- $\mathbf{Y} \in \mathbb{R}^{batch \times output\_dim}$ 输出

**直观理解**：每个输出神经元是所有输入的加权和。

**复杂度**：O(batch × input_dim × output_dim)

**分块乘法（Tiling）**：为了让矩阵乘法高效利用 CPU/GPU 缓存，实际实现会把大矩阵切成小块来计算。这就是 cuBLAS、MKL、BLIS 等高性能库做的事情。

### 2.2.2 矩阵乘法的并行化策略

```
CPU 层：
  - OpenMP / Rayon：多线程并行处理不同的行列块
  - SIMD（AVX-512 / NEON）：向量化乘加运算（FMA 指令）
  - 分块（Tiling）：匹配 L1/L2/L3 缓存大小

GPU 层：
  - CUDA Warp（32线程为一组）：单指令多线程（SIMT）
  - Shared Memory：块内共享内存减少全局内存访问
  - Tensor Core（HMMA）：硬件级矩阵乘加单元
  - 线程映射：每个线程计算输出矩阵的一个元素
```

### 2.2.3 其他重要矩阵运算

| 运算 | 数学表达 | 用途 |
|------|---------|------|
| 转置 | $W^T$ | 反向传播时的梯度传递 |
| 逐元素乘 | $A \odot B$ | 门控机制、Element-wise 模块 |
| 求和 | $\sum_i x_i$ | Reduction、Loss 计算 |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | 分类层输出概率分布 |
| LayerNorm | $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | Transformer 中稳定训练 |

## 2.3 激活函数：引入非线性的关键

如果没有激活函数，无论多少层矩阵乘法都等价于一个矩阵乘法（线性变换的复合仍是线性变换）。激活函数引入非线性，使网络能学习复杂模式。

### 2.3.1 常见激活函数对比

```
┌──────────┬──────────────────────────────┬──────────────────────┬──────────────────────────┐
│ 函数名   │ 公式                          │ 导数                  │ 适用场景                  │
├──────────┼──────────────────────────────┼──────────────────────┼──────────────────────────┤
│ ReLU     │ f(x) = max(0, x)              │ x > 0 → 1, else 0    │ 隐藏层首选，计算最快       │
│ LeakyReLU│ f(x) = x if x>0 else α*x      │ x>0→1, else→α        │ 解决 ReLU 的"神经元死亡"  │
│ Sigmoid  │ f(x) = 1/(1+e^(-x))            │ f(x)(1-f(x))          │ 二分类输出层              │
│ Tanh     │ f(x) = (e^x - e^(-x))/(e^x+e^(-x))│ 1 - f(x)^2         │ RNN 隐藏层（已过时）       │
│ GELU     │ f(x) = x·Φ(x) ≈ 0.5x(1+erf(x/√2))│ 复杂，见下方          │ BERT/GPT 系列默认         │
│ Swish    │ f(x) = x·σ(βx)               │ σ(βx) + βxσ(βx)(1-σ) │ 搜索发现，常优于 ReLU     │
│ SiLU     │ f(x) = x·σ(x)                 │ σ(x) + xσ(x)(1-σ)    │ PyTorch 默认，Swish(β=1) │
└──────────┴──────────────────────────────┴──────────────────────┴──────────────────────────┘
```

### 2.3.2 GELU 的精确实现

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准正态分布的 CDF。实际实现中会用近似公式：

$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### 2.3.3 激活函数的导数：反向传播必需

每一个激活函数都必须有导数公式，因为反向传播时需要用链式法则：

```
前向：h = σ(a)         # 激活
反向：dL/da = dL/dh · σ'(a)    # 链式法则传递梯度
```

## 2.4 损失函数：衡量模型好坏的标尺

损失函数（Loss Function）计算模型预测值与真实值之间的差距。训练的目标就是最小化这个损失。

### 2.4.1 回归任务

**均方误差 MSE**：

$$\text{MSE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

导数：$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$

**Huber Loss**（对异常值更鲁棒）：

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

### 2.4.2 分类任务

**二元交叉熵 BCE**：

$$\text{BCE}(y, p) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$

导数：$\frac{\partial \text{BCE}}{\partial p_i} = \frac{p_i - y_i}{p_i(1 - p_i)}$

**多分类交叉熵 Cross-Entropy**：

$$\text{CE}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

其中 $y$ 是 one-hot 向量，$\hat{y}$ 是 softmax 输出。

**关键洞察**：Softmax + CrossEntropy 的组合有一个简洁的梯度：

$$\frac{\partial \text{CE}}{\partial z_j} = \text{softmax}(z)_j - y_j$$

其中 $z$ 是 softmax 的输入（logits）。这意味着实现时可以把 Softmax 和 CrossEntropy 合成一个优化的 kernel，避免数值稳定性问题。

### 2.4.3 生成任务

**负对数似然 NLL**（语言模型）：

$$L = -\sum_{t=1}^{T} \log P(w_t | w_1, ..., w_{t-1})$$

这就是 GPT 模型训练时的标准损失。

## 2.5 梯度下降与优化算法

### 2.5.1 梯度下降的基本原理

**梯度**：损失函数对所有可训练参数的偏导数组成的向量

$$\nabla_\theta L = \left(\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, ..., \frac{\partial L}{\partial \theta_n}\right)$$

梯度指向损失函数增长最快的方向。**梯度下降**就是反方向走一步：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)$$

其中 $\eta$ 是**学习率**（learning rate），控制每步走多远。

### 2.5.2 三种梯度下降变体

| 类型 | 训练数据 | 特点 |
|------|---------|------|
| 批量梯度下降 BGD | 全部数据 | 准确但慢，不支持在线学习 |
| 随机梯度下降 SGD | 单个样本 | 快但噪声大，容易震荡 |
| 小批量梯度下降 Mini-batch SGD | 32/64/128 样本 | 最常用，平衡速度和稳定 |

### 2.5.3 现代优化器对比

**SGD + Momentum**：

$$v_t = \beta \cdot v_{t-1} + \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

动量让更新方向更稳定，像下山时带着惯性。

**Adam（Adaptive Moment Estimation）**：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{（一阶矩：梯度的指数平均）}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{（二阶矩：梯度平方的指数平均）}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad \text{（偏差修正）}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

默认参数：$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$

**AdamW（带权重衰减的 Adam）**：

直接在参数更新时加入 L2 惩罚项：

$$\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda \cdot \theta_t$$

$\lambda$ 是权重衰减系数（通常 0.01）。**AdamW ≠ Adam + L2正则化**，两者有细微差异。

### 2.5.4 学习率调度策略

```
StepLR:       每 N 个 epoch 学习率乘以衰减因子
CosineAnneal: 学习率从最大值余弦下降到最小值
Warmup:       前 N 步线性增加学习率到目标值，然后衰减
ReduceLROnPlateau: 损失连续 N 步不降则降低学习率

Transformer 标准配置：Warmup(4000步) + CosineAnneal
```

## 2.6 自动微分：反向传播的工程实现

自动微分（Automatic Differentiation, autograd）是深度学习框架的灵魂。没有它，你需要手动为每个函数推导梯度——这在现代网络中不现实。

### 2.6.1 两种自动微分模式

```
┌──────────────────────────────────────────────────────────┐
│  前向模式 Forward-Mode Autograd                          │
│                                                          │
│  伴随每个变量同时追踪其对输入的导数                         │
│  z = f(x) 在计算 z 的同时算出 dz/dx                        │
│                                                          │
│  优点：实现简单，内存固定                                  │
│  缺点：需要对每个输入单独跑一遍，n个输入要n遍               │
│  适用：输入维度远小于输出维度的场景                        │
├──────────────────────────────────────────────────────────┤
│  反向模式 Reverse-Mode Autograd                          │
│                                                          │
│  前向传播：记录计算图（DAG），存储所有中间值               │
│  反向传播：从输出开始，按拓扑序反向遍历 DAG                 │
│            应用链式法则，一次性算出所有输入的梯度            │
│                                                          │
│  优点：无论输出维度多少，只需一次反向传播                   │
│  缺点：前向传播时需要保存所有中间激活值（显存消耗大）       │
│  适用：神经网络（通常只有1个损失输出，但有百万级参数输入） │
└──────────────────────────────────────────────────────────┘
```

### 2.6.2 反向模式自动微分的核心算法

**前向阶段**：构建计算图

```python
# 伪代码
def forward(x, W, b):
    # 每个运算节点记录：输入、运算类型、以及 backward 函数
    a = MatMul(x, W)       # 记录: {inputs:[x,W], op:"matmul"}
    c = Add(a, b)          # 记录: {inputs:[a,b], op:"add"}
    h = GELU(c)            # 记录: {inputs:[c], op:"gelu"}
    y = Linear(h, W2, b2)  # 记录: {inputs:[h,W2,b2], op:"linear"}
    return y
```

**反向阶段**：拓扑排序 + 链式法则

```python
def backward(loss):
    # 1. 从 loss 开始，梯度设为 1（dL/dL = 1）
    grads[loss] = Tensor.ones_like(loss)
    
    # 2. 反向拓扑遍历计算图
    for node in reversed(topological_sort(compute_graph)):
        grad_output = grads[node.output]
        
        # 3. 根据运算类型，调用对应的 backward 函数
        if node.op == "matmul":
            x, W = node.inputs
            # y = x @ W，所以：
            # dL/dx = grad_output @ W^T
            # dL/dW = x^T @ grad_output
            grads[x] = grad_output @ W.transpose()
            grads[W] = x.transpose() @ grad_output
            
        elif node.op == "gelu":
            c = node.inputs[0]
            grads[c] = grad_output * gelu_derivative(c)
    
    # 4. 返回所有需要梯度的输入的梯度
    return {param: grads[param] for param in leaf_nodes}
```

### 2.6.3 计算图的表示方式

**动态图（Define-by-Run，PyTorch/Candle/Burn）**：
- 每次前向传播时动态构建新的计算图
- 优点：灵活，支持条件分支、循环、任意 Python 控制流
- 缺点：无法提前优化（算子融合、常量折叠）

**静态图（Define-and-Run，TensorFlow 1.x）**：
- 先定义完整的计算图，再编译运行
- 优点：可优化、可导出
- 缺点：不灵活

**Burn 的创新**：动态图执行 + 静态图优化。通过 Rust 的类型系统，Burn 在编译期推导出部分图结构，运行时动态部分正常执行，同时对静态部分做内核融合等优化。

### 2.6.4 梯度检查（Gradient Checking）

手动实现 autograd 时必须验证正确性。方法：

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

用微小扰动 $\epsilon = 10^{-5}$ 做有限差分近似，与 autograd 计算的梯度对比。两者相对误差应小于 $10^{-5}$。

### 2.6.5 梯度累积与截断

**梯度累积**：显存不足以跑大 batch 时，多个小 batch 不立即更新参数，而是累积梯度后再统一更新。

```
accumulate_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    if (i + 1) % accumulate_steps == 0:
        optimizer.step()       # 4个小batch的平均梯度更新一次
        optimizer.zero_grad()
```

**梯度裁剪**：防止梯度爆炸（RNN/Transformer 中常见）

```
# 全局范数裁剪
grad_norm = sqrt(sum(g^2 for all grads))
if grad_norm > max_norm:
    scale = max_norm / grad_norm
    for grad in all_grads:
        grad *= scale
```

**梯度截断**：解决 RNN 的梯度消失/爆炸问题（BPTT 中的 Truncated BPTT）

---

# 第三章 AI模型框架（工程实现层）

> 本章回答 **"如何高效实现 AI 计算"** 这个工程问题。底层数学是不变的，但框架决定了性能、易用性、可移植性。

## 3.1 张量库设计

### 3.1.1 核心数据结构

```rust
// Rust 风格的张量核心结构
pub struct Tensor<T, D: Device> {
    /// 形状：每个维度的大小
    shape: Vec<usize>,
    /// 步长：每个维度相邻元素在内存中跨越的长度
    strides: Vec<usize>,
    /// 底层数据缓冲区（设备相关：CPU上是Vec<T>，GPU上是DeviceBuffer<T>）
    data: Storage<T, D>,
    /// 引用计数（Rc/Arc），支持张量共享底层数据（如 view / slice）
    ref_count: Arc<AtomicUsize>,
    /// 自动微分状态：是否需要追踪梯度、计算图节点ID
    autograd_state: Option<AutogradState>,
}
```

### 3.1.2 设备抽象层

```rust
// 统一的设备抽象，实现 Backend trait
pub trait Backend: Sized {
    type Device;
    type Storage<T>: StorageOps<T, Self>;
    
    // 张量创建
    fn tensor_from_data<T>(data: Vec<T>, shape: &[usize]) -> Tensor<T, Self>;
    fn zeros<T>(shape: &[usize]) -> Tensor<T, Self>;
    fn ones<T>(shape: &[usize]) -> Tensor<T, Self>;
    fn randn<T>(shape: &[usize]) -> Tensor<T, Self>;
    
    // 张量运算
    fn matmul(a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Tensor<T, Self>;
    fn add(a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Tensor<T, Self>;
    fn gelu(a: &Tensor<T, Self>) -> Tensor<T, Self>;
    fn softmax(a: &Tensor<T, Self>, dim: usize) -> Tensor<T, Self>;
    
    // 其他上百个算子...
}

// 具体后端实现
pub struct Cpu;        // CPU 后端：纯 Rust / SIMD / BLAS
pub struct Cuda;       // NVIDIA GPU 后端：CUDA / cuBLAS
pub struct Metal;      // Apple GPU 后端：Metal Performance Shaders
pub struct Wgpu;       // 跨平台 GPU：WebGPU (Vulkan/Metal/DX12)
pub struct NdArray;    // 基于 ndarray crate 的 CPU 后端
pub struct LibTorch;   // 基于 tch crate 的 PyTorch C++ 后端
```

### 3.1.3 内存管理策略

```
张量内存管理的核心问题：

1. 谁拥有内存？（Ownership）
   → Rust 的所有权系统天然解决：张量被 drop 时自动释放底层缓冲
   → 但 view/slice 需要共享底层内存，所以用 Arc<Mutex<Storage>> 或类似机制

2. 什么时候分配？（Allocation）
   → 惰性分配：有些操作（如 reshape）只是改元信息，不分配新内存
   → 内存池：GPU 上显存分配很慢，框架会维护内存池

3. 设备间如何传输？（Transfer）
   → CPU→GPU: cudaMemcpy(HostToDevice) / Metal Buffer.copy
   → GPU→CPU: 反向传输，通常只在 Debug 或推理结束时
   → 多 GPU: NCCL / RCCL / 自定义 AllReduce

4. 如何避免不必要的拷贝？（View vs Copy）
   ┌──────────────────────────────────────────────┐
   │ reshape  → 只改 shape，不改 data（零拷贝）   │
   │ transpose → 只改 strides（零拷贝）             │
   │ slice    → 只改 shape + data 起始指针（零拷贝）│
   │ permute  → 只改 strides（零拷贝）              │
   │                                          BUT  │
   │ contiguous() → 如果 strides 不是连续的，     │
   │                必须做真实的数据拷贝重新排列     │
   └──────────────────────────────────────────────┘
```

## 3.2 自动微分引擎

### 3.2.1 两种架构对比

```
┌────────────────────────────────────────────────────────────┐
│  架构 A：张量持有计算图节点（PyTorch / Candle / Burn）      │
│                                                            │
│  Tensor = 数据 + Option<Node>                              │
│                                                            │
│  Tensor.node = Some(Node {                                  │
│      inputs: Vec<Node>,                                    │
│      backward_fn: Box<dyn Fn(Vec<Tensor>) -> Vec<Tensor>>,│
│  })                                                        │
│                                                            │
│  优点：张量即计算图节点，自然直观                             │
│  缺点：需要 Arc<Mutex<>> 管理 Node 的生命周期               │
├────────────────────────────────────────────────────────────┤
│  架构 B：独立的计算图管理器（tinygrad / volta）              │
│                                                            │
│  Graph = HashMap<NodeId, Node>                             │
│  Tensor = 数据 + NodeId                                    │
│                                                            │
│  优点：Graph 可以独立序列化、可视化、做全局优化              │
│  缺点：Tensor 不再自包含，图的生命周期需要额外管理           │
└────────────────────────────────────────────────────────────┘
```

### 3.2.2 计算图节点定义

```rust
// 每个运算对应一个节点
pub struct Node {
    /// 运算类型枚举
    op: Op,
    /// 输入张量的 NodeId 列表
    inputs: Vec<NodeId>,
    /// 输入张量（前向传播后需要保留，反向传播要用）
    input_tensors: Vec<Arc<Tensor>>,
    /// 输出张量
    output: Arc<Tensor>,
}

/// 运算类型枚举（每种运算都有前向和反向实现）
pub enum Op {
    // 逐元素运算
    Add, Sub, Mul, Div,
    ReLU, Sigmoid, Tanh, GELU, Swish, Silu,
    Exp, Log, Sqrt, Pow,
    
    // 规约运算
    Sum { dim: Option<usize> },
    Mean { dim: Option<usize> },
    Max { dim: Option<usize> },
    
    // 矩阵运算
    MatMul { transpose_a: bool, transpose_b: bool },
    BatchMatMul { transpose_a: bool, transpose_b: bool },
    
    // 形状运算
    Reshape,
    Transpose { dim1: usize, dim2: usize },
    Permute { dims: Vec<usize> },
    
    // 注意力相关
    Softmax { dim: usize },
    LayerNorm,
    
    // 损失函数（末端节点）
    CrossEntropyLoss,
    MSELoss,
    BCELoss,
    
    // 叶子节点（模型参数）
    Leaf { is_parameter: bool },
}
```

### 3.2.3 backward 函数的自动注册

每个 Op 都需要实现 backward 函数。关键设计是如何让前向传播时自动"挂接"反向逻辑：

```rust
impl Tensor {
    // 张量的加法运算
    fn add(&self, other: &Tensor) -> Tensor {
        // 1. 执行实际的加法（底层）
        let result = self.backend.add(&self.data, &other.data);
        
        // 2. 如果开启了 autograd，记录计算图节点
        if self.requires_grad() || other.requires_grad() {
            let node_id = Graph::current().add_node(Node {
                op: Op::Add,
                inputs: vec![self.node_id(), other.node_id()],
                input_tensors: vec![self.clone(), other.clone()],
                output: result.clone(),
            });
            result.set_node_id(node_id);
        }
        
        result
    }
    
    // 反向传播入口
    fn backward(&self) {
        // 1. 拓扑排序计算图（从叶子到根的反向顺序）
        let sorted_nodes = Graph::current().topo_sort_reverse(self.node_id());
        
        // 2. 根节点的梯度设为全 1（dL/dL = 1）
        let mut grads: HashMap<NodeId, Tensor> = HashMap::new();
        grads.insert(self.node_id(), Tensor::ones_like(self));
        
        // 3. 按拓扑序反向遍历
        for node_id in sorted_nodes {
            let node = Graph::current().get_node(node_id);
            let grad_output = grads.get(&node_id).unwrap();
            
            // 4. 根据 Op 类型调用对应的 backward 函数
            let input_grads = node.op.backward(&node.input_tensors, grad_output);
            
            // 5. 将梯度分配给各输入节点
            for (input_id, input_grad) in node.inputs.iter().zip(input_grads) {
                grads.insert(*input_id, input_grad);
            }
        }
        
        // 6. 把叶子节点的梯度写入 Tensor.grad 字段
        for (leaf_id, grad) in grads {
            let leaf = Graph::current().get_node(leaf_id);
            if leaf.op == Op::Leaf {
                Graph::current().get_tensor(leaf_id).set_grad(grad);
            }
        }
    }
}
```

### 3.2.4 梯度累积的内存优化

反向模式需要在前向时保存所有中间张量——这是训练时显存爆炸的主要原因。优化手段：

```
1. 检查点（Gradient Checkpointing）
   不保存全部中间张量，只保存部分关键节点
   反向传播到达没保存的节点时，重新跑前向计算
   时间换空间：训练慢 30-40%，但显存可减少 50%+

2. 算子融合（Kernel Fusion）
   多个逐元素运算融合成一个 CUDA kernel
   避免中间张量的显存读写
   例如：x -> gelu -> add -> layernorm 四个融合成一个 kernel

3. 释放不再需要的中间张量
   反向传播过某个节点后，该节点的 input_tensors 可以释放
   需要用 drop / 手动 Arc::get_mut 来释放

4. BF16 / FP16 混合精度
   大部分算子用低精度（省显存、更快）
   关键累积操作（如 Softmax 分母）用 FP32（保持精度）
   自动损失缩放（Loss Scaling）防止梯度下溢
```

## 3.3 神经网络模块系统（nn）

### 3.3.1 Module trait 设计

```rust
// 所有神经网络模块都实现这个 trait
pub trait Module<B: Backend>: Send + Sync {
    /// 前向传播
    fn forward(&self, input: Tensor<B::Device, f32>) -> Tensor<B::Device, f32>;
    
    /// 返回所有可训练参数
    fn parameters(&self) -> Vec<Tensor<B::Device, f32>>;
    
    /// 设置为训练模式（影响 Dropout / BatchNorm 行为）
    fn train(&mut self);
    
    /// 设置为推理模式
    fn eval(&mut self);
}
```

### 3.3.2 Linear（全连接层）

```rust
pub struct Linear<B: Backend> {
    weight: Tensor<B::Device, f32>,  // [in_features, out_features]
    bias: Option<Tensor<B::Device, f32>>,  // [out_features]
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> Linear<B> {
    // 构造函数：自动初始化参数
    pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        // 权重初始化：Xavier/Glorot 或 Kaiming/He
        // He 初始化：ReLU 系列首选
        let std = (2.0 / in_features as f32).sqrt();
        let weight = Tensor::<B>::randn([in_features, out_features], device) * std;
        
        // Xavier 初始化：Sigmoid/Tanh 首选
        // let std = (2.0 / (in_features + out_features) as f32).sqrt();
        
        let bias = Some(Tensor::<B>::zeros([out_features], device));
        
        Self { weight, bias, in_features, out_features }
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    fn forward(&self, x: Tensor<B::Device, f32>) -> Tensor<B::Device, f32> {
        // y = x @ W + b
        let y = Tensor::<B>::matmul(&x, &self.weight);
        match &self.bias {
            Some(b) => y + b,
            None => y,
        }
    }
    
    fn parameters(&self) -> Vec<Tensor<B::Device, f32>> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias { params.push(b.clone()); }
        params
    }
}
```

### 3.3.3 LayerNorm

```rust
pub struct LayerNorm<B: Backend> {
    gamma: Tensor<B::Device, f32>,  // 缩放参数，初始化为 1
    beta: Tensor<B::Device, f32>,   // 偏移参数，初始化为 0
    eps: f32,                        // 数值稳定项，通常 1e-5
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(normalized_shape: usize, eps: f32, device: &B::Device) -> Self {
        Self {
            gamma: Tensor::<B>::ones([normalized_shape], device),
            beta: Tensor::<B>::zeros([normalized_shape], device),
            eps,
        }
    }
}

impl<B: Backend> Module<B> for LayerNorm<B> {
    fn forward(&self, x: Tensor<B::Device, f32>) -> Tensor<B::Device, f32> {
        // 1. 计算均值和方差（对最后一个维度）
        let mean = x.mean(-1, true);      // [batch, seq, 1]
        let var = ((x - mean.clone()) * (x - mean.clone())).mean(-1, true); // [batch, seq, 1]
        
        // 2. 归一化
        let normalized = (x - mean) / (var.sqrt() + self.eps);
        
        // 3. 仿射变换
        normalized * self.gamma.clone() + self.beta.clone()
    }
}
```

### 3.3.4 Dropout

```rust
pub struct Dropout {
    p: f32,         // 丢弃概率，如 0.1 表示 10% 的神经元被丢弃
    training: bool,  // 训练/推理模式
}

impl Dropout {
    pub fn forward<B: Backend>(&self, x: Tensor<B::Device, f32>) -> Tensor<B::Device, f32> {
        if !self.training || self.p == 0.0 {
            return x;  // 推理时直接返回，做恒等映射
        }
        
        // 生成随机 mask，值为 0 或 1
        let mask = Tensor::<B>::randn_like(&x).gt(self.p);
        // Scale-by-inverted-probability（缩放保持期望值不变）
        // 训练时用 scale = 1/(1-p)，推理时用 1.0
        x * mask * (1.0 / (1.0 - self.p))
    }
}
```

## 3.4 注意力机制与 Transformer 架构

### 3.4.1 Scaled Dot-Product Attention

Transformer 的核心算子：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```rust
pub struct Attention {
    d_k: usize,  // Key 的维度（除以 sqrt(d_k) 是为了防止点积值过大导致 softmax 梯度消失）
}

impl Attention {
    pub fn forward<B: Backend>(
        &self,
        q: Tensor<B::Device, f32>,  // [batch, num_heads, seq_q, d_k]
        k: Tensor<B::Device, f32>,  // [batch, num_heads, seq_k, d_k]
        v: Tensor<B::Device, f32>,  // [batch, num_heads, seq_k, d_v]
        mask: Option<&Tensor<B::Device, bool>>,  // 可选的注意力 mask（如 causal mask）
    ) -> Tensor<B::Device, f32> {
        // 1. 计算 Q @ K^T / sqrt(d_k)
        let scale = (self.d_k as f32).sqrt();
        let scores = Tensor::<B>::matmul(&q, &k.transpose(-2, -1)) / scale;
        
        // 2. 应用 mask（可选）
        let scores = match mask {
            Some(m) => scores.masked_fill(m, f32::NEG_INFINITY),
            None => scores,
        };
        
        // 3. Softmax 归一化（对最后一个维度）
        let attention_weights = scores.softmax(-1);
        
        // 4. 加权求和
        Tensor::<B>::matmul(&attention_weights, &v)
    }
}
```

### 3.4.2 Multi-Head Attention

多头注意力：把 Q/K/V 拆成多个头，并行做 attention，最后拼接。

```rust
pub struct MultiHeadAttention<B: Backend> {
    num_heads: usize,
    d_model: usize,      // 模型总维度
    d_k: usize,          // 每个头的维度 = d_model / num_heads
    w_q: Linear<B>,      // 投影矩阵 W_Q
    w_k: Linear<B>,      // 投影矩阵 W_K
    w_v: Linear<B>,      // 投影矩阵 W_V
    w_o: Linear<B>,      // 输出投影矩阵 W_O
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B::Device, f32>,  // [batch, seq_len, d_model]
        mask: Option<&Tensor<B::Device, bool>>,
    ) -> Tensor<B::Device, f32> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];
        
        // 1. 线性投影得到 Q, K, V
        let q = self.w_q.forward(&x);  // [batch, seq, d_model]
        let k = self.w_k.forward(&x);
        let v = self.w_v.forward(&x);
        
        // 2. reshape + transpose 拆分多头
        // [batch, seq, d_model] → [batch, seq, heads, d_k] → [batch, heads, seq, d_k]
        let q = q.reshape([batch, seq_len, self.num_heads, self.d_k])
                 .transpose(1, 2);
        let k = k.reshape([batch, seq_len, self.num_heads, self.d_k])
                 .transpose(1, 2);
        let v = v.reshape([batch, seq_len, self.num_heads, self.d_k])
                 .transpose(1, 2);
        
        // 3. 每个头做 attention（这就是 Scaled Dot-Product Attention）
        let attention_output = Attention { d_k: self.d_k }
            .forward(&q, &k, &v, mask);
        
        // 4. 拼接多头输出
        // [batch, heads, seq, d_k] → [batch, seq, heads, d_k] → [batch, seq, d_model]
        let concat = attention_output.transpose(1, 2)
                                     .reshape([batch, seq_len, self.d_model]);
        
        // 5. 输出线性投影
        self.w_o.forward(&concat)
    }
}
```

### 3.4.3 Transformer Encoder Block

```
                    ┌─────────────────────────────┐
                    │      Multi-Head Self-Attn   │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┴───────────────┐
          │     Residual Connection + LayerNorm     │
          └───────────────────────┬───────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │     Feed-Forward Network      │
                    │  (Linear → GELU → Linear)     │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┴───────────────┐
          │     Residual Connection + LayerNorm     │
          └───────────────────────┬───────────────┘
                                  │
                               输出
```

```rust
pub struct TransformerBlock<B: Backend> {
    // 预层归一化（Pre-LN，优于 Post-LN）
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    // 前馈网络 FFN
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B::Device, f32>,  // [batch, seq, d_model]
        mask: Option<&Tensor<B::Device, bool>>,
    ) -> Tensor<B::Device, f32> {
        // 子层 1：Self-Attention + 残差
        let residual = x.clone();
        let x = self.ln1.forward(&x);
        let x = self.attention.forward(&x, mask);
        let x = self.dropout.forward(&x);
        let x = x + residual;  // 残差连接

        // 子层 2：FFN + 残差
        let residual = x.clone();
        let x = self.ln2.forward(&x);
        let x = self.ff_linear1.forward(&x);
        let x = x.gelu();
        let x = self.ff_linear2.forward(&x);
        let x = self.dropout.forward(&x);
        let x = x + residual;  // 残差连接
        
        x
    }
}
```

### 3.4.4 位置编码（Positional Encoding）

Transformer 本身是置换不变的（permutation invariant），需要显式注入位置信息。

```rust
// 正弦位置编码（Sinusoidal PE，Transformer 论文原版）
pub fn sinusoidal_position_encoding<B: Backend>(
    max_len: usize,
    d_model: usize,
    device: &B::Device,
) -> Tensor<B::Device, f32> {
    // PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    // PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    
    let mut pe = Tensor::<B>::zeros([max_len, d_model], device);
    for pos in 0..max_len {
        for i in 0..(d_model / 2) {
            let div_term = (10000.0_f32.powf(2.0 * i as f32 / d_model as f32));
            pe[pos, 2*i] = (pos as f32 / div_term).sin();
            pe[pos, 2*i+1] = (pos as f32 / div_term).cos();
        }
    }
    pe
}

// 可学习位置编码（Learnable PE，BERT/GPT 实际使用）
pub struct LearnablePosEmbedding<B: Backend> {
    embeddings: Tensor<B::Device, f32>,  // [max_len, d_model]，随机初始化
}
```

## 3.5 训练引擎与数据流水线

### 3.5.1 完整训练循环

```rust
pub struct Trainer<B: Backend> {
    model: Box<dyn Module<B>>,
    optimizer: AdamW<B>,
    loss_fn: CrossEntropyLoss,
    device: B::Device,
}

impl<B: Backend> Trainer<B> {
    pub fn train(&mut self, dataloader: DataLoader, epochs: usize) {
        self.model.train();
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0f32;
            let mut num_batches = 0;
            
            for (batch_x, batch_y) in dataloader.iter() {
                // 步骤 1：把数据搬到目标设备
                let x = batch_x.to(&self.device);
                let y = batch_y.to(&self.device);
                
                // 步骤 2：梯度清零（上一轮的梯度会累积）
                self.optimizer.zero_grad();
                
                // 步骤 3：前向传播
                let logits = self.model.forward(&x);
                
                // 步骤 4：计算损失
                let loss = self.loss_fn.forward(&logits, &y);
                
                // 步骤 5：反向传播（自动微分引擎工作）
                loss.backward();
                
                // 步骤 6：梯度裁剪（防止梯度爆炸）
                self.optimizer.clip_grad_norm(1.0);
                
                // 步骤 7：参数更新
                self.optimizer.step();
                
                // 统计
                total_loss += loss.item();  // .item() 把标量张量取为纯 Rust 值
                num_batches += 1;
            }
            
            println!("Epoch {}, Loss: {}", epoch, total_loss / num_batches);
        }
    }
}
```

### 3.5.2 学习率调度集成

```rust
pub struct AdamWWithScheduler<B: Backend> {
    optimizer: AdamW<B>,
    scheduler: CosineAnnealingWithWarmup,
    step_count: usize,
}

impl<B: Backend> Optimizer<B> for AdamWWithScheduler<B> {
    fn step(&mut self) {
        // 先更新学习率
        let lr = self.scheduler.compute_lr(self.step_count);
        self.optimizer.set_lr(lr);
        // 再用更新后的学习率做参数更新
        self.optimizer.step();
        self.step_count += 1;
    }
}
```

## 3.6 模型持久化与序列化

### 3.6.1 保存格式

```
┌──────────────────────────────────────────────────────────┐
│  Rust 生态中常见的模型权重格式：                           │
│                                                          │
│  格式             扩展名    特点                           │
│  ────────────────────────────────────────────────────── │
│  safetensors      .safetensors  JSON 头 + 原始数据块      │
│                                  无 Python 依赖，推荐      │
│  PyTorch StateDict .pt/.bin  Pickle 序列化                │
│                              有 Python 依赖，有安全风险    │
│  GGUF              .gguf     Llama.cpp 专用格式           │
│                              含量化元数据，推理高效        │
│  ONNX              .onnx     开放格式，跨框架互导          │
│                              含计算图结构                  │
│  Burn 原生          .json + .bin  Burn 框架原生格式       │
└──────────────────────────────────────────────────────────┘
```

### 3.6.2 safetensors 格式结构

```
safetensors 文件 = 头部 JSON + 数据区二进制

头部 JSON（小端序，8字节长度 + JSON 内容）:
{
  "tensor_name_1": {
    "dtype": "F32",
    "shape": [30522, 768],
    "data_offsets": [0, 93606912]    // 在整个文件中的字节偏移
  },
  "tensor_name_2": {
    "dtype": "F32",
    "shape": [768],
    "data_offsets": [93606912, 93609984]
  },
  "__metadata__": {
    "format": "pt",
    "total_size": 93609984
  }
}

数据区：原始 float32 数组连续排列，与 data_offsets 对应
```

### 3.6.3 保存/加载实现

```rust
impl<B: Backend> Module<B> {
    pub fn save_safetensors(&self, path: &Path) -> Result<()> {
        let mut tensors: HashMap<String, Tensor<B>> = HashMap::new();
        self.collect_params("model", &mut tensors);
        
        let mut header = HashMap::new();
        let mut offset = 0u64;
        
        for (name, tensor) in &tensors {
            let byte_len = tensor.numel() * std::mem::size_of::<f32>() as u64;
            header.insert(name.clone(), json!({
                "dtype": "F32",
                "shape": tensor.shape(),
                "data_offsets": [offset, offset + byte_len]
            }));
            offset += byte_len;
        }
        
        // 写文件
        let header_bytes = serde_json::to_vec(&header)?;
        let mut file = File::create(path)?;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&header_bytes)?;
        for tensor in tensors.values() {
            file.write_all(tensor.data_as_bytes())?;
        }
        
        Ok(())
    }
}
```

---

# 第四章 AI模型执行原理

> 本章把底层和框架串起来，讲清楚训练和推理过程中到底发生了什么。

## 4.1 前向传播（Forward Pass）

### 4.1.1 完整流程图解

以一个 2 层 MLP 二分类为例：

```
输入: x = [x₁, x₂]                    标签: y ∈ {0, 1}

   ┌───────┐    h = xW₁ + b₁   ┌─────────┐    a = ReLU(h)    ┌───────┐
   │  x    │ ─────────────────→ │  Linear  │ ───────────────→ │ ReLU  │
   │[1, 2] │                   │ W₁:2×4  │                    └───┬───┘
   └───────┘                   └────┬────┘                        │
                                     │                              │
                                     │                              ▼
                                     │    记录计算图节点:            │
                                     │    Node1: MatMul(x, W₁)    ┌─────────┐
                                     │    Node2: Add(Node1, b₁)   │  Linear  │
                                     │    Node3: ReLU(Node2)      │ W₂:4×1  │
                                     │    Node4: MatMul(Node3,W₂) └────┬────┘
                                     │    Node5: Add(Node4, b₂)         │
                                     │                                 ▼
                                     │                           ┌─────────┐
                                     │                           │ Sigmoid │ → ŷ = σ(z)
                                     │                           └─────────┘
                                     └────────────────────────────────┘
```

### 4.1.2 每个节点做两件事

在反向模式 autograd 中，每个运算同时执行两件事：

```
1. 计算实际结果（数值计算）
   y = x @ W + b  → 得到一个新的 Tensor

2. 注册计算图节点（为反向传播做准备）
   node.id = 5
   node.op = MatMul
   node.inputs = [node3.id, node_w2.id, node_b2.id]
   node.output = y
   node.backward_fn = 闭包，封装 dL/dx、dL/dW、dL/db 的计算逻辑
```

### 4.1.3 内存快照

```
前向传播后的显存状态（2层 MLP，batch=64, hidden=256）：

| 张量          | 形状        | 大小(F32)  | 来源              | 用途           |
|--------------|------------|-----------|-------------------|---------------|
| x            | [64, 784]  | 196 KB    | DataLoader        | 输入           |
| W₁           | [784, 256] | 784 KB    | 模型参数          | 可训练         |
| b₁           | [256]      | 1 KB      | 模型参数          | 可训练         |
| h = xW₁+b₁  | [64, 256]  | 64 KB     | MatMul + Add      | 计算图中间值    |
| a = ReLU(h)  | [64, 256]  | 64 KB     | ReLU              | 计算图中间值    |
| W₂           | [256, 1]   | 1 KB      | 模型参数          | 可训练         |
| b₂           | [1]        | 4 B       | 模型参数          | 可训练         |
| z = aW₂+b₂  | [64, 1]    | 256 B     | MatMul + Add      | 计算图中间值    |
| ŷ = σ(z)     | [64, 1]    | 256 B     | Sigmoid           | 输出           |
| L = BCE(ŷ,y) | [1]        | 4 B       | CrossEntropy      | 损失值         |

总大小 ≈ 1.1 MB  ← 但反向传播时需要额外的梯度张量，总显存约 4 MB
```

## 4.2 损失计算

损失函数通常是计算图的末端节点：

```
L = CrossEntropy(ŷ, y)

这个运算同时：
1. 计算标量损失值
2. 在计算图上注册为根节点（backward 从这里开始）
3. 因为 y 是标签（不需要梯度），所以只连接 ŷ 到 L
```

## 4.3 反向传播（Backward Pass）

### 4.3.1 拓扑排序

反向传播的第一步是对计算图做**反向拓扑排序**：

```
前向传播的依赖方向（数据流）：
  x → Node1(MatMul) → Node2(Add) → Node3(ReLU) → Node4(MatMul) → Node5(Add) → σ → L

反向传播的依赖方向（梯度流）：
  L → σ → Node5 → Node4 → Node3 → Node2 → Node1 → W₂, b₂, W₁, b₁, x

拓扑排序保证：要计算某个节点的梯度时，它所有后继节点的梯度已经算好了。
```

### 4.3.2 链式法则逐层应用

以 W₁ 的梯度为例，完整的链式法则推导：

```
dL/dW₁ = dL/dŷ · dŷ/dz · dz/da · da/dh · dh/dW₁

逐步代入：

Step 1: dL/dŷ · dŷ/dz = σ'(z) · (ŷ - y) · ...不对，直接 CE + Softmax 更简洁
  用 BCE + Sigmoid 的简洁梯度：
  dL/dz = (ŷ - y)  （这是 BCE + Sigmoid 的漂亮简化）

Step 2: dz/da = W₂ᵀ
  因为 z = a @ W₂ + b₂，所以 dz/da = W₂ᵀ

Step 3: da/dh = ReLU'(h)
  ReLU 的导数：h > 0 → 1, h ≤ 0 → 0

Step 4: dh/dW₁ = xᵀ @ ... 不对
  h = x @ W₁ + b₁，所以 dh/dW₁ = xᵀ @ (上面的梯度)

组合：
  dL/dW₁ = xᵀ @ [ReLU'(h) ⊙ (W₂ᵀ @ dL/dz)]
  dL/dW₂ = aᵀ @ dL/dz
  dL/db₂ = sum(dL/dz)  (广播梯度求和)
  dL/db₁ = ReLU'(h) ⊙ (W₂ᵀ @ dL/dz) 对 batch 维求和
```

### 4.3.3 梯度累积过程

```
反向传播中 grads HashMap 的填充过程：

grads[L]        = [1]                     (根节点梯度为1)
grads[σ]        = dL/dz = ŷ - y          (BCE + Sigmoid的简洁结果)
grads[Node5]    = W₂ᵀ @ grads[σ]         (MatMul反向)
grads[Node4]    = ReLU'(h) ⊙ grads[Node5] (ReLU反向，逐元素乘)
grads[Node3]    = grads[Node4]           (Add反向，梯度原样传递)
grads[Node2]    = grads[Node3]           (MatMul反向的中间结果)

grads[W₂]       = aᵀ @ grads[σ]          (直接存到W₂.grad字段)
grads[b₂]       = sum(grads[σ], dim=0)   (直接存到b₂.grad字段)
grads[W₁]       = xᵀ @ grads[Node4]      (直接存到W₁.grad字段)
grads[b₁]       = sum(grads[Node4], dim=0)
```

### 4.3.4 为什么要保存中间张量

上一章提到前向传播要保存所有中间张量，现在清楚了——

```
反向传播时，每个 backward 函数需要前向传播时的值：

MatMul backward:  dL/dx = grad_output @ Wᵀ    → 需要 W（前向输入）
                 dL/dW = xᵀ @ grad_output     → 需要 x（前向输入）

ReLU backward:    dL/dx = grad_output ⊙ ReLU'(x) → 需要 x（前向输入，计算导数）

Softmax backward: dL/dx = softmax(x) ⊙ (dL/dy - sum(softmax(x) ⊙ dL/dy))
                   → 需要 softmax(x)（前向输出，或可重算）
```

## 4.4 参数更新

### 4.4.1 从梯度到新参数

以 AdamW 为例：

```
梯度已计算好（上一步）：grad_W₁ ∈ grd_W₁
                    grad_W₂ ∈ grd_W₂

优化器状态（初始化时创建，与参数一一对应）：
  m_W₁: 一阶矩（初始全0）
  v_W₁: 二阶矩（初始全0）
  beta1 = 0.9, beta2 = 0.999, lr = 3e-4, weight_decay = 0.01

第 t 步更新 W₁：

1. 计算指数平均：
   m_W₁ = β₁·m_W₁ + (1-β₁)·grad_W₁
   v_W₁ = β₂·v_W₁ + (1-β₂)·grad_W₁²

2. 偏差修正：
   m̂_W₁ = m_W₁ / (1 - β₁ᵗ)
   v̂_W₁ = v_W₁ / (1 - β₂ᵗ)

3. 更新参数（包含权重衰减）：
   W₁ = W₁ - lr · m̂_W₁ / (√v̂_W₁ + ε) - lr · weight_decay · W₁

4. 清空梯度（为下一轮做准备）：
   optimizer.zero_grad()
   → grad_W₁ = 0, grad_W₂ = 0  ...
```

### 4.4.2 优化器状态的内存占用

```
每个可训练参数，AdamW 额外需要：
  m  (与参数同形状，f32)   → +1x 参数量
  v  (与参数同形状，f32)   → +1x 参数量

所以 AdamW 训练的总参数量 =
  模型本身    1x
  梯度         1x
  一阶矩 m    1x
  二阶矩 v    1x
  ─────────────
  合计       4x

这就是为什么 LLaMA-7B 用 AdamW 训练需要 ~28GB 显存（7B × 4 bytes × 4 ≈ 112 GB？不对，还要考虑中间激活值和显存碎片）
```

## 4.5 推理阶段（Inference）

### 4.5.1 推理 vs 训练的区别

```
┌──────────────┬────────────────────┬────────────────────┐
│              │ 训练 Training       │ 推理 Inference      │
├──────────────┼────────────────────┼────────────────────┤
│ autograd    │ 开启，构建计算图    │ 关闭，不记录计算图  │
│ 中间张量     │ 全部保存，供反向    │ 用完即释，不保存    │
│ Dropout     │ 激活，随机丢弃      │ 关闭，做恒等映射    │
│ BatchNorm   │ 用当前 batch 统计   │ 用全局统计量        │
│ 损失计算     │ 需要               │ 不需要              │
│ 反向传播     │ 需要               │ 不需要              │
│ 参数更新     │ 需要               │ 不需要              │
│ 显存占用     │ 大（参数×4 + 激活） │ 小（参数×1 + 激活） │
│ 速度         │ 较慢               │ 较快（无反向开销）   │
└──────────────┴────────────────────┴────────────────────┘
```

### 4.5.2 推理的性能优化

```
1. 算子融合（Kernel Fusion）
   LayerNorm → GELU → Linear 的逐元素操作融合成一个 kernel，减少显存读写

2. KV Cache（大模型推理）
   自回归生成时，已生成 token 的 K 和 V 矩阵可以缓存
   每步只需计算新 token 的 Q，与缓存的 K/V 做 attention
   时间复杂度从 O(n²) 降到 O(n) （n 是序列长度）

3. 量化（Quantization）
   权重量化到 INT8 / INT4 / INT3，大幅减少显存占用和计算量
   GPTQ、AWQ、SmoothQuant 算法
   GGUF 格式原生支持量化

4. Speculative Decoding（投机解码）
   用小模型先猜多个 token，大模型并行验证
   接受正确的 token，拒绝错误的继续采样
   加速比可达 2-3x

5. Continuous Batching（vLLM 技术）
   把不同请求的序列拼成大 batch 统一处理
   Pad 到同一长度后用 FlashAttention 高效计算
```

### 4.5.3 自回归生成的完整流程（GPT 推理）

```
输入序列: [t₀, t₁, t₂, ..., tₙ₋₁]

for step in range(max_tokens):
    # Step 1: 前向传播（用 KV Cache）
    logits = model([t₀, ..., tₙ₊step₋₁])   # 只计算新 token 的 Q
    
    # Step 2: 取最后一个位置的 logits
    next_token_logits = logits[:, -1, :]   # [batch, vocab_size]
    
    # Step 3: 应用温度 / top-p / top-k 采样
    probs = softmax(next_token_logits / temperature)
    probs = top_k_filter(probs, k=50)
    probs = nucleus_filter(probs, p=0.9)
    next_token = sample_from(probs)
    
    # Step 4: 拼接到序列
    sequence.append(next_token)
    
    # Step 5: 更新 KV Cache（只保存新 token 的 K/V）
    kv_cache.append(new_k, new_v)
    
    # Step 6: 检查终止条件（EOS / 长度上限）
    if next_token == eos_token_id: break

# 返回完整生成序列
return sequence
```

---

# 第五章 Rust 生态 AI 框架现状

## 5.1 框架对比

| 框架 | 类型 | 训练能力 | 推理能力 | 后端 | 成熟度 | 最新版本 |
|------|------|---------|---------|------|--------|---------|
| **Burn** | 完整框架 | ✅ 强 | ✅ 强 | CubeCL (CUDA/Metal/Vulkan/WebGPU) | 生产级 | 0.23.0-pre.2 |
| **Candle** | 极简框架 | ⚠️ 弱 | ✅ 强 | CUDA / Metal / CPU | 生产级（推理） | 0.11.1 |
| **tch-rs** | PyTorch 绑定 | ✅ 强 | ✅ 强 | LibTorch C++ | 生产级 | 0.27.0 |
| **tract** | 推理引擎 | ❌ | ✅ 强 | CPU / GPU | 生产级 | — |
| **pergrine-ml** | 从零实现 | ✅ | ✅ | 自己实现 | 实验性 | — |
| **volta** | 教学框架 | ✅ | ✅ | 自己实现 | 实验性 | 0.3.0 |

## 5.2 核心推荐

### Burn — 纯 Rust 全栈框架（最推荐用于训练）

- **特性**：纯 Rust，动态图 + 静态优化，CubeCL 多后端（一个模型编译到 CUDA/RoCm/Metal/Vulkan/WebGPU/CPU）
- **架构亮点**：编译期类型推导张量形状，运行期动态执行静态优化
- **版本**：0.22.1（稳定）/ 0.23.0-pre.2（预发布，2026-11-09 更新）
- **适用**：想在 Rust 中完成训练 + 推理全流程，追求性能和可移植性

### Candle — Hugging Face 极简框架（最推荐用于推理）

- **特性**：代码量小（可完整阅读），GGUF/safetensors 支持，100+ 模型示例
- **生态**：mistral.rs、Kyutai Moshi、kalosm 都基于 Candle
- **版本**：0.11.1（2026-11-01）
- **适用**：LLM 推理、想学习框架内部实现、嵌入到 Rust 应用

### tch-rs — PyTorch C++ 绑定

- **特性**：直接调用 LibTorch C++ API，获得 PyTorch 完整能力
- **版本**：0.27.0（2026-11-10）
- **缺点**：需要 PyTorch C++ 运行时，体积大，不是纯 Rust
- **适用**：已经熟悉 PyTorch，想在 Rust 中复用现有模型

## 5.3 当前实验性项目

| 项目 | 特点 | 学习价值 |
|------|------|---------|
| peregrine-ml | 纯 Rust，用 Apple Accelerate 做 BLAS | 理解从零实现框架 |
| volta | 教学目的，动态图 + NumPy 风格广播 | 理解 autograd 内部 |
| neuralrs | 手写 CUDA kernel | 理解 GPU kernel 编写 |
| zyx | LGPL，自己的 ML 库 | 备选方案 |

## 5.4 Rust AI 生态的现状总结

```
┌──────────────────────────────────────────────────────────┐
│  Rust AI 生态定位（2026年底）：                           │
│                                                          │
│  ✅ 推理：非常成熟                                         │
│    Candle + GGUF + mistral.rs → 生产级 LLM 推理           │
│    无需 Python 运行时                                    │
│                                                          │
│  ⚠️ 训练：可用但生态仍小                                  │
│    Burn 是唯一能独立训练大模型的纯 Rust 框架              │
│    tch-rs 绑定 PyTorch C++ 做训练                        │
│    PEFT / LoRA / 多节点训练尚未在 Burn 中完善             │
│                                                          │
│  ❌ 预训练：还不可能（需要 Python 生态的海量工具链）       │
│                                                          │
│  实践建议：                                               │
│  - 训练大模型 → 在 Python/PyTorch 中训练                 │
│  - 导出 ONNX / safetensors → 在 Rust 中推理              │
│  - 小模型训练 → 用 Burn 直接在 Rust 中完成               │
└──────────────────────────────────────────────────────────┘
```

---

# 第六章 从零开发路线图

> 现在你已经理解了三层架构，下面是从零实现的推荐顺序。

## 阶段 1：张量库（底层）

```
目标：实现一个最小可用的张量数据结构 + CPU 上的基础运算

Step 1.1  张量数据结构
  - 定义 Tensor<T>：Vec<T> + shape + strides
  - 实现 reshape / transpose / permute（零拷贝 view）
  - 实现 contiguous()（数据拷贝）

Step 1.2  基础运算（CPU 版）
  - 逐元素：add / sub / mul / div / neg
  - 规约：sum / mean / max / min
  - 矩阵：matmul（朴素 O(n³) 即可）
  - 广播机制
  - 激活函数：relu / gelu / sigmoid / tanh / silu

Step 1.3  性能优化
  - SIMD：用 rayon crate 做并行 matmul
  - 可选：绑定 BLAS（blas-sys crate）
  - 测试 correctness vs NumPy / PyTorch

预计：1-2 周（不做 GPU，不做 autograd）
```

## 阶段 2：自动微分引擎（框架核心）

```
目标：为张量加上反向模式自动微分

Step 2.1  计算图节点定义
  - Node struct：op / inputs / backward_fn
  - Op enum：覆盖所有已有运算
  - Graph 管理器：全局单例或 ThreadLocal

Step 2.2  backward 函数实现
  - 为每个 Op 手写 backward 公式
  - MatMul backward：dL/dx = grad @ Wᵀ, dL/dW = xᵀ @ grad
  - ReLU backward：dL/dx = grad ⊙ (x > 0 ? 1 : 0)
  - Softmax + CrossEntropy 组合梯度
  - 广播的梯度处理（需要 sum 回原始形状）

Step 2.3  backward 入口
  - 拓扑排序（Kahn 算法或 DFS 后序）
  - 按序遍历，累积梯度
  - 叶子节点参数的 .grad 字段更新

Step 2.4  梯度检查
  - 实现 numerical gradient（有限差分）
  - 写单元测试：对每个 Op 验证 forward + backward 的正确性

预计：2-3 周（最核心也最容易出错的阶段）
```

## 阶段 3：神经网络模块与优化器（框架层）

```
目标：有了 autograd 后，封装常用模块和优化器

Step 3.1  模块系统
  - Module trait：forward / parameters / train / eval
  - Linear（全连接层）
  - LayerNorm / BatchNorm
  - Dropout
  - Embedding
  - Conv2d（进阶）

Step 3.2  优化器
  - SGD + Momentum
  - AdamW（重点）
  - 参数组管理（param_groups）
  - zero_grad / step

Step 3.3  损失函数
  - MSE
  - CrossEntropy（含 Softmax）
  - BCEWithLogits

预计：1 周
```

## 阶段 4：构建实际模型（模型层）

```
目标：用框架组合出 Transformer

Step 4.1  位置编码
  - Sinusoidal PE（固定）
  - Learnable PE（可训练）
  - Rotary Embedding（RoPE，LLaMA 风格）

Step 4.2  Multi-Head Attention
  - Scaled Dot-Product Attention
  - Mask 机制（Causal Mask 用于 Decoder）
  - 多头拆分与拼接

Step 4.3  Transformer Block
  - Pre-LN + Self-Attention + FFN
  - 残差连接 + Dropout

Step 4.4  Encoder / Decoder / Full Transformer
  - BERT 风格 Encoder-only
  - GPT 风格 Decoder-only（自回归）
  - T5 风格 Encoder-Decoder

预计：1-2 周
```

## 阶段 5：训练与推理流水线

```
目标：端到端跑通训练 + 推理

Step 5.1  DataLoader
  - 数据预处理（tokenization 可选）
  - Padding + Masking
  - 批量采样（shuffle）

Step 5.2  训练循环
  - epoch / batch 管理
  - 梯度累积
  - 梯度裁剪
  - 学习率调度（Warmup + CosineAnneal）
  - Loss / Accuracy 日志

Step 5.3  模型保存/加载
  - safetensors 格式（推荐）
  - 或简单的 JSON + bin

Step 5.4  推理
  - 关闭 autograd 的 forward（no_grad 模式）
  - Greedy / Top-k / Top-p 采样
  - EOS 检测 + 序列截断

预计：1 周
```

## 阶段 6：可选进阶

```
Step 6.1  GPU 加速
  - CUDA：用 candle-core 的 CUDA kernel 或自己写
  - Metal：Metal Performance Shaders
  - 推荐：先直接用 Burn 框架（它已经有成熟的 GPU 后端）

Step 6.2  显存优化
  - Gradient Checkpointing
  - Mixed Precision（BF16 / FP16）
  - KV Cache（Transformer 推理）

Step 6.3  算子融合
  - LayerNorm + GELU + Linear 融合
  - FlashAttention 算法

Step 6.4  多 GPU / 分布式
  - DDP（Data Parallel）
  - ZeRO 优化器状态分片

Step 6.5  项目打包
  - Workspace 架构拆分 crate
  - 模块清晰分离
  - 单元测试 + 集成测试
```

---

# 附录 A：关键数值稳定性技巧

## 为什么 Softmax 不能直接算

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

当 $x_i$ 值很大时（如 1000），$e^{1000}$ 会溢出 f32 的最大值（~3.4e38）。标准做法：

$$\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}$$

减掉最大值后最大的 $e^{x_i - \max} = e^0 = 1$，数值稳定。

## CrossEntropy 同理

$$\text{CE}(y, \hat{y}) = -\sum_i y_i \log(\text{softmax}(x)_i)$$

用 log-softmax 的稳定公式：

$$\log\text{softmax}(x)_i = x_i - \max_j x_j - \log\sum_j e^{x_j - \max_k x_k}$$

## LayerNorm 的 eps

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$\epsilon$ 不能省。如果方差极小（$\sigma^2 \approx 0$），分母接近 0 会产生数值爆炸。通常 $\epsilon = 10^{-5}$ 或 $10^{-6}$。

# 附录 B：Rust 生态关键 crate 速查

| 用途 | Crate | 版本 | 说明 |
|------|-------|------|------|
| 深度学习框架 | burn | 0.22.1 | 纯 Rust 完整框架 |
| 极简推理框架 | candle-core | 0.11.1 | Hugging Face 出品 |
| PyTorch 绑定 | tch | 0.27.0 | LibTorch C++ API 封装 |
| safetensors 读写 | safetensors | — | 权重格式 |
| 张量运算基础 | ndarray | — | NumPy 风格的 n-d 数组 |
| BLAS 绑定 | blas-sys | — | 基础线性代数（OpenBLAS / Intel MKL）|
| 并行计算 | rayon | — | CPU 多线程 |
| CUDA 绑定 | cuda-sys / cublas-sys | — | GPU 加速 |
| WebGPU | wgpu | — | 跨平台 GPU API |
| 序列化 | serde + serde_json | — | 模型配置 |
| Rand | rand | — | 权重初始化、Dropout 随机数 |

# 附录 C：术语对照表

| 中文 | 英文 | 缩写/符号 |
|------|------|----------|
| 张量 | Tensor | 多维数组 |
| 矩阵乘法 | Matrix Multiplication | MatMul / GEMM |
| 自动微分 | Automatic Differentiation | Autograd |
| 反向传播 | Backpropagation | Backward Pass |
| 前向传播 | Forward Pass | Forward Pass |
| 计算图 | Computation Graph | DAG（有向无环图）|
| 链式法则 | Chain Rule | — |
| 梯度下降 | Gradient Descent | GD |
| 小批量 | Mini-batch | — |
| 学习率 | Learning Rate | η / lr |
| 优化器 | Optimizer | Adam / AdamW |
| 残差连接 | Residual Connection | Skip Connection |
| 位置编码 | Positional Encoding | PE |
| 自注意力 | Self-Attention | — |
| 缩放点积注意力 | Scaled Dot-Product Attention | — |
| 多头注意力 | Multi-Head Attention | MHA |
| 归一化 | Normalization | LayerNorm / BatchNorm |
| 权重衰减 | Weight Decay | L2 Regularization |
| 丢弃 | Dropout | — |
| 张量并行 | Tensor Parallelism | TP |
| 数据并行 | Data Parallelism | DP / DDP |
| 张量并行 | Kernel Fusion | 算子融合 |
| 混合精度 | Mixed Precision | FP16 / BF16 |
| 梯度检查点 | Gradient Checkpointing | Recomputation |

---

> 文档结束。这份指南严格按照 **底层数学 → 框架工程 → 执行原理** 的三层结构组织，每一层都有明确的职责边界。理解了这三层，你就能从零写出自己的 AI 框架，或者更深入地使用 Burn / Candle 这些成熟框架。
