# TIGER 生成式推荐复现（含两处修复）

基于 [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER)（MIT License）的复现。
在原实现基础上**补齐了论文的前缀树约束解码**，并**定位、修复了 RQ-VAE 的码本塌缩问题**。

数据集：Amazon Beauty（2014 版），12,095 个物品。

---

## 一、这条链路在做什么

传统召回是"用户向量 × 物品向量找最近邻"。TIGER 换了个思路：**把物品变成一个 3–4 个 token 的语义 ID，让模型像写句子一样"生成"下一个物品的 ID。**

```text
物品文本（title / description / categories / brand）
        ↓ sentence-t5-base
    768 维语义向量
        ↓ RQ-VAE（三层残差量化）
    语义 ID：<a3> <b17> <c402> [<p>]

用户历史（SID 序列） → T5 encoder-decoder → 生成下一个物品的 SID
        ↓ beam search（K=30）
    召回候选
```

---

## 二、环境

```bash
pip install "pandas==2.1.4" "numpy<2" "pyarrow==15.0.2" \
            polars tqdm "transformers==4.40.0" \
            sentence-transformers huggingface_hub
```

> 不要直接用仓库自带的 `requirements.txt`——它锁的是 `python==3.7.16` / `torch==1.10.1`，
> 与 PyTorch 2.x 镜像冲突。

---

## 三、复现步骤

```bash
# 1. 数据预处理（把物品文本编码成 768 维向量）
jupyter notebook data/process.ipynb

# 2. 训练 RQ-VAE（生成语义 ID）
cd rqvae
python main.py --data_path ../data/Beauty/item_emb.parquet \
    --epochs 3000 --warmup_epochs 50 --device cuda

# 3. 生成 SID 文件
#    先把 generate_code.py 里的 ckpt_path 改成上一步的新目录
python generate_code.py

# 4. 训练 TIGER
cd ../model
python main.py --num_epochs 200 --infer_size 32
```

---

## 四、结果

| 指标 | A：原始实现<br>（塌缩 SID + 约束解码） | **D：本次修复<br>（修复 SID + 约束解码）** | 变化 |
| --- | --- | --- | --- |
| Recall@5 | 2.77% | **3.03%** | +0.26pp |
| Recall@10 | 4.54% | **5.04%** | +0.50pp |
| **Recall@20** | **6.86%** | **7.57%** | **+0.71pp（+10.3%）** |
| NDCG@5 | 1.79% | **1.93%** | +0.14pp |
| NDCG@10 | 2.36% | **2.57%** | +0.21pp |
| **NDCG@20** | **2.94%** | **3.21%** | **+0.27pp（+9.2%）** |

两组训练轮数相当（A 为 44 轮、D 为 41 轮），所以提升不是来自多训。

模型规模：4,590,848 参数（4.6M），`d_model=128`、4 层编码器 + 4 层解码器、6 个头。

---

## 五、我修的两处

### 5.1 补齐前缀树约束解码

上游实现直接调用 `model.generate()`，**没有论文里的约束解码**。

问题规模：三层码本各 256 个码字，理论上能组成 `256³ = 16,777,216` 种 SID，
而真实物品只有 12,095 个——**有效率仅 0.072%，99.93% 的组合是无效的**。
不约束时，beam 里会有大量位置被无效序列占掉。

实现方式：把所有合法 SID 建成前缀树，用 HuggingFace 的 `prefix_allowed_tokens_fn`
在每一步只允许"能构成合法前缀"的 token 参与 beam search。

### 5.2 定位并修复 RQ-VAE 码本塌缩

**现象**：第 1 层 256 个码字，实际只用了 **9 个**。

**根因**：原训练配置是 `--epochs 50`（默认 3000），而 `--warmup_epochs` 默认也是 50——
**warmup 吃掉了全部轮数**，学习率全程处于爬坡阶段，训练还没进入正常阶段就停了。

**修复**：`--epochs 3000 --warmup_epochs 50`（warmup 占比降到 1.7%）。

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 第 1 层码字使用 | 9 / 256 | **36 / 256** |
| 第 2 层码字使用 | 125 / 256 | **256 / 256** |
| 前 3 token 碰撞率 | **5.18%** | **0.64%**（↓ 8 倍） |

码字分布也做了检查：第 1 层前 10% 的码字承载 22.1% 的物品（完全均匀是 10%），
`max/median = 5.2`，属于轻微不均，不是长尾塌缩。

---

## 六、消融实验

当前两组：

| 组 | SID | 约束解码 | Recall@20 |
| --- | --- | --- | --- |
| A | 塌缩（50 轮） | ❌ | 6.86% |
| D | 修复（3000 轮） | ✅ | 7.57% |

**A → D 是两项修复合计的效果。** 单独的归因（约束解码贡献多少、码本修复贡献多少）
尚未做，原因见「已知局限」。

---

## 七、和论文的差距

论文在 Amazon Beauty 上 Recall@5 约 6.7%（以论文表格为准），本文为 3.03%，约为论文的 45%。三个原因：

| 原因 | 说明 |
| --- | --- |
| 训练轮数 | 本文 41 轮（受预算限制中断），论文 200 轮 |
| SID 质量 | 第 1 层仍只用到 36/256 个码字，粗粒度区分力弱于论文 |
| 骨干模型 | 本文是从零训练的 4.6M 小 T5，论文用预训练 T5-base |

---

## 八、已知局限

- **没有做单独的消融**：约束解码与码本修复是同时生效的，无法分离各自贡献
- **训练未跑满**：41/200 轮，因云实例欠费中断；验证集指标仍在上升区间
- **没有实现 dead-code restart**：第一层码字使用率仍偏低（36/256）
- 只验证了随机初始化的 T5，未对比预训练模型

---

## 九、目录结构

```text
LICENSE                 原仓库 MIT 协议（必须保留）
model/
  main.py               TIGER 训练 + 约束解码（本次修改）
  dataset.py            滑窗构造样本、SID 映射
  dataloader.py         批次拼接、padding
  logs/
    tiger_fixed_clean.log      本次运行日志（D 组）
    tiger_collapsed_44ep.log   基线运行日志（A 组）
  ckpt/
    tiger.pth                  本次最佳模型（第 41 轮）
    tiger_collapsed_44ep.pth   基线模型
rqvae/
  main.py               RQ-VAE 训练
  generate_code.py      生成 SID
  rqvae_3000.log        RQ-VAE 训练日志（碰撞率证据）
data/Beauty/
  Beauty_t5_rqvae.npy   新 SID（第 1 层 36 码字）
  OLD_sid_collapsed.npy 旧 SID（第 1 层 9 码字）
```

## 十、许可

代码基于 [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER)，遵循 MIT License。
本仓库的改动（约束解码实现、码本问题定位与修复）同样以 MIT 发布。
