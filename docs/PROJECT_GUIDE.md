# TIGER 复现代码结构

## 完整链路

```text
Amazon Beauty reviews + metadata
        ↓ data/process.ipynb
用户序列 + 768 维 Sentence-T5 物品向量
        ↓ rqvae/main.py
RQ-VAE：Encoder → 三层残差量化 → Decoder
        ↓ rqvae/generate_code.py
每件物品 3–4 token 的语义 ID
        ↓ model/dataset.py + dataloader.py
历史 SID 输入、目标 SID 标签、attention mask
        ↓ model/main.py
T5 encoder-decoder + 前缀树约束 beam search
        ↓
Recall@K / NDCG@K
```

当前项目只包含生成式召回，没有第二阶段排序。

## 文件职责

- `data/process.ipynb`：解析 reviews/meta、构造用户序列、生成 Sentence-T5 embedding。
- `rqvae/datasets.py`：读取 `item_emb.parquet`。
- `rqvae/models/rqvae.py`：Encoder、ResidualVectorQuantizer、Decoder 和总损失。
- `rqvae/models/rq.py`：三层残差量化。
- `rqvae/models/vq.py`：码本最近邻、commitment/codebook loss、Sinkhorn。
- `rqvae/trainer.py`：warmup、训练、碰撞率评估和 checkpoint。
- `rqvae/generate_code.py`：生成 SID、处理碰撞、增加末位消歧。
- `model/dataset.py`：滑窗样本、padding、物品 ID 到 SID。
- `model/dataloader.py`：批次拼接和 attention mask。
- `model/main.py`：TIGER 训练、前缀树约束、beam 评估和 `--eval_only`。

## 核心结论

| 组 | SID | 约束解码 | Recall@20 | NDCG@20 |
| --- | --- | --- | ---: | ---: |
| A | 塌缩 | 否 | 6.86% | 2.94% |
| B | 塌缩 | 是 | 6.86% | 2.94% |
| C | 修复 | 否 | 7.57% | 3.21% |
| D | 修复 | 是 | **7.60%** | **3.21%** |

码本修复贡献约 +10.4% 相对提升；约束解码只贡献约 +0.03pp，但能保证生成 SID 合法。

## 推荐解释顺序

1. 文本内容 → 语义向量 → 离散 SID。
2. 多层残差量化相当于“先粗后细”。
3. 用户历史 SID → 自回归生成下一个 SID。
4. beam search 产生候选，trie 限制合法前缀。
5. 修码本提升表达力；约束解码提升合法性而非显著指标。
6. 当前仍是从零训练的小 T5，训练轮数和骨干规模限制了与论文的差距。
