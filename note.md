## 方案总览

1. 数据准备 → train/val/test（按样本、按标签分层拆分）
2. 合适的 encoder（文本→DistilBERT / 表格→MLP/LightGBM / 混合→concat+MLP）抽 embedding
3. 可选降维（PCA/UMAP）到 D=32–128（加速与去噪）
4. 运行 **ADPGBG-like 粒球生成器**（密度峰中心 + shortest-hetero-distance 半径上限 + 扩张/纯度修剪）
5. 粒球后处理（合并/分裂/离群弱化 / 多尺度细化）
6. 每标签三域判定（per-label score → α/β 网格搜索）
7. 边界回退（DEFER → k-NN 或 局部再粒化）
8. 严格评估（micro/macro-F1、Coverage@Accuracy、per-label AUC、GB 质量指标）
9. 消融与可视化（UMAP + GB overlay，ablation：no-hetero, no-postprocess, varying min_size）

## 流程

```
原始表格数据
        |
        V
[ FT-Transformer ]
(特征编码器，学习高质量特征表示)
        |
        V
特征表示向量 (Embeddings)
        |
        V
[ ADPGBG ]
(在表示空间进行基于密度的自适应粒球生成)
        |
        V
高质量粒球集合 (每个粒球有中心、半径、类别分布等)
        |
        V
[ 三域决策 (3WD) ]
(根据粒球纯度/分布划分到 POS, NEG, BND)
        |
        |---------------------|
        |                     |
        V                     V
[ POS/NEG粒球处理 ]     [ BND粒球处理 ]
(快速分类决策)         (精细处理：如拆分粒球、
                      用原始点/复杂模型再决策)
        |                     |
        V                     V
        ---------------------
                    |
                    V
              最终分类结果
```



-   **阶段一：模型训练与规则构建**
    -   FT-Transformer → 表示空间
    -   （可选）SCAN-style 微调 → 增强语义聚类效果
    -   ADPGBG → 生成初始粒球
    -   粒球后处理 → 去重叠 / 相切调整 / 离群削弱
    -   粒球标注（prevalence 向量 / 主标签）
    -   三域决策规则构建（α / β 阈值 → RuleSet）
    -   输出结果就是 *已训练好的决策系统*
-   **阶段二：应用与决策**

    * 新数据输入 → FT-Transformer 得到嵌入
    * 应用 RuleSet → 判定 ACCEPT / REJECT / DEFER
    * 若 DEFER → 回退机制（局部 kNN / 再粒化）
    * 最终预测输出（类别 / 拒绝 / 延迟再判定）
