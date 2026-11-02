# 教程结果图

这里集中保存各个 `tutorials/*/README.md` 展示的结果图及来源说明。目录按“教程编号 / 示例数据集”组织，图片使用相对于仓库的路径，便于本地和 GitHub 浏览。

```text
result/
├── README.md
└── 07_biological/
    └── A549_DMEM/
        ├── README.md
        ├── manifest.json
        ├── pathway_umap.png
        ├── corum_similarity.png
        ├── embedding_clusters.png
        └── cluster3_enrichment.png
```

| 已有示例 | 教程 | 图像目录 |
|---|---|---|
| A549 DMEM 生物学解释 | [07_biological](../tutorials/07_biological/README.md) | [07_biological/A549_DMEM](07_biological/A549_DMEM/README.md) |

其他教程添加图像时，沿用 `result/<教程目录名>/<示例数据集>/`，并记录图片来源。只有已有真实图像时才添加示例，不用空文件代替结果。

从 `tutorials/07_biological/README.md` 引用本目录的示例：

```markdown
![A549 通路 UMAP](../../result/07_biological/A549_DMEM/pathway_umap.png)
```

完整分析运行的默认输出位置由各教程命令指定；当前 A549 示例写到 `outputs/07_biological/A549_DMEM/`。这里保留用于文档展示的图像副本，不会随分析重跑自动更新。复制新图片时，应同时更新来源记录。

Git 忽略规则允许本目录下的 PNG、JPEG、SVG、PDF、Markdown 和 JSON；数据、权重及日志仍受原有忽略规则控制。
