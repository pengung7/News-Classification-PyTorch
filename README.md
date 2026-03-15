# Chinese-News-Classification-MultiModel
> 基于随机森林/FastText/BERT的中文新闻分类系统 | 实践项目（双基线模型对比+深度学习优化）

## 1. 项目背景
本项目为专业实践课程成果，在指导老师带领下与同学协作完成。核心目标是针对中文新闻资讯平台的自动分类需求，通过**双基线模型对比+进阶深度学习优化**的思路，探索文本分类任务的最优解决方案，最终实现高准确率、轻量化的新闻分类效果。

## 2. 核心技术栈
| 技术类型       | 具体工具/模型                     |
|----------------|-----------------------------------|
| 框架           | PyTorch 2.0                       |
| 基线模型       | 随机森林（Random Forest）、FastText |
| 进阶模型       | BERT-base-chinese、DistilBERT     |
| 数据处理       | Pandas、Jieba（分词）、Scikit-learn（TF-IDF/评估） |
| 可视化         | Matplotlib      |

## 3. 项目设计思路（核心：突出双基线）
为了科学验证不同算法的性能，项目采用「**基线对照→进阶优化→轻量化迭代**」的思路：
### 3.1 基线模型选择逻辑
- **随机森林（传统机器学习基线）**：
  - 特征工程：采用TF-IDF提取文本特征，覆盖传统机器学习的经典方案；
  - 选择原因：验证“纯特征工程+传统算法”的性能上限，作为最基础的对照基准。
- **FastText（轻量化深度学习基线）**：
  - 核心优势：专为文本分类设计，训练速度快、部署成本低，能捕捉词向量语义信息；
  - 选择原因：作为深度学习的基础对照，平衡“效果”与“效率”，为后续大模型优化提供参考。

### 3.2 模型性能对比（量化结果，替换成你的实际数值）
| 模型         | 分类准确率 | 推理速度（单条文本） | 模型体积 |
|--------------|------------|----------------------|----------|
| 随机森林     | 78%        | 0.02s                | <10MB    |
| FastText     | 85%        | 0.01s                | <50MB    |
| BERT         | 91%        | 0.15s                | 410MB    |
| DistilBERT   | 89%        | 0.05s                | 120MB    |

## 4. 项目结构（和你的本地文件夹完全对应）
```
├── 01-data/          # 新闻数据集（清洗/划分/特征提取）
├── 02-random_forest/ # 随机森林基线模型（训练/预测/评估）
├── 03-fasttext/     # FastText基线模型（训练/预测/评估）
├── 04-Bert/         # BERT进阶模型（微调/训练/预测）
├── 05-bert_distil/  # DistilBERT轻量化优化（蒸馏/评估）
└── README.md        # 项目说明文档
```

## 5. 快速运行指南
### 5.1 安装依赖
```bash
pip install torch transformers pandas scikit-learn jieba fasttext matplotlib
```

### 5.2 运行步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/你的用户名/Chinese-News-Classification-MultiModel.git
   cd Chinese-News-Classification-MultiModel
   ```
2. 运行基线模型（以随机森林为例）：
   ```bash
   cd 02-random_forest
   python train_rf.py
   ```
3. 运行FastText基线：
   ```bash
   cd ../03-fasttext
   python train_fasttext.py
   ```
4. 运行BERT模型：
   ```bash
   cd ../04-Bert
   python train_bert.py
   ```
