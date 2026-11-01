# 国家股权与实际资源约束：统一理论框架（NCS-RR）

| 路径 | 内容 |
|---|---|
| `docs/v0.2/国家股权与实际资源约束_统一理论框架_v0.2.md` / `.docx` | v0.2 增补稿：P9—P11 的解析证明与 HANK 数值实验（v0.1 原文不改，v0.1 保存在本地工作目录） |
| `docs/v0.2/reference_zh.docx` | pandoc 转 Word 用的样式模板（中文主题字体） |
| `code/ncsrr/` | 精简版模型 U：一资产 HANK（`household.py`）+ 模块化方程组（`model.py`） |
| `code/ncsrr/p11_minsky.py` | P11 Minsky 式地方信用周期 |
| `code/ncsrr/p9_constraints.py`, `p9_post.py` | P9 多重实际约束（闲置资源、外部、碳） |
| `code/ncsrr/p10_structure.py` | P10 绿色投资的最优货币股权占比 |
| `code/ncsrr/output/` | 全部表格（CSV）与图 |

复现：

```
pip install numpy scipy matplotlib
cd code/ncsrr
python3 p11_minsky.py
python3 p9_constraints.py && python3 p9_post.py
python3 p10_structure.py
cd ../../docs/v0.2 && pandoc 国家股权与实际资源约束_统一理论框架_v0.2.md -o 国家股权与实际资源约束_统一理论框架_v0.2.docx --resource-path=. --reference-doc=reference_zh.docx
```

首次运行时会校准家庭块（约 70 秒），之后读取 `output/cache/`。
