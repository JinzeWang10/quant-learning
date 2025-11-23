# 量化交易策略与回测系统

> 从 Python 基础到机器学习选股的完整量化交易学习与实战项目

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Backtrader](https://img.shields.io/badge/Backtrader-1.9+-green.svg)](https://www.backtrader.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目简介

本仓库记录了从零开始学习量化交易的完整历程，包含**基础学习**、**策略开发**、**回测验证**三大模块。核心成果是一个基于**随机森林机器学习**的沪深 300 选股策略，实现了系统化的量化交易流程。

**核心特色**:

- 🎯 **完整学习路径**: Python 基础 → 技术指标 → 回测框架 → 策略优化 → 机器学习
- 📊 **实战导向**: 所有策略均可回测验证，配有详细说明文档
- 🤖 **机器学习驱动**: 使用随机森林预测股票短期上涨概率

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **核心依赖**: backtrader, pandas, numpy, scikit-learn, baostock

### 安装依赖

```bash
pip install backtrader pandas numpy scikit-learn matplotlib seaborn baostock
```

### 运行最终策略（基于随机森林的沪深 300 选股）

```bash
# 1. 下载沪深300成分股数据（2022-01-01至今）
python download_stock_data_baostock.py
# 选择: 2 (下载指数成分股) → 2 (沪深300)

# 2. 运行回测（自动训练模型+回测）
python final_strategy.py

# 3. 查看结果
# - 终端输出: 特征分析、模型准确率、回测收益
# - 图表文件:
#   feature_correlation_heatmap.png  (特征相关性热图)
#   feature_importance.png           (特征重要性)
#   backtest_visualization.png       (资金曲线与基准对比)
```

---

## 📂 项目结构

```
quant/
│
├── README.md                              # 项目说明文档（本文件）
├── .gitignore                             # Git忽略规则
│
├── ==================== 📊 核心策略文件 ====================
│
├── final_strategy.py                      # ⭐ 最终策略：基于随机森林的沪深300选股
├── final_strategy.md                      # 最终策略的旧版说明文档
├── 策略说明文档.md                         # ⭐ 最新完整策略说明文档（v2.0）
├── hs300_stocks.py                        # 沪深300成分股列表（约300只）
├── download_stock_data_baostock.py        # 数据下载工具（baostock接口）
│
├── ==================== 📁 数据目录 ====================
│
├── stock_data/                            # 股票历史数据（CSV格式）
│   ├── 000001_平安银行.csv
│   ├── 600519_贵州茅台.csv
│   └── ...
│
├── test_data/                             # 测试数据
│
├── ==================== 📚 学习路径 ====================
│
├── intro_to_quant/                        # 入门学习资料
│   ├── study_plan.md                      # 学习计划总览
│   ├── python_intro.ipynb                 # Python基础语法
│   ├── Pandas practice.ipynb              # Pandas数据处理
│   ├── Seaborn practice.ipynb             # 数据可视化
│   ├── akshare_quant_tutorial.ipynb       # AKShare金融数据获取
│   └── backtrader_intro.ipynb             # Backtrader回测框架入门
│
├── basic_strategy/                        # 基础策略示例
│   ├── simple_ma_example.py               # 双均线策略（简单版）
│   ├── ma_strategy.ipynb                  # 双均线策略（完整版）
│   ├── moving_average.ipynb               # 移动平均线详解
│   └── RSI.ipynb                          # RSI指标详解与应用
│
├── more strategy+backtest/                # 进阶策略与实验
│   ├── 动态买卖数量0829.py                 # 动态仓位管理策略
│   ├── 动态买卖数量+多指标策略1006.py       # 动态仓位+多指标组合
│   ├── 多重滤波器买入评分1011.py            # 综合评分选股系统
│   ├── 1011note.md                        # 学习笔记
│   ├── martingale_simple.py               # 马丁格尔策略（简单版）
│   ├── martingale_strategy.py             # 马丁格尔策略（完整版）
│   ├── multi-strategy.py                  # 多策略组合回测
│   ├── multi_stock_bt.py                  # 多股票综合评分轮动
│   ├── backtest_rf_simple.py              # 随机森林策略（简化版）
│   └── backtest_rf_strategy.py            # 随机森林策略（早期版本）
│
├── ==================== 📊 可视化输出 ====================
│
├── feature_correlation_heatmap.png        # 特征相关性热图
├── feature_importance.png                 # 特征重要性排序
└── backtest_visualization.png             # 回测资金曲线图
```

---

## 🎯 核心策略：基于随机森林的沪深 300 选股

### 策略概览

**策略类型**: 机器学习选股 + 单股票集中持仓
**股票池**: 沪深 300 成分股（约 300 只大中盘股）
**预测目标**: 未来 5 日涨幅>3%的概率
**持仓方式**: 单股票 95%仓位（追求高确定性）

### 核心逻辑

```python
# 买入信号
每日开盘前:
    1. 计算所有沪深300股票的上涨概率（随机森林模型预测）
    2. 筛选概率 > 60% 的候选股票
    3. 选择概率最高的1只股票
    4. 使用95%资金买入（以开盘价成交）

# 卖出信号
每日开盘前:
    1. 计算当前持仓股票的上涨概率
    2. 若概率 < 50%，全部卖出（止损）
```

## 📈 学习路径

### 阶段一：Python 基础与数据分析

**学习目标**: 掌握 Python 基础语法和数据处理能力

**核心文件**:

- [python_intro.ipynb](intro_to_quant/python_intro.ipynb) - Python 基础语法
- [Pandas practice.ipynb](intro_to_quant/Pandas%20practice.ipynb) - Pandas 数据处理
- [Seaborn practice.ipynb](intro_to_quant/Seaborn%20practice.ipynb) - 数据可视化

**关键技能**:

- 数据结构：list, dict
- 控制流：for/while/if-else
- 数据处理：pandas.DataFrame, numpy.array
- 数据可视化：matplotlib, seaborn

---

### 阶段二：金融数据与技术指标

**学习目标**: 熟悉金融数据获取与技术指标计算

**核心文件**:

- [akshare_quant_tutorial.ipynb](intro_to_quant/akshare_quant_tutorial.ipynb) - 金融数据获取
- [moving_average.ipynb](basic_strategy/moving_average.ipynb) - 移动平均线详解
- [RSI.ipynb](basic_strategy/RSI.ipynb) - RSI 指标详解

**关键概念**:

- 股票数据获取：AKShare, BaoStock
- 技术指标：MA, RSI, MACD, Bollinger Bands
- K 线数据：OHLCV（开高低收量）

---

### 阶段三：回测框架与策略实战

**学习目标**: 掌握 Backtrader 框架，实现策略回测

**核心文件**:

- [backtrader_intro.ipynb](intro_to_quant/backtrader_intro.ipynb) - Backtrader 入门
- [simple_ma_example.py](basic_strategy/simple_ma_example.py) - 双均线策略
- [ma_strategy.ipynb](basic_strategy/ma_strategy.ipynb) - 完整回测流程

**关键概念**:

- Backtrader 架构：Strategy, Data Feed, Broker
- 买卖信号：self.buy(), self.sell()
- 回测指标：收益率、最大回撤、夏普比率

---

### 阶段四：策略优化与风险管理

**学习目标**: 学习仓位管理、止盈止损、参数优化

**核心文件**:

- [动态买卖数量 0829.py](more%20strategy+backtest/动态买卖数量0829.py) - 动态仓位管理
- [动态买卖数量+多指标策略 1006.py](more%20strategy+backtest/动态买卖数量+多指标策略1006.py) - 多指标组合

**关键概念**:

- 动态仓位：基于 RSI 强度调整买入数量
- 止盈止损：固定比例/移动止损
- 参数优化：Grid Search
- 多指标过滤：RSI + MA + Volume

---

### 阶段五：高级策略与多股票系统

**学习目标**: 实现马丁格尔策略、多股票轮动、综合评分

**核心文件**:

- [martingale_strategy.py](more%20strategy+backtest/martingale_strategy.py) - 马丁格尔策略
- [multi_stock_bt.py](more%20strategy+backtest/multi_stock_bt.py) - 多股票综合评分
- [多重滤波器买入评分 1011.py](more%20strategy+backtest/多重滤波器买入评分1011.py) - 综合评分系统

**关键概念**:

- 马丁格尔策略（加仓策略）
- 股票筛选与评分系统
- 多股票轮动
- 综合信号过滤

---

### 阶段六：机器学习量化策略（当前）

**学习目标**: 使用机器学习模型预测股票走势

**核心文件**:

- [final_strategy.py](final_strategy.py) - ⭐ 基于随机森林的沪深 300 选股策略
- [策略说明文档.md](策略说明文档.md) - 完整策略说明（v2.0）
- [hs300_stocks.py](hs300_stocks.py) - 沪深 300 成分股列表

**关键概念**:

- 随机森林分类器
- 特征工程：技术指标作为特征
- 标签定义：未来 5 日涨幅>3%
- 时间序列数据划分（避免未来数据泄露）
- 概率预测与阈值过滤

---

## 🛠️ 工具与技术栈

### 数据获取

- **BaoStock**: 免费 A 股数据接口（本项目主要使用）
- **AKShare**: 金融数据万能接口

### 数据处理

- **Pandas**: 表格数据处理
- **NumPy**: 数值计算

### 机器学习

- **scikit-learn**: 随机森林、XGBoost 等模型
- **特征工程**: 技术指标计算

### 回测框架

- **Backtrader**: 专业量化回测框架
- **支持功能**: 多股票、多策略、参数优化

### 可视化

- **Matplotlib**: 基础绘图
- **Seaborn**: 统计可视化

---

## 📊 回测指标说明

### 收益指标

- **总收益率**: (最终资金 - 初始资金) / 初始资金
- **年化收益率**: 总收益率 × (365 / 回测天数)

### 风险指标

- **最大回撤**: max((峰值 - 当前值) / 峰值)
- **夏普比率**: (年化收益率 - 无风险利率) / 收益率标准差
- **波动率**: 日收益率标准差 × √252

### 交易指标

- **胜率**: 盈利交易次数 / 总交易次数
- **盈亏比**: 平均盈利 / 平均亏损
- **平均持仓天数**: 总持仓天数 / 交易次数

---

## 🔧 优化方向

### 1. 特征工程优化

- [ ] 增加基本面特征（PE、PB、ROE、营收增长率）
- [ ] 增加资金流特征（北向资金、融资融券、大单流入）
- [ ] 增加情绪指标（换手率、振幅、涨停板数量）
- [ ] 特征选择（RFE、LASSO、基于重要性筛选）

### 2. 模型优化

- [ ] 尝试其他算法（XGBoost, LightGBM, 神经网络）
- [ ] 超参数调优（网格搜索、贝叶斯优化）
- [ ] 集成学习（多模型投票/加权）
- [ ] 滚动训练（每月重新训练模型）

### 3. 策略优化

- [ ] 多股持仓（Top3 持仓，每只 30%）
- [ ] 动态仓位（根据概率高低调整仓位）
- [ ] 止损止盈（固定比例/移动止损）
- [ ] 行业轮动（每个行业选 1 只）

### 4. 风险控制

- [ ] 最大回撤止损（回撤>20%暂停交易）
- [ ] 单日最大亏损限制（单日亏损>5%停止）
- [ ] 波动率过滤（波动率过高时不买入）

---

## ⚠️ 风险提示

**本项目仅供学习研究使用，不构成任何投资建议。**

- ❌ 历史回测收益不代表未来收益
- ❌ 实盘交易存在滑点、流动性、涨跌停等因素
- ❌ 量化策略存在失效风险
- ❌ 股市有风险，投资需谨慎

**模型有效性问题**:

- ⚠️ 过拟合风险：训练集准确率过高，测试集表现差
- ⚠️ 市场环境变化：历史规律在未来可能失效
- ⚠️ 特征相关性：高相关特征可能导致多重共线性

**实盘差异**:

- ⚠️ 滑点：实际成交价可能偏离开盘价
- ⚠️ 冲击成本：大单会推动价格
- ⚠️ 涨跌停：无法在涨停买入、跌停卖出
- ⚠️ 时间延迟：信号产生到下单有延迟

---

## 📚 参考资料

### 官方文档

- [Backtrader 官方文档](https://www.backtrader.com/)
- [AKShare 文档](https://akshare.akfamily.xyz/)
- [BaoStock 文档](http://baostock.com/)
- [scikit-learn 文档](https://scikit-learn.org/)

### 推荐书籍

- 《Python 金融大数据分析》
- 《量化交易：如何建立自己的算法交易事业》
- 《机器学习与量化投资》

---

## 📄 License

MIT License - 仅供学习研究使用，不承担任何投资风险。
