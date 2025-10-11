# 量化交易学习路径

本仓库记录了我的量化交易学习历程，从 Python 基础到策略实战的完整学习轨迹。

---

## 📚 学习计划概览

参考文档：[study_plan.md](study_plan.md)

---

## 🎯 阶段一：Python 基础与数据分析

**学习目标**：掌握 Python 基础语法和数据处理能力

### 核心文件

- [python_intro.ipynb](python_intro.ipynb) - Python 基础语法入门
- [Pandas practice.ipynb](Pandas%20practice.ipynb) - Pandas 数据处理练习
- [Seaborn practice.ipynb](Seaborn%20practice.ipynb) - 数据可视化练习

### 关键技能

- 数据结构：list, dict
- 控制流：for/while/if-else
- 数据处理：pandas, numpy
- 数据可视化：matplotlib, seaborn

---

## 🎯 阶段二：金融数据获取与技术指标

**学习目标**：熟悉金融数据获取与技术指标计算

### 核心文件

#### 数据获取

- [akshare_quant_tutorial.ipynb](akshare_quant_tutorial.ipynb) - AKShare 量化教程

#### 技术指标学习

- [moving_average.ipynb](moving_average.ipynb) - 移动平均线详解
- [simple_ma_example.py](simple_ma_example.py) - 移动平均线简单示例
- [RSI.ipynb](RSI.ipynb) - RSI 指标详解与应用

### 关键概念

- 股票数据获取（AKShare）
- 技术指标：MA（移动平均线）、RSI（相对强弱指标）
- K 线数据分析
- 成交量分析

---

## 🎯 阶段三：回测框架与单策略实战

**学习目标**：掌握 Backtrader 回测框架，实现单一策略回测

### 核心文件

#### 回测框架学习

- [backtrader_intro.ipynb](backtrader_intro.ipynb) - Backtrader 框架入门

#### 单策略回测实战

- [ma_strategy.ipynb](ma_strategy.ipynb) - 双均线策略 Jupyter 实现
- [双均线策略量化回测教程.ipynb](双均线策略量化回测教程.ipynb) - 双均线策略完整教程

### 关键概念

- Backtrader 框架基础
- 策略编写：买入/卖出信号
- 回测指标：收益率、最大回撤、夏普比率
- 可视化分析

---

## 🎯 阶段四：策略优化与风险管理

**学习目标**：学习仓位管理、止盈止损、参数优化

### 核心文件

#### 动态仓位管理

- [动态买卖数量 0829.py](动态买卖数量0829.py) - 动态仓位与买卖数量管理

#### 策略优化与调参

- [动态买卖数量+多指标策略 1006.py](动态买卖数量+多指标策略1006.py) - 动态仓位 + 多指标组合策略

### 关键概念

- 动态仓位管理：基于 RSI 强度动态调整买入数量
- 止盈止损机制
- 参数优化（Grid Search）
- 回撤与最大亏损的区别
- 风险收益比
- 多指标组合过滤（RSI + MA + Volume）

```python
# Grid Search调参示例
period_list = [5, 10, 15, 20, 60]
ratio_list = [1 + 0.1*i for i in range(5)]

for p in period_list:
    for r in ratio_list:
        backtest(p, r)
```

---

## 🎯 阶段五：高级策略与多股票系统

**学习目标**：实现马丁格尔策略、多股票轮动、综合评分系统

### 核心文件

#### 马丁格尔策略

- [martingale_simple.py](martingale_simple.py) - 马丁格尔策略简单版
- [martingale_strategy.py](martingale_strategy.py) - 马丁格尔策略完整版

#### 多策略系统

- [multi-strategy.py](multi-strategy.py) - 多策略组合回测

#### 多股票系统

- [multi_stock_bt.py](multi_stock_bt.py) - 多股票综合评分策略

### 关键概念

- 马丁格尔策略（加仓策略）
- 多策略组合
- 股票筛选与评分系统
- 多股票轮动策略
- 综合信号过滤（RSI + MA + Volume）

**学习笔记**：[1011note.md](1011note.md)

---

## 🎯 阶段六：多重滤波器与综合评分系统（进行中）

**学习目标**：从单一指标到多重滤波器，构建稳健的交易系统

### 最新文件

- [多重滤波器买入评分 1011.py](多重滤波器买入评分1011.py) - 多重滤波器综合评分策略（2024-10-11）
- [1011note.md](1011note.md) - 最新学习笔记

### 核心改进方向

#### 1. 信号优化：单一指标 → 多重滤波器

```
买入逻辑：综合评分系统
- RSI指标得分（权重：40%）
  - RSI < 30: 超卖信号，高分
  - 考虑RSI背离
- MA趋势得分（权重：30%）
  - 短期均线上穿长期均线：看涨信号
  - 价格相对均线位置
- Volume放量得分（权重：30%）
  - 成交量突破均量：资金流入信号
  - 放量配合价格上涨

卖出逻辑：多重触发条件
- RSI超买（RSI > 70）
- 跌破移动平均线
- 止损止盈触发
```

#### 2. 选股系统

- 全市场股票扫描
- 多维度综合评分（0-100 分制）
- 只买入评分>=70 的股票
- 动态评分实时更新

#### 3. 仓位控制策略

- 固定仓位上限
- 单股固定份额买入
- 基于综合得分的资金分配
- 风险分散：支持多股同时持仓

---

## 📊 学习总结

### 技术栈

- **数据获取**：AKShare, yfinance
- **数据处理**：Pandas, NumPy
- **数据可视化**：Matplotlib, Seaborn
- **回测框架**：Backtrader
- **性能分析**：QuantStats

### 策略演进

1. **简单 MA 策略** → 双均线交叉
2. **RSI 策略** → RSI + 成交量过滤
3. **单股策略** → 多股票轮动
4. **单一指标** → 综合评分系统
5. **固定仓位** → 动态仓位管理

### 关键指标理解

- **最大回撤 vs 最大亏损**：最大回撤从资产峰值计算，最大亏损从初始资金计算
- **夏普比率**：衡量风险调整后收益
- **胜率 vs 盈亏比**：不仅要看胜率，更要看盈亏比

---

## 📁 目录结构

```
quant/
├── README.md                                    # 本文档
├── study_plan.md                                # 学习计划
│
├── 阶段一：Python基础/
│   ├── python_intro.ipynb                       # Python基础语法
│   ├── Pandas practice.ipynb                    # Pandas练习
│   └── Seaborn practice.ipynb                   # 可视化练习
│
├── 阶段二：技术指标/
│   ├── akshare_quant_tutorial.ipynb             # AKShare教程
│   ├── moving_average.ipynb                     # 移动平均线
│   ├── simple_ma_example.py                     # MA简单示例
│   └── RSI.ipynb                                # RSI指标
│
├── 阶段三：单策略回测/
│   ├── backtrader_intro.ipynb                   # Backtrader入门
│   ├── ma_strategy.ipynb                        # 双均线策略
│   └── 双均线策略量化回测教程.ipynb                # 双均线教程
│
├── 阶段四：策略优化/
│   ├── 动态买卖数量0829.py                       # 动态仓位管理
│   └── 动态买卖数量+多指标策略1006.py              # 动态仓位+多指标
│
├── 阶段五：高级策略/
│   ├── martingale_simple.py                     # 马丁格尔简单版
│   ├── martingale_strategy.py                   # 马丁格尔完整版
│   ├── multi-strategy.py                        # 多策略组合
│   └── multi_stock_bt.py                        # 多股票综合评分
│
└── 阶段六：多重滤波器/
    ├── 多重滤波器买入评分1011.py                  # 多重滤波器策略
    └── 1011note.md                              # 学习笔记
```

---

## 📝 学习资源

- **AKShare 文档**：https://akshare.akfamily.xyz/
- **Backtrader 文档**：https://www.backtrader.com/
- **QuantStats 文档**：https://github.com/ranaroussi/quantstats

---

## ⚠️ 风险提示

本仓库所有代码仅供学习交流使用，不构成任何投资建议。

量化交易存在风险，投资需谨慎。
