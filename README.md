# 量化交易学习路径

本仓库记录了我的量化交易学习历程，从Python基础到策略实战的完整学习轨迹。

---

## 📚 学习计划概览

参考文档：[study_plan.md](study_plan.md) | [study_plan.pdf](study_plan.pdf)

---

## 🎯 阶段一：Python基础与数据分析

**学习目标**：掌握Python基础语法和数据处理能力

### 核心文件

- [Pandas practice.ipynb](Pandas%20practice.ipynb) - Pandas数据处理练习
- [Seaborn practice.ipynb](Seaborn%20practice.ipynb) - 数据可视化练习
- [draft_code.ipynb](draft_code.ipynb) - Python基础练习代码

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
- [akshare_intro.ipynb](akshare_intro.ipynb) - AKShare库入门
- [akshare_quant_tutorial.ipynb](akshare_quant_tutorial.ipynb) - AKShare量化教程

#### 技术指标学习
- [moving_average.ipynb](moving_average.ipynb) - 移动平均线详解
- [simple_ma_example.py](simple_ma_example.py) - 移动平均线简单示例
- [RSI.ipynb](RSI.ipynb) - RSI指标详解与应用

### 关键概念
- 股票数据获取（AKShare）
- 技术指标：MA（移动平均线）、RSI（相对强弱指标）
- K线数据分析
- 成交量分析

---

## 🎯 阶段三：回测框架与单策略实战

**学习目标**：掌握Backtrader回测框架，实现单一策略回测

### 核心文件

#### 回测框架学习
- [backtrader_intro.ipynb](backtrader_intro.ipynb) - Backtrader框架入门

#### 单策略回测实战
- [ma_strategy.ipynb](ma_strategy.ipynb) - 双均线策略Jupyter实现
- [双均线策略量化回测教程.ipynb](双均线策略量化回测教程.ipynb) - 双均线策略完整教程
- [quant_strategies_tutorial.ipynb](quant_strategies_tutorial.ipynb) - 量化策略教程

### 关键概念
- Backtrader框架基础
- 策略编写：买入/卖出信号
- 回测指标：收益率、最大回撤、夏普比率
- 可视化分析

**学习笔记**：[note0830](note0830)

---

## 🎯 阶段四：策略优化与风险管理

**学习目标**：学习仓位管理、止盈止损、参数优化

### 核心文件

#### 风险管理
- [dynamic_position_size0829.py](dynamic_position_size0829.py) - 动态仓位管理

#### 策略优化与调参
- [1006.py](1006.py) - RSI策略优化版本
- [batch_backtest.py](batch_backtest.py) - 批量回测框架

#### 回测结果
- [backtest_results_RSIStrategy_20251006.csv](backtest_results_RSIStrategy_20251006.csv)
- [backtest_results_MartingaleStrategy_20251006.csv](backtest_results_MartingaleStrategy_20251006.csv)

### 关键概念
- 动态仓位管理
- 止盈止损机制
- 参数优化（Grid Search）
- 回撤与最大亏损的区别
- 风险收益比

**学习笔记**：[note1006.md](note1006.md)

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
- [万丰奥威多策略回测.py](万丰奥威多策略回测.py) - 个股多策略回测案例

#### 多股票系统
- [stock_scanner.py](stock_scanner.py) - 股票扫描与筛选
- [multi_stock_backtest.py](multi_stock_backtest.py) - 多股票回测系统
- [multi_stock_bt.py](multi_stock_bt.py) - 多股票综合评分策略（最新）

### 关键概念
- 马丁格尔策略（加仓策略）
- 多策略组合
- 股票筛选与评分系统
- 多股票轮动策略
- 综合信号过滤（RSI + MA + Volume）

**学习笔记**：[1011note.md](1011note.md)

---

## 🎯 阶段六：策略迭代与优化（进行中）

**学习目标**：从单一指标到多重滤波器，构建稳健的交易系统

### 最新文件
- [1011.py](1011.py) - 最新策略优化代码（2024-10-11）
- [1011note.md](1011note.md) - 最新学习笔记

### 核心改进方向

#### 1. 信号优化：单一指标 → 多重滤波器
```
买入逻辑：综合评分系统
- RSI权重: 40%
- MA权重: 30%
- Volume权重: 30%
```

#### 2. 选股系统
- 全市场股票扫描
- 多维度综合评分
- Top N选股策略

#### 3. 仓位控制策略
- 固定仓位上限
- 动态仓位调整
- 基于市场环境的自适应仓位

---

## 📊 学习总结

### 技术栈
- **数据获取**：AKShare, yfinance
- **数据处理**：Pandas, NumPy
- **数据可视化**：Matplotlib, Seaborn
- **回测框架**：Backtrader
- **性能分析**：QuantStats

### 策略演进
1. **简单MA策略** → 双均线交叉
2. **RSI策略** → RSI + 成交量过滤
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
├── README.md                           # 本文档
├── study_plan.md                       # 学习计划
│
├── 阶段一：Python基础/
│   ├── Pandas practice.ipynb
│   ├── Seaborn practice.ipynb
│   └── draft_code.ipynb
│
├── 阶段二：技术指标/
│   ├── akshare_intro.ipynb
│   ├── moving_average.ipynb
│   ├── simple_ma_example.py
│   └── RSI.ipynb
│
├── 阶段三：单策略回测/
│   ├── backtrader_intro.ipynb
│   ├── ma_strategy.ipynb
│   └── 双均线策略量化回测教程.ipynb
│
├── 阶段四：策略优化/
│   ├── dynamic_position_size0829.py
│   ├── batch_backtest.py
│   ├── 1006.py
│   └── note1006.md
│
├── 阶段五：高级策略/
│   ├── martingale_strategy.py
│   ├── multi-strategy.py
│   ├── stock_scanner.py
│   ├── multi_stock_backtest.py
│   └── multi_stock_bt.py
│
└── 阶段六：最新迭代/
    ├── 1011.py
    └── 1011note.md
```

---

## 🚀 下一步计划

1. **策略优化**
   - 完善多因子评分模型
   - 引入机器学习模型
   - 情绪指标整合

2. **风险管理**
   - 更精细的止损机制
   - 相关性分析（避免过度集中）
   - 压力测试

3. **实盘准备**
   - 模拟盘测试
   - 滑点与手续费建模
   - 实时数据接入

---

## 📝 学习资源

- **AKShare文档**：https://akshare.akfamily.xyz/
- **Backtrader文档**：https://www.backtrader.com/
- **QuantStats文档**：https://github.com/ranaroussi/quantstats

---

## ⚠️ 风险提示

本仓库所有代码仅供学习交流使用，不构成任何投资建议。

量化交易存在风险，投资需谨慎。

---

*最后更新：2024-10-11*
