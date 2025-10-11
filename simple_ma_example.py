# 移动平均线(Moving Average) - 初学者友好版
# 用最简单的方式解释移动平均线

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 移动平均线(Moving Average) 初学者教程 ===\n")

# 1. 什么是移动平均线？
print("1️⃣ 什么是移动平均线？")
print("="*50)

print("""
🔍 移动平均线的定义：
移动平均线 = 过去N天收盘价的平均值

📊 举个例子：
假设平安银行过去5天的收盘价是：
- 第1天：10.5元
- 第2天：10.8元  
- 第3天：11.2元
- 第4天：10.9元
- 第5天：11.5元

那么5日移动平均线(MA5) = (10.5+10.8+11.2+10.9+11.5) ÷ 5 = 10.98元

第6天价格变成12元时：
MA5 = (10.8+11.2+10.9+11.5+12) ÷ 5 = 11.28元

注意：窗口"移动"了，去掉了第1天的数据，加入了第6天的数据
""")

# 2. 创建示例数据
print("\n2️⃣ 创建示例数据")
print("="*50)

# 模拟股票价格数据
dates = pd.date_range('2023-01-01', periods=30, freq='D')
np.random.seed(42)  # 固定随机种子，确保结果可重现

# 生成模拟价格数据（有趋势的随机数据）
base_price = 10
trend = np.linspace(0, 2, 30)  # 上升趋势
noise = np.random.normal(0, 0.3, 30)  # 随机波动
prices = base_price + trend + noise

# 创建DataFrame
stock_data = pd.DataFrame({
    '日期': dates,
    '收盘价': prices
})
stock_data.set_index('日期', inplace=True)

print("模拟股票数据（前10天）：")
print(stock_data.head(10).round(2))

# 3. 计算移动平均线
print("\n3️⃣ 计算移动平均线")
print("="*50)

# 计算不同周期的移动平均线
stock_data['MA5'] = stock_data['收盘价'].rolling(window=5).mean()
stock_data['MA10'] = stock_data['收盘价'].rolling(window=10).mean()
stock_data['MA20'] = stock_data['收盘价'].rolling(window=20).mean()

print("包含移动平均线的数据（前15天）：")
print(stock_data.head(15).round(2))

# 4. 解释不同周期的移动平均线
print("\n4️⃣ 不同周期移动平均线的特点")
print("="*50)

print("""
📈 移动平均线的特点：

🔴 MA5 (5日移动平均线)：
   - 反应最灵敏，跟随价格变化快
   - 适合短期交易
   - 容易产生假信号

🟠 MA10 (10日移动平均线)：
   - 中等灵敏度
   - 平衡短期和长期趋势
   - 常用作短期支撑阻力

🔵 MA20 (20日移动平均线)：
   - 平滑效果好，显示中期趋势
   - 常用作中期支撑阻力
   - 适合趋势跟踪

🟣 MA60 (60日移动平均线)：
   - 最平滑，显示长期趋势
   - 适合判断大趋势方向
   - 滞后性最强
""")

# 5. 可视化展示
print("\n5️⃣ 移动平均线可视化")
print("="*50)

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# 上图：价格和所有移动平均线
ax1.plot(stock_data.index, stock_data['收盘价'], label='收盘价', linewidth=2, color='black', marker='o', markersize=4)
ax1.plot(stock_data.index, stock_data['MA5'], label='MA5', linewidth=2, color='red')
ax1.plot(stock_data.index, stock_data['MA10'], label='MA10', linewidth=2, color='orange')
ax1.plot(stock_data.index, stock_data['MA20'], label='MA20', linewidth=2, color='blue')

ax1.set_title('移动平均线对比图', fontsize=16, fontweight='bold')
ax1.set_ylabel('价格 (元)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下图：价格与MA20的对比
ax2.plot(stock_data.index, stock_data['收盘价'], label='收盘价', linewidth=2, color='black', marker='o', markersize=4)
ax2.plot(stock_data.index, stock_data['MA20'], label='MA20', linewidth=2, color='blue')

# 填充价格与MA20之间的区域
ax2.fill_between(stock_data.index, stock_data['收盘价'], stock_data['MA20'], 
                 where=(stock_data['收盘价'] >= stock_data['MA20']), 
                 color='red', alpha=0.3, label='价格>MA20')
ax2.fill_between(stock_data.index, stock_data['收盘价'], stock_data['MA20'], 
                 where=(stock_data['收盘价'] < stock_data['MA20']), 
                 color='green', alpha=0.3, label='价格<MA20')

ax2.set_title('价格与MA20的关系', fontsize=16, fontweight='bold')
ax2.set_ylabel('价格 (元)')
ax2.set_xlabel('日期')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 金叉死叉概念
print("\n6️⃣ 金叉死叉概念")
print("="*50)

# 计算金叉死叉信号
stock_data['MA5_MA20_Signal'] = 0
stock_data.loc[stock_data['MA5'] > stock_data['MA20'], 'MA5_MA20_Signal'] = 1
stock_data.loc[stock_data['MA5'] < stock_data['MA20'], 'MA5_MA20_Signal'] = -1

# 找到金叉死叉点
golden_cross = (stock_data['MA5_MA20_Signal'] == 1) & (stock_data['MA5_MA20_Signal'].shift(1) == -1)
death_cross = (stock_data['MA5_MA20_Signal'] == -1) & (stock_data['MA5_MA20_Signal'].shift(1) == 1)

print("""
🎯 金叉死叉概念：

🟢 金叉 (Golden Cross)：
   - 短期均线上穿长期均线
   - 例如：MA5上穿MA20
   - 通常认为是买入信号
   - 表示短期趋势转强

🔴 死叉 (Death Cross)：
   - 短期均线下穿长期均线
   - 例如：MA5下穿MA20
   - 通常认为是卖出信号
   - 表示短期趋势转弱
""")

# 显示信号统计
golden_cross_dates = stock_data[golden_cross].index
death_cross_dates = stock_data[death_cross].index

print(f"\n📊 信号统计：")
print(f"金叉次数：{len(golden_cross_dates)}")
print(f"死叉次数：{len(death_cross_dates)}")

if len(golden_cross_dates) > 0:
    print(f"金叉日期：{[d.strftime('%m-%d') for d in golden_cross_dates]}")
if len(death_cross_dates) > 0:
    print(f"死叉日期：{[d.strftime('%m-%d') for d in death_cross_dates]}")

# 7. 金叉死叉可视化
fig, ax = plt.subplots(figsize=(15, 8))

# 绘制价格和均线
ax.plot(stock_data.index, stock_data['收盘价'], label='收盘价', linewidth=2, color='black', alpha=0.7)
ax.plot(stock_data.index, stock_data['MA5'], label='MA5', linewidth=2, color='red')
ax.plot(stock_data.index, stock_data['MA20'], label='MA20', linewidth=2, color='blue')

# 标记金叉死叉
if len(golden_cross_dates) > 0:
    ax.scatter(golden_cross_dates, stock_data.loc[golden_cross_dates, '收盘价'], 
               color='green', s=150, marker='^', label='金叉', zorder=5)
if len(death_cross_dates) > 0:
    ax.scatter(death_cross_dates, stock_data.loc[death_cross_dates, '收盘价'], 
               color='red', s=150, marker='v', label='死叉', zorder=5)

ax.set_title('金叉死叉信号图', fontsize=16, fontweight='bold')
ax.set_ylabel('价格 (元)')
ax.set_xlabel('日期')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# 8. 实际应用建议
print("\n7️⃣ 实际应用建议")
print("="*50)

print("""
💡 移动平均线的实际应用：

✅ 买入时机：
1. 价格突破MA20向上
2. MA5上穿MA20（金叉）
3. 价格回调到MA20获得支撑
4. 多头排列（MA5>MA10>MA20）

❌ 卖出时机：
1. 价格跌破MA20向下
2. MA5下穿MA20（死叉）
3. 价格反弹到MA20遇到阻力
4. 空头排列（MA5<MA10<MA20）

⚠️ 注意事项：
1. 不要单独使用移动平均线
2. 结合成交量确认信号
3. 设置止损控制风险
4. 在趋势明显的市场中使用
5. 避免在震荡市频繁交易
""")

print("\n" + "="*50)
print("🎉 移动平均线教程完成！")
print("="*50)
print("""
📚 学习要点总结：
1. 移动平均线是平滑价格波动的技术指标
2. 短期均线反应灵敏，长期均线显示趋势
3. 金叉死叉提供买卖信号
4. 需要结合其他指标和风险管理使用
5. 不同股票的特性会影响均线效果

🚀 下一步学习：
- RSI相对强弱指数
- MACD指标
- 布林带
- 成交量分析
- 量化策略回测
""") 