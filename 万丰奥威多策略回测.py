#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
万丰奥威多策略量化回测
使用akshare获取数据，backtrader进行多策略回测和可视化
"""

import akshare as ak
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data(symbol='002085', start_date='20230101', end_date='20241231'):
    """
    使用akshare获取股票数据
    万丰奥威股票代码：002085
    """
    try:
        print(f"正在获取 {symbol} 的股票数据...")
        stock_data = ak.stock_zh_a_hist(
            symbol=symbol, 
            period='daily',
            start_date=start_date, 
            end_date=end_date, 
            adjust='qfq'
        )
        
        # 重命名列以符合backtrader要求
        stock_data = stock_data[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
        stock_data.columns = ['datetime', 'open', 'close', 'high', 'low', 'volume']
        stock_data['datetime'] = pd.to_datetime(stock_data['datetime'])
        stock_data.set_index('datetime', inplace=True)
        stock_data = stock_data.sort_index()
        
        # 确保数据类型正确
        for col in ['open', 'close', 'high', 'low', 'volume']:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
        
        # 删除包含NaN的行
        stock_data = stock_data.dropna()
        
        print(f"成功获取 {symbol} 从 {start_date} 到 {end_date} 的数据，共 {len(stock_data)} 条记录")
        print(f"数据范围: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")
        return stock_data
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

# 定义多个策略类
class RSIStrategy(bt.Strategy):
    """RSI策略：超买超卖策略"""
    params = (
        ('rsi_period', 14),      # RSI周期
        ('oversold', 30),        # 超卖阈值
        ('overbought', 70),      # 超买阈值
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi_period)
        self.order = None
        self.trades = []  # 记录交易
        
    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None  # 重置订单状态
        
    def next(self):
        if self.order:
            return
            
        if not self.position:  # 没有持仓
            if self.rsi[0] < self.params.oversold and self.rsi[0] > 0:  # RSI超卖，买入信号
                # 检查资金是否足够
                if self.broker.getcash() >= self.datas[0].close[0] * 100:
                    self.order = self.buy(size=100)
                    self.trades.append({
                        'date': self.datas[0].datetime.date(0),
                        'type': '买入',
                        'price': self.datas[0].close[0],
                        'rsi': self.rsi[0]
                    })

        else:  # 已有持仓
            if self.rsi[0] > self.params.overbought:  # RSI超买，卖出信号
                self.order = self.sell(size=self.position.size)  # 卖出全部持仓
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '卖出',
                    'price': self.datas[0].close[0],
                    'rsi': self.rsi[0]
                })


class MACDStrategy(bt.Strategy):
    """MACD策略：趋势跟踪策略"""
    params = (
        ('fast_period', 12),     # 快线周期
        ('slow_period', 26),     # 慢线周期
        ('signal_period', 9),    # 信号线周期
    )
    
    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period
        )
        self.order = None
        self.trades = []  # 记录交易
        
    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None  # 重置订单状态
        
    def next(self):
        if self.order:
            return
            
        if not self.position:  # 没有持仓
            if (self.macd.macd[0] > self.macd.signal[0] and 
                self.macd.macd[-1] <= self.macd.signal[-1] and
                len(self.macd.macd) > 1):
                # MACD线上穿信号线，买入
                self.order = self.buy(size=100)
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '买入',
                    'price': self.datas[0].close[0],
                    'macd': self.macd.macd[0],
                    'signal': self.macd.signal[0]
                })
        else:  # 已有持仓
            if (self.macd.macd[0] < self.macd.signal[0] and 
                self.macd.macd[-1] >= self.macd.signal[-1] and
                len(self.macd.macd) > 1):
                # MACD线下穿信号线，卖出
                self.order = self.sell(size=self.position.size)  # 卖出全部持仓
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '卖出',
                    'price': self.datas[0].close[0],
                    'macd': self.macd.macd[0],
                    'signal': self.macd.signal[0]
                })

class BollingerBandsStrategy(bt.Strategy):
    """布林带策略：波动率策略"""
    params = (
        ('bb_period', 20),       # 布林带周期
        ('bb_dev', 2),           # 标准差倍数
    )
    
    def __init__(self):
        self.bb = bt.indicators.BollingerBands(
            self.datas[0].close, 
            period=self.params.bb_period, 
            devfactor=self.params.bb_dev
        )
        self.order = None
        self.trades = []  # 记录交易
        
    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None  # 重置订单状态
        
    def next(self):
        if self.order:
            return
            
        if not self.position:  # 没有持仓
            if (self.datas[0].close[0] < self.bb.lines.bot[0] and 
                self.bb.lines.bot[0] > 0):  # 价格触及下轨，买入
                self.order = self.buy(size=100)
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '买入',
                    'price': self.datas[0].close[0],
                    'bb_bot': self.bb.lines.bot[0]
                })
        else:  # 已有持仓
            if (self.datas[0].close[0] > self.bb.lines.top[0] and 
                self.bb.lines.top[0] > 0):  # 价格触及上轨，卖出
                self.order = self.sell(size=self.position.size)  # 卖出全部持仓
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '卖出',
                    'price': self.datas[0].close[0],
                    'bb_top': self.bb.lines.top[0]
                })

class DualMAStrategy(bt.Strategy):
    """双均线策略：经典趋势策略"""
    params = (
        ('fast_period', 5),      # 快线周期
        ('slow_period', 20),     # 慢线周期
    )
    
    def __init__(self):
        self.ma_fast = bt.indicators.SMA(self.datas[0].close, period=self.params.fast_period)
        self.ma_slow = bt.indicators.SMA(self.datas[0].close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)
        self.order = None
        self.trades = []  # 记录交易
        
    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None  # 重置订单状态
        
    def next(self):
        if self.order:
            return
            
        if not self.position:  # 没有持仓
            if (self.crossover > 0 and 
                self.ma_fast[0] > 0 and self.ma_slow[0] > 0):  # 金叉：快线上穿慢线
                self.order = self.buy(size=100)
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '买入',
                    'price': self.datas[0].close[0],
                    'ma_fast': self.ma_fast[0],
                    'ma_slow': self.ma_slow[0]
                })
        else:  # 已有持仓
            if (self.crossover < 0 and 
                self.ma_fast[0] > 0 and self.ma_slow[0] > 0):  # 死叉：快线下穿慢线
                self.order = self.sell(size=self.position.size)  # 卖出全部持仓
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': '卖出',
                    'price': self.datas[0].close[0],
                    'ma_fast': self.ma_fast[0],
                    'ma_slow': self.ma_slow[0]
                })

class BuyAndHoldStrategy(bt.Strategy):
    """买入持有策略：基准策略"""
    
    def __init__(self):
        self.order = None
        self.bought = False
        self.trades = []  # 记录交易
        
    def next(self):
        if self.order:
            return
            
        if not self.position and not self.bought:  # 没有持仓，在第一个交易日买入
            self.order = self.buy(size=100)  # 买入100股
            self.bought = True
            self.trades.append({
                'date': self.datas[0].datetime.date(0),
                'type': '买入',
                'price': self.datas[0].close[0]
            })

def run_multi_strategy_backtest(data, initial_cash=1000000):
    """
    运行多策略回测
    """
    # 定义策略列表
    strategies = [
        ('买入持有策略', BuyAndHoldStrategy, {}),
        ('RSI策略', RSIStrategy, {'rsi_period': 14, 'oversold': 30, 'overbought': 70}),
        ('MACD策略', MACDStrategy, {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        ('布林带策略', BollingerBandsStrategy, {'bb_period': 20, 'bb_dev': 2}),
        ('双均线策略', DualMAStrategy, {'fast_period': 5, 'slow_period': 20}),
    ]
    
    # 存储每个策略的结果
    strategy_results = []
    
    print("开始多策略回测...")
    print("=" * 50)
    
    for name, strategy_class, params in strategies:
        print(f"正在运行 {name}...")
        
        try:
            # 创建Cerebro引擎
            cerebro = bt.Cerebro()
            
            # 添加数据
            datafeed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(datafeed)
            
            # 设置初始资金
            cerebro.broker.setcash(initial_cash)
            
            # 设置手续费
            cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
            
            # 添加策略
            cerebro.addstrategy(strategy_class, **params)
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # 运行回测
            results = cerebro.run()
            strategy = results[0]
            
            # 获取分析结果
            sharpe = strategy.analyzers.sharpe.get_analysis()
            drawdown = strategy.analyzers.drawdown.get_analysis()
            returns = strategy.analyzers.returns.get_analysis()
            trades = strategy.analyzers.trades.get_analysis()
            
            # 计算交易统计
            total_trades = trades.get('total', {}).get('total', 0) if trades else 0
            won_trades = trades.get('won', {}).get('total', 0) if trades else 0
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            final_value = strategy.broker.getvalue()
            total_return = (final_value - initial_cash) / initial_cash * 100
            
            strategy_results.append({
                'name': name,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe.get('sharperatio', 0),
                'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
                'annual_return': returns.get('rnorm100', 0),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'cerebro': cerebro,
                'strategy': strategy,
                'trades': strategy.trades  # 保存交易记录
            })
            
            print(f"  {name} 完成 - 总收益率: {total_return:.2f}%, 夏普比率: {sharpe.get('sharperatio', 0):.4f}, 交易次数: {total_trades}")
            
        except Exception as e:
            print(f"  {name} 运行失败: {e}")
            # 添加默认结果
            strategy_results.append({
                'name': name,
                'final_value': initial_cash,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'annual_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'cerebro': None,
                'strategy': None,
                'trades': []
            })
    
    return strategy_results

def visualize_results(data, strategy_results):
    """
    可视化回测结果
    """
    try:
        # 检查数据是否为空
        if not strategy_results:
            print("警告：没有策略结果数据")
            return None
            
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('万丰奥威多策略回测结果分析', fontsize=16)
        
        # 1. 股价走势图
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['close'], label='收盘价', color='black', linewidth=1)
        ax1.set_title('万丰奥威股价走势')
        ax1.set_ylabel('价格 (元)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 策略收益率对比
        ax2 = axes[0, 1]
        names = [result['name'] for result in strategy_results]
        returns = [result['total_return'] for result in strategy_results]
        colors = ['red', 'blue', 'green', 'orange']
        
        # 确保颜色列表长度匹配
        while len(colors) < len(names):
            colors.extend(['purple', 'brown', 'pink', 'gray'])
        
        bars = ax2.bar(names, returns, color=colors[:len(names)], alpha=0.7)
        ax2.set_title('各策略总收益率对比')
        ax2.set_ylabel('收益率 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, return_val in zip(bars, returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{return_val:.1f}%', ha='center', va='bottom')
        
        # 3. 夏普比率对比
        ax3 = axes[1, 0]
        sharpe_ratios = [result['sharpe_ratio'] for result in strategy_results]
        
        bars = ax3.bar(names, sharpe_ratios, color=colors[:len(names)], alpha=0.7)
        ax3.set_title('各策略夏普比率对比')
        ax3.set_ylabel('夏普比率')
        ax3.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, sharpe in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{sharpe:.3f}', ha='center', va='bottom')
        
        # 4. 最大回撤对比
        ax4 = axes[1, 1]
        drawdowns = [result['max_drawdown'] for result in strategy_results]
        
        bars = ax4.bar(names, drawdowns, color=colors[:len(names)], alpha=0.7)
        ax4.set_title('各策略最大回撤对比')
        ax4.set_ylabel('最大回撤 (%)')
        ax4.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, dd in zip(bars, drawdowns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height - 0.5,
                    f'{dd:.1f}%', ha='center', va='top')
        
        plt.tight_layout()
        plt.show()
        
        # 创建详细结果表格
        results_df = pd.DataFrame(strategy_results)
        
        # 检查必要的列是否存在
        required_columns = ['name', 'total_return', 'sharpe_ratio', 'max_drawdown', 
                           'annual_return', 'total_trades', 'win_rate']
        
        available_columns = [col for col in required_columns if col in results_df.columns]
        
        if len(available_columns) >= 4:  # 至少要有基本的列
            results_df = results_df[available_columns]
            
            # 创建中文列名映射
            column_mapping = {
                'name': '策略名称',
                'total_return': '总收益率(%)',
                'sharpe_ratio': '夏普比率',
                'max_drawdown': '最大回撤(%)',
                'annual_return': '年化收益率(%)',
                'total_trades': '总交易次数',
                'win_rate': '胜率(%)'
            }
            
            # 重命名列
            results_df.columns = [column_mapping.get(col, col) for col in available_columns]
            
            print("\n" + "=" * 60)
            print("详细回测结果对比")
            print("=" * 60)
            print(results_df.round(2).to_string(index=False))
        else:
            print("警告：结果数据列不完整")
            print("可用列：", list(results_df.columns))
        
        return results_df
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("错误详情：")
        import traceback
        traceback.print_exc()
        return None

def plot_individual_strategy(data, strategy_results):
    """
    绘制每个策略的详细图表
    """
    try:
        if not strategy_results:
            print("警告：没有策略结果数据")
            return
            
        for result in strategy_results:
            name = result['name']
            trades = result.get('trades', [])
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'{name} - 详细分析', fontsize=16)
            
            # 第一个子图：价格和指标
            ax1.plot(data.index, data['close'], label='收盘价', color='black', alpha=0.7)
            
            # 标注买卖点
            if trades:
                buy_dates = []
                buy_prices = []
                sell_dates = []
                sell_prices = []
                
                for trade in trades:
                    if trade['type'] == '买入':
                        buy_dates.append(trade['date'])
                        buy_prices.append(trade['price'])
                    elif trade['type'] == '卖出':
                        sell_dates.append(trade['date'])
                        sell_prices.append(trade['price'])
                
                # 绘制买卖点
                if buy_dates:
                    ax1.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, label='买入点', zorder=5)
                if sell_dates:
                    ax1.scatter(sell_dates, sell_prices, color='green', marker='v', s=100, label='卖出点', zorder=5)
            
            # 根据策略类型添加相应指标
            if 'RSI' in name:
                # 使用pandas计算RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                ax1_twin = ax1.twinx()
                ax1_twin.plot(data.index, rsi, label='RSI', color='red', alpha=0.7)
                ax1_twin.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                ax1_twin.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                ax1_twin.set_ylabel('RSI')
                ax1_twin.legend()
                
            elif 'MACD' in name:
                # 使用pandas计算MACD
                exp1 = data['close'].ewm(span=12).mean()
                exp2 = data['close'].ewm(span=26).mean()
                macd_line = exp1 - exp2
                signal_line = macd_line.ewm(span=9).mean()
                
                ax1_twin = ax1.twinx()
                ax1_twin.plot(data.index, macd_line, label='MACD', color='blue', alpha=0.7)
                ax1_twin.plot(data.index, signal_line, label='Signal', color='red', alpha=0.7)
                ax1_twin.set_ylabel('MACD')
                ax1_twin.legend()
                
            elif '布林带' in name:
                # 使用pandas计算布林带
                ma20 = data['close'].rolling(window=20).mean()
                std20 = data['close'].rolling(window=20).std()
                upper_band = ma20 + (std20 * 2)
                lower_band = ma20 - (std20 * 2)
                
                ax1.plot(data.index, upper_band, label='上轨', color='red', alpha=0.7)
                ax1.plot(data.index, ma20, label='中轨', color='blue', alpha=0.7)
                ax1.plot(data.index, lower_band, label='下轨', color='green', alpha=0.7)
                
            elif '双均线' in name:
                ma5 = data['close'].rolling(window=5).mean()
                ma20 = data['close'].rolling(window=20).mean()
                ax1.plot(data.index, ma5, label='MA5', color='blue', alpha=0.7)
                ax1.plot(data.index, ma20, label='MA20', color='red', alpha=0.7)
            
            ax1.set_title(f'{name} - 价格与指标')
            ax1.set_ylabel('价格 (元)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 第二个子图：资金曲线
            ax2.plot(data.index, [result['final_value']] * len(data), 
                    label=f'最终资金: {result["final_value"]:,.0f}', 
                    color='green', linewidth=2)
            ax2.set_title(f'{name} - 资金曲线')
            ax2.set_ylabel('资金 (元)')
            ax2.set_xlabel('日期')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"绘制单个策略图表时出现错误: {e}")
        print("错误详情：")
        import traceback
        traceback.print_exc()

def print_trade_details(strategy_results):
    """
    输出每个策略的交易详情
    """
    print("\n" + "=" * 80)
    print("各策略交易详情")
    print("=" * 80)
    
    for result in strategy_results:
        name = result['name']
        trades = result.get('trades', [])
        
        print(f"\n{name}:")
        print("-" * 50)
        
        if trades:
            for i, trade in enumerate(trades, 1):
                print(f"交易 {i}: {trade['date']} - {trade['type']} - 价格: {trade['price']:.2f}")
                
                # 输出策略特定的指标信息
                if 'rsi' in trade:
                    print(f"  RSI: {trade['rsi']:.2f}")
                elif 'macd' in trade:
                    print(f"  MACD: {trade['macd']:.4f}, Signal: {trade['signal']:.4f}")
                elif 'bb_bot' in trade:
                    print(f"  布林带下轨: {trade['bb_bot']:.2f}")
                elif 'bb_top' in trade:
                    print(f"  布林带上轨: {trade['bb_top']:.2f}")
                elif 'ma_fast' in trade:
                    print(f"  快线: {trade['ma_fast']:.2f}, 慢线: {trade['ma_slow']:.2f}")
        else:
            print("无交易记录")

def main():
    """
    主函数
    """
    print("万丰奥威多策略量化回测系统")
    print("=" * 50)
    
    # 1. 获取数据
    data = get_stock_data('002085', '20230101', '20241231')
    if data is None:
        print("数据获取失败，程序退出")
        return
    
    print(f"数据获取成功，时间范围: {data.index[0].date()} 到 {data.index[-1].date()}")
    print(f"数据点数量: {len(data)}")
    
    # 2. 运行多策略回测
    strategy_results = run_multi_strategy_backtest(data, initial_cash=1000)
    
    # 3. 输出交易详情
    print_trade_details(strategy_results)
    
    # 4. 可视化结果
    if strategy_results:
        try:
            results_df = visualize_results(data, strategy_results)
            
            # 5. 绘制每个策略的详细图表
            plot_individual_strategy(data, strategy_results)
            
            # 6. 找出最佳策略
            best_strategy = max(strategy_results, key=lambda x: x['total_return'])
            print(f"\n最佳策略: {best_strategy['name']}")
            print(f"总收益率: {best_strategy['total_return']:.2f}%")
            print(f"夏普比率: {best_strategy['sharpe_ratio']:.4f}")
            print(f"最大回撤: {best_strategy['max_drawdown']:.2f}%")
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
    
    print("\n回测完成！")

if __name__ == "__main__":
    main()
