# A股量化交易回测系统
# 使用RSI策略进行回测分析

import akshare as ak
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== A股量化交易回测系统 ===")
print("本系统将演示RSI策略在A股市场的应用")

# 1. 获取股票数据
def get_stock_data(symbol, period="daily", start_date="20220101", end_date="20241231"):
    """
    获取股票数据
    symbol: 股票代码 (如: '000001')
    period: 数据周期
    start_date: 开始日期
    end_date: 结束日期
    """
    try:
        # 获取股票历史数据
        df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
        
        # 重命名列以符合backtrader要求
        df.rename(columns={
            '日期': 'datetime',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)
        
        # 设置索引
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None

# 2. RSI策略
class RSIStrategy(bt.Strategy):
    """
    RSI策略：
    - 当RSI低于30时买入（超卖）
    - 当RSI高于70时卖出（超买）
    - 智能仓位管理：根据RSI偏离程度调整买入数量
    """

    params = (
        ('rsi_period', 14),     # RSI周期
        ('rsi_lower', 30),      # RSI下轨
        ('rsi_upper', 70),      # RSI上轨
        ('risk_percent', 1),  # 每次交易风险资金比例
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # 用于记录交易信号和技术指标
        self.buy_signals = []
        self.sell_signals = []


    def calculate_position_ratio(self):
        current_rsi = self.rsi[0]
        def expo_func_lower(x):
            return (-(x-self.params.rsi_lower)/self.params.rsi_lower)**2

        def expo_func_upper(x):
            return ((x-self.params.rsi_upper)/self.params.rsi_lower)**2
        if current_rsi < self.params.rsi_lower:
            # RSI越低，信号越强
            signal_strength = expo_func_lower(current_rsi)
        elif current_rsi > self.params.rsi_upper:
            # RSI越高，信号越强
            signal_strength = expo_func_upper(current_rsi)

        return signal_strength

    def calculate_position_size(self,buy_or_sell):
        # 如果要进行买入

        current_price = self.data.close[0]
        if buy_or_sell:
            base_amt = self.broker.get_cash()
            shares = int(base_amt * self.calculate_position_ratio() / current_price / 100) * 100
        else:
            base_amt = self.position.size
            shares = int(self.position.size * self.calculate_position_ratio() /100)

        return max(shares, 100)


    def next(self):
        if self.rsi < self.params.rsi_lower:  # RSI超卖买入
            size = self.calculate_position_size(True)
            if size >= 100:
                self.buy(size=size)
                self.buy_signals.append({
                    'date': self.data.datetime.date(0),
                    'price': self.data.close[0],
                    'rsi': self.rsi[0],
                    'size': size
                })

        elif self.rsi > self.params.rsi_upper and self.position:  # RSI超买卖出
            self.sell(size=self.calculate_position_size(False))  # 全部卖出
            self.sell_signals.append({
                'date': self.data.datetime.date(0),
                'price': self.data.close[0],
                'rsi': self.rsi[0],
                'size': self.position.size
            })

# 4. 回测函数
def run_backtest(data, strategy_class, cash=100000, commission=0.001, name="策略"):
    """
    运行回测
    """
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(strategy_class)
    
    # 添加数据
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    # 设置初始资金
    cerebro.broker.setcash(cash)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f'{name} - 初始资金: {cash:,.0f}')
    
    # 运行回测
    results = cerebro.run()
    
    # 获取最终资金
    final_value = cerebro.broker.getvalue()
    
    # 获取分析结果
    strat = results[0]
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
    total_return = (final_value - cash) / cash * 100
    
    print(f'{name} - 最终资金: {final_value:,.0f}')
    print(f'{name} - 总收益率: {total_return:.2f}%')
    print(f'{name} - 夏普比率: {sharpe_ratio:.2f}' if sharpe_ratio else f'{name} - 夏普比率: N/A')
    print(f'{name} - 最大回撤: {max_drawdown:.2f}%')
    print("-" * 50)
    
    return cerebro, results, strat

# 5. 增强的可视化函数
def plot_strategy_results(data, strategy, strategy_name, stock_code):
    """
    绘制策略回测结果
    - 添加买卖点垂直辅助线
    - 显示交易时的技术指标值
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    # 主图：股价
    ax1.plot(data.index, data['close'], label='收盘价', linewidth=2, alpha=0.8, color='navy')

    # 计算RSI
    rsi_values = []
    for i in range(len(data)):
        if i >= strategy.params.rsi_period:
            period_data = data['close'].iloc[i-strategy.params.rsi_period+1:i+1]
            delta = period_data.diff()
            gain = delta.where(delta > 0, 0).mean()
            loss = -delta.where(delta < 0, 0).mean()

            if loss != 0:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100

            rsi_values.append(rsi)
        else:
            rsi_values.append(np.nan)

    # 标记买卖点并添加垂直辅助线
    if hasattr(strategy, 'buy_signals') and strategy.buy_signals:
        for signal in strategy.buy_signals:
            if isinstance(signal, dict):
                date, price = signal['date'], signal['price']
                rsi_val = signal.get('rsi', 0)
                size = signal.get('size', 0)
            else:
                date, price = signal
                rsi_val, size = 0, 0

            # 买入点
            ax1.scatter(date, price, color='red', marker='^', s=120, label='买入点' if signal == strategy.buy_signals[0] else "", zorder=5, edgecolors='darkred', linewidth=1)

            # 垂直辅助线
            ax1.axvline(x=date, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax2.axvline(x=date, color='red', linestyle='--', alpha=0.6, linewidth=1)

            # 添加交易信息标注
            ax1.annotate(f'买入\n{size}股\nRSI:{rsi_val:.1f}' if size > 0 else f'买入\nRSI:{rsi_val:.1f}',
                       xy=(date, price), xytext=(5, 15),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                       color='white', fontweight='bold')

    if hasattr(strategy, 'sell_signals') and strategy.sell_signals:
        for signal in strategy.sell_signals:
            if isinstance(signal, dict):
                date, price = signal['date'], signal['price']
                rsi_val = signal.get('rsi', 0)
                size = signal.get('size', 0)
            else:
                date, price = signal
                rsi_val, size = 0, 0

            # 卖出点
            ax1.scatter(date, price, color='green', marker='v', s=120, label='卖出点' if signal == strategy.sell_signals[0] else "", zorder=5, edgecolors='darkgreen', linewidth=1)

            # 垂直辅助线
            ax1.axvline(x=date, color='green', linestyle='--', alpha=0.6, linewidth=1)
            ax2.axvline(x=date, color='green', linestyle='--', alpha=0.6, linewidth=1)

            # 添加交易信息标注
            ax1.annotate(f'卖出\n{size}股\nRSI:{rsi_val:.1f}' if size > 0 else f'卖出\nRSI:{rsi_val:.1f}',
                       xy=(date, price), xytext=(5, -25),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                       color='white', fontweight='bold')

    ax1.set_ylabel('价格')
    ax1.set_title(f'{stock_code} - {strategy_name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 副图显示RSI
    ax2.plot(data.index, rsi_values, label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=strategy.params.rsi_upper, color='r', linestyle='-', alpha=0.8, linewidth=2, label=f'超买线({strategy.params.rsi_upper})')
    ax2.axhline(y=strategy.params.rsi_lower, color='g', linestyle='-', alpha=0.8, linewidth=2, label=f'超卖线({strategy.params.rsi_lower})')
    ax2.axhline(y=50, color='black', linestyle=':', alpha=0.5, label='中轴线')

    # RSI区域填充
    ax2.fill_between(data.index, strategy.params.rsi_upper, 100, alpha=0.2, color='red', label='超买区域')
    ax2.fill_between(data.index, 0, strategy.params.rsi_lower, alpha=0.2, color='green', label='超卖区域')

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI值')
    ax2.set_xlabel('日期')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 6. 增强的交易分析函数
def analyze_trades(strategy, strategy_name):
    """
    分析交易详情
    """
    print(f"\n--- {strategy_name} 交易详情分析 ---")
    
    if hasattr(strategy, 'buy_signals') and strategy.buy_signals:
        print(f"总买入次数: {len(strategy.buy_signals)}")
        if strategy.buy_signals and isinstance(strategy.buy_signals[0], dict):
            total_buy_shares = sum([signal.get('size', 0) for signal in strategy.buy_signals])
            print(f"累计买入股数: {total_buy_shares:,}")
        
        # 显示前3次买入详情
        print("买入详情(前3次):")
        for i, signal in enumerate(strategy.buy_signals[:3]):
            if isinstance(signal, dict):
                print(f"  {i+1}. 日期:{signal['date']}, 价格:{signal['price']:.2f}, "
                      f"股数:{signal.get('size', 0)}, RSI:{signal.get('rsi', 0):.2f}")
            else:
                print(f"  {i+1}. 日期:{signal[0]}, 价格:{signal[1]:.2f}")
    else:
        print("无买入信号")
    
    if hasattr(strategy, 'sell_signals') and strategy.sell_signals:
        print(f"总卖出次数: {len(strategy.sell_signals)}")
        if strategy.sell_signals and isinstance(strategy.sell_signals[0], dict):
            total_sell_shares = sum([signal.get('size', 0) for signal in strategy.sell_signals])
            print(f"累计卖出股数: {total_sell_shares:,}")
    else:
        print("无卖出信号")
    
    print("-" * 40)

# 6. 主程序
def main():
    # 定义要分析的股票（选择3只具有代表性的A股）
    stocks = {
        '000001': '平安银行',     # 金融股
        '000002': '万科A',       # 地产股  
        '600036': '招商银行'     # 银行股
    }
    
    # 数据获取日期范围
    start_date = "20230101"
    end_date = datetime.now().strftime("%Y%m%d")  # 当前日期
    
    print("正在获取股票数据...")
    
    # 存储股票数据
    stock_data = {}
    
    # 获取每只股票的数据
    for code, name in stocks.items():
        print(f"\n获取 {name}({code}) 数据...")
        data = get_stock_data(code, start_date=start_date, end_date=end_date)
        if data is not None and len(data) > 50:  # 确保有足够的数据
            stock_data[code] = {
                'data': data,
                'name': name
            }
        else:
            print(f"警告: {name}({code}) 数据不足，跳过该股票")
    
    if not stock_data:
        print("错误: 无法获取任何股票数据，请检查网络连接或股票代码")
        return
    
    print(f"\n成功获取 {len(stock_data)} 只股票数据")
    
    # 回测参数
    initial_cash = 100000  # 初始资金10万
    commission = 0.001     # 手续费0.1%
    
    # 存储回测结果
    results_summary = []
    
    # 对每只股票进行RSI策略回测
    for code, stock_info in stock_data.items():
        data = stock_info['data']
        name = stock_info['name']

        print(f"\n{'='*60}")
        print(f"开始回测 {name}({code})")
        print(f"数据时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        # RSI策略
        print(f"\n--- {name} RSI策略回测 ---")
        try:
            cerebro, _, strat = run_backtest(
                data, RSIStrategy,
                cash=initial_cash,
                commission=commission,
                name="RSI策略"
            )

            # 可视化RSI策略结果
            plot_strategy_results(data, strat, "RSI策略", f"{name}({code})")

            # 分析交易详情
            analyze_trades(strat, "RSI策略")

            # 记录结果
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - initial_cash) / initial_cash * 100
            results_summary.append({
                'stock': f"{name}({code})",
                'strategy': 'RSI策略',
                'return': total_return,
                'final_value': final_value
            })

        except Exception as e:
            print(f"RSI策略回测失败: {e}")
    
    # 7. 结果汇总和对比分析
    print(f"\n{'='*80}")
    print("回测结果汇总")
    print(f"{'='*80}")
    
    if results_summary:
        # 创建结果DataFrame
        results_df = pd.DataFrame(results_summary)
        print("\n详细回测结果:")
        print(results_df.to_string(index=False, formatters={
            'return': '{:.2f}%'.format,
            'final_value': '{:,.0f}'.format
        }))

        # 绘制收益率对比图
        plt.figure(figsize=(12, 8))

        stocks_list = [r['stock'] for r in results_summary]
        returns = [r['return'] for r in results_summary]

        plt.bar(range(len(stocks_list)), returns,
                alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1.5)

        plt.ylabel('收益率 (%)', fontsize=12)
        plt.title('RSI策略不同股票收益率对比', fontsize=16, fontweight='bold')
        plt.xticks(range(len(stocks_list)), [s.split('(')[0] for s in stocks_list], fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, ret in enumerate(returns):
            plt.text(i, ret + (max(returns) * 0.01),
                    f'{ret:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.show()

        # 策略表现分析
        print(f"\n{'='*80}")
        print("策略表现分析")
        print(f"{'='*80}")

        avg_return = np.mean(returns)

        print(f"RSI策略:")
        print(f"  - 平均收益率: {avg_return:.2f}%")
        print(f"  - 最高收益率: {max(returns):.2f}%")
        print(f"  - 最低收益率: {min(returns):.2f}%")
        print(f"  - 收益波动性: {np.std(returns):.2f}%")
    
    else:
        print("没有成功的回测结果")


# 运行主程序
if __name__ == "__main__":    
    # 运行回测
    main()