"""
===========================================
A股马丁格尔量化交易策略
===========================================

策略原理：
马丁格尔策略源于赌场，核心思想是"每次亏损后加倍下注"，以期望通过后续盈利弥补前期亏损。

交易逻辑：
1. 初始买入：当RSI指标显示超卖（<30）时，买入基础仓位（如100股）
2. 马丁加仓：如果买入后价格继续下跌5%，则加倍买入（200股）
3. 持续加仓：每下跌5%，仓位翻倍（400股、800股...），摊低平均成本
4. 止盈退出：当整体盈利达到8%时，全部卖出平仓
5. 风险控制：限制最大加仓次数（如4次），避免无限亏损

风险提示：
⚠️ 马丁格尔策略在持续下跌中风险极高，可能导致巨额亏损
⚠️ 仅适合震荡市场或有明确支撑位的情况
⚠️ 务必设置严格的止损和最大加仓次数限制
"""

import akshare as ak          # A股数据获取库
import backtrader as bt       # 量化回测框架
import pandas as pd           # 数据处理
import matplotlib.pyplot as plt  # 绘图
import numpy as np            # 数值计算
from datetime import datetime # 日期处理
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置中文字体，避免图表中文显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== A股马丁格尔量化交易策略 ===")
print("策略说明：")
print("1. 基础信号：RSI超卖时买入")
print("2. 马丁加仓：价格下跌时加倍仓位摊低成本")
print("3. 风险控制：限制最大加仓次数和单次仓位")
print("4. 止盈退出：整体盈利达到目标后全部卖出")

# ============================================================
# 1. 数据获取模块
# ============================================================

def get_stock_data(symbol, period="daily", start_date="20220101", end_date="20241231"):
    """
    从AKShare获取A股历史数据

    参数说明:
        symbol: 股票代码，如 '000001'（平安银行）
        period: 数据周期，默认'daily'（日线）
        start_date: 开始日期，格式'YYYYMMDD'
        end_date: 结束日期，格式'YYYYMMDD'

    返回值:
        DataFrame: 包含开高低收量的股票数据，索引为日期
        None: 获取失败时返回None
    """
    try:
        # 使用akshare获取前复权后的股票数据（adjust="qfq"表示前复权）
        # 前复权：保持最新价格不变，向前调整历史价格，适合做技术分析
        df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq")

        # 重命名列名为英文，符合backtrader要求
        df.rename(columns={
            '日期': 'datetime',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)

        # 将日期列转换为datetime类型并设置为索引
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # 确保所有价格和成交量数据都是数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None

# 2. 马丁格尔策略
class MartingaleStrategy(bt.Strategy):
    """
    马丁格尔策略（改进版，带风险控制）：
    - 初始买入：RSI < 30时买入基础仓位
    - 马丁加仓：价格下跌达到阈值时，加倍仓位摊低成本
    - 止盈退出：整体盈利达到目标后全部卖出
    - 止损保护：连续加仓次数达到上限后停止，避免无限亏损
    """

    params = (
        ('rsi_period', 14),             # RSI周期
        ('rsi_threshold', 30),          # RSI买入阈值
        ('base_stake', 100),            # 基础买入股数
        ('drop_threshold', 0.05),       # 下跌加仓阈值（5%）
        ('profit_target', 0.1),        # 整体止盈目标（8%）
        ('max_pyramiding', 5),          # 最大加仓次数
        ('martingale_multiplier', 2),   # 马丁倍数（每次加仓倍数）
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # 记录交易信息
        self.buy_signals = []
        self.sell_signals = []

        # 马丁格尔状态跟踪
        self.entry_prices = []          # 所有买入价格
        self.entry_sizes = []           # 所有买入数量
        self.pyramiding_count = 0       # 当前加仓次数
        self.last_entry_price = 0       # 上次买入价格
        self.total_cost = 0             # 总成本
        self.total_shares = 0           # 总持仓

    def notify_order(self, order):
        """订单通知"""
        if order.status in [order.Completed]:
            if order.isbuy():
                # 记录买入信息
                self.entry_prices.append(order.executed.price)
                self.entry_sizes.append(order.executed.size)
                self.total_cost += order.executed.price * order.executed.size
                self.total_shares += order.executed.size
                self.last_entry_price = order.executed.price
                self.pyramiding_count += 1

    def calculate_avg_price(self):
        """
        计算平均持仓成本

        公式: 平均成本 = 总成本 / 总持仓股数

        举例:
        买入100股@10元 + 200股@9.5元 + 400股@9元
        总成本 = 100×10 + 200×9.5 + 400×9 = 6500元
        总持仓 = 700股
        平均成本 = 6500 / 700 = 9.29元/股

        返回值:
            float: 平均持仓成本价格
        """
        if self.total_shares > 0:
            return self.total_cost / self.total_shares
        return 0

    def calculate_profit_pct(self):
        """
        计算当前整体盈亏百分比

        公式: 盈亏比例 = (当前价 - 平均成本) / 平均成本

        举例:
        平均成本9.29元，当前价10元
        盈亏比例 = (10 - 9.29) / 9.29 = 7.6%

        返回值:
            float: 盈亏百分比（小数形式，如0.08表示8%）
        """
        if self.total_shares == 0:
            return 0
        avg_price = self.calculate_avg_price()
        current_price = self.data.close[0]
        return (current_price - avg_price) / avg_price

    def reset_position_tracking(self):
        """
        重置所有仓位跟踪变量

        在以下情况调用此函数：
        1. 止盈卖出全部持仓后
        2. 止损清仓后

        重置后可以重新开始新一轮的马丁格尔交易
        """
        self.entry_prices = []
        self.entry_sizes = []
        self.pyramiding_count = 0
        self.last_entry_price = 0
        self.total_cost = 0
        self.total_shares = 0

    def next(self):
        """
        策略核心逻辑：每个交易日都会调用此函数

        执行流程：
        1. 如果有持仓：
           a. 检查是否达到止盈目标 → 全部卖出
           b. 检查是否需要加仓 → 价格下跌且未达加仓上限时加仓
        2. 如果无持仓：
           a. 等待RSI超卖信号 → 买入基础仓位
        """
        current_price = self.data.close[0]  # 获取当前收盘价

        # ========== 情况1：已有持仓，管理仓位 ==========
        if self.position:
            profit_pct = self.calculate_profit_pct()  # 计算当前盈亏比例
            avg_price = self.calculate_avg_price()    # 计算平均成本

            # 止盈逻辑：整体盈利达到目标，全部卖出
            if profit_pct >= self.params.profit_target:
                self.sell(size=self.position.size)  # 卖出全部持仓
                # 记录卖出信号，用于后续分析
                self.sell_signals.append({
                    'date': self.data.datetime.date(0),
                    'price': current_price,
                    'size': self.position.size,
                    'avg_cost': avg_price,
                    'profit_pct': profit_pct * 100,  # 转换为百分比显示
                    'reason': '止盈'
                })
                self.reset_position_tracking()  # 清空所有状态，准备下一轮交易
                return  # 本次交易结束，直接返回

            # 马丁加仓逻辑：价格下跌时加倍买入摊低成本
            if self.pyramiding_count < self.params.max_pyramiding:  # 未达到最大加仓次数
                # 计算相对于上次买入价格的跌幅
                drop_pct = (current_price - self.last_entry_price) / self.last_entry_price

                # 跌幅达到阈值（如-5%），触发加仓
                if drop_pct <= -self.params.drop_threshold:
                    # 计算下次买入数量 = 上次数量 × 马丁倍数
                    # 例如：上次200股，这次400股
                    next_stake = self.entry_sizes[-1] * self.params.martingale_multiplier
                    next_stake = int(next_stake / 100) * 100  # A股最小单位100股，向下取整

                    # 检查资金是否充足
                    required_cash = next_stake * current_price
                    if self.broker.get_cash() >= required_cash and next_stake >= 100:
                        self.buy(size=next_stake)  # 执行加仓买入
                        # 记录买入信号
                        self.buy_signals.append({
                            'date': self.data.datetime.date(0),
                            'price': current_price,
                            'size': next_stake,
                            'pyramiding_level': self.pyramiding_count + 1,  # 加仓级别
                            'rsi': self.rsi[0],
                            'type': '马丁加仓'
                        })

        # ========== 情况2：无持仓，等待买入信号 ==========
        else:
            # 当RSI指标显示超卖（<30）时，开始新一轮马丁格尔交易
            if self.rsi < self.params.rsi_threshold:
                # 买入基础仓位（第一次买入）
                stake = self.params.base_stake

                # 检查资金充足后执行买入
                if self.broker.get_cash() >= stake * current_price:
                    self.buy(size=stake)
                    # 记录初始买入信号
                    self.buy_signals.append({
                        'date': self.data.datetime.date(0),
                        'price': current_price,
                        'size': stake,
                        'pyramiding_level': 1,  # 第1层仓位
                        'rsi': self.rsi[0],
                        'type': '初始买入'
                    })

# ============================================================
# 3. 回测引擎
# ============================================================

def run_backtest(data, strategy_class, cash=100000, commission=0.001, name="策略"):
    """
    运行回测引擎

    参数:
        data: 股票历史数据DataFrame
        strategy_class: 策略类（如MartingaleStrategy）
        cash: 初始资金，默认10万元
        commission: 手续费率，默认0.1%
        name: 策略名称

    返回值:
        cerebro: 回测引擎对象
        results: 回测结果列表
        strat: 策略实例，包含所有交易信号
    """
    # 创建回测引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(strategy_class)

    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f'\n{name} - 初始资金: {cash:,.0f}')

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
    total_return = (final_value - cash) / cash * 100

    print(f'{name} - 最终资金: {final_value:,.0f}')
    print(f'{name} - 总收益率: {total_return:.2f}%')
    print(f'{name} - 夏普比率: {sharpe_ratio:.2f}' if sharpe_ratio else f'{name} - 夏普比率: N/A')
    print(f'{name} - 最大回撤: {max_drawdown:.2f}%')
    print("-" * 60)

    return cerebro, results, strat

# 4. 可视化函数
def plot_martingale_results(data, strategy, strategy_name, stock_code):
    """绘制马丁格尔策略回测结果"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

    # 主图：股价和买卖点
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

    # 标记买入点（区分初始买入和马丁加仓）
    if hasattr(strategy, 'buy_signals') and strategy.buy_signals:
        for signal in strategy.buy_signals:
            if isinstance(signal, dict):
                date = signal['date']
                price = signal['price']
                size = signal.get('size', 0)
                buy_type = signal.get('type', '买入')
                level = signal.get('pyramiding_level', 1)

                # 不同颜色区分初始买入和加仓
                if buy_type == '初始买入':
                    color = 'red'
                    marker = '^'
                    label = '初始买入' if signal == strategy.buy_signals[0] else ""
                else:
                    color = 'orange'
                    marker = '^'
                    label = '马丁加仓' if buy_type == '马丁加仓' and len([s for s in strategy.buy_signals[:strategy.buy_signals.index(signal)] if s.get('type') == '马丁加仓']) == 0 else ""

                ax1.scatter(date, price, color=color, marker=marker, s=150,
                           label=label, zorder=5, edgecolors='darkred', linewidth=1.5)

                # 添加标注
                ax1.annotate(f'{buy_type}\nLv.{level}\n{size}股',
                           xy=(date, price), xytext=(5, 15),
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           color='white', fontweight='bold')

    # 标记卖出点
    if hasattr(strategy, 'sell_signals') and strategy.sell_signals:
        for signal in strategy.sell_signals:
            if isinstance(signal, dict):
                date = signal['date']
                price = signal['price']
                size = signal.get('size', 0)
                profit_pct = signal.get('profit_pct', 0)

                ax1.scatter(date, price, color='green', marker='v', s=150,
                           label='止盈卖出' if signal == strategy.sell_signals[0] else "",
                           zorder=5, edgecolors='darkgreen', linewidth=1.5)

                ax1.annotate(f'止盈\n{size}股\n+{profit_pct:.1f}%',
                           xy=(date, price), xytext=(5, -30),
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           color='white', fontweight='bold')

    ax1.set_ylabel('价格', fontsize=12)
    ax1.set_title(f'{stock_code} - {strategy_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 副图：RSI指标
    ax2.plot(data.index, rsi_values, label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=strategy.params.rsi_threshold, color='g', linestyle='--',
                alpha=0.8, linewidth=2, label=f'买入线({strategy.params.rsi_threshold})')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.8, linewidth=2, label='超买线(70)')
    ax2.axhline(y=50, color='black', linestyle=':', alpha=0.5, label='中轴线')

    ax2.fill_between(data.index, 0, strategy.params.rsi_threshold, alpha=0.2, color='green', label='超卖区域')

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI值', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 5. 交易分析函数
def analyze_martingale_trades(strategy, strategy_name):
    """分析马丁格尔交易详情"""
    print(f"\n{'='*60}")
    print(f"{strategy_name} - 交易详情分析")
    print(f"{'='*60}")

    if hasattr(strategy, 'buy_signals') and strategy.buy_signals:
        total_buys = len(strategy.buy_signals)
        initial_buys = len([s for s in strategy.buy_signals if s.get('type') == '初始买入'])
        martingale_buys = len([s for s in strategy.buy_signals if s.get('type') == '马丁加仓'])

        print(f"\n买入统计:")
        print(f"  - 总买入次数: {total_buys}")
        print(f"  - 初始买入次数: {initial_buys}")
        print(f"  - 马丁加仓次数: {martingale_buys}")

        total_shares = sum([s.get('size', 0) for s in strategy.buy_signals])
        print(f"  - 累计买入股数: {total_shares:,}")

        print(f"\n买入详情(前5次):")
        for i, signal in enumerate(strategy.buy_signals[:5]):
            if isinstance(signal, dict):
                print(f"  {i+1}. {signal.get('type', '买入')} - "
                      f"日期:{signal['date']}, 价格:{signal['price']:.2f}, "
                      f"股数:{signal.get('size', 0)}, 级别:Lv.{signal.get('pyramiding_level', 1)}, "
                      f"RSI:{signal.get('rsi', 0):.2f}")
    else:
        print("无买入信号")

    if hasattr(strategy, 'sell_signals') and strategy.sell_signals:
        total_sells = len(strategy.sell_signals)
        print(f"\n卖出统计:")
        print(f"  - 总卖出次数: {total_sells}")

        total_profit = sum([s.get('profit_pct', 0) for s in strategy.sell_signals])
        avg_profit = total_profit / total_sells if total_sells > 0 else 0
        print(f"  - 平均单次盈利: {avg_profit:.2f}%")

        print(f"\n卖出详情:")
        for i, signal in enumerate(strategy.sell_signals):
            if isinstance(signal, dict):
                print(f"  {i+1}. 日期:{signal['date']}, 价格:{signal['price']:.2f}, "
                      f"股数:{signal.get('size', 0)}, 盈利:{signal.get('profit_pct', 0):.2f}%, "
                      f"平均成本:{signal.get('avg_cost', 0):.2f}")
    else:
        print("无卖出信号")

    # 策略参数
    print(f"\n策略参数:")
    print(f"  - 基础仓位: {strategy.params.base_stake}股")
    print(f"  - 马丁倍数: {strategy.params.martingale_multiplier}x")
    print(f"  - 最大加仓次数: {strategy.params.max_pyramiding}次")
    print(f"  - 下跌加仓阈值: {strategy.params.drop_threshold*100}%")
    print(f"  - 止盈目标: {strategy.params.profit_target*100}%")

    print("-" * 60)

# 6. 主程序
def main():
    # 测试股票
    stocks = {
        '000001': '平安银行',
        '600036': '招商银行',
        '000002': '万科A'
    }

    start_date = "20230101"
    end_date = datetime.now().strftime("%Y%m%d")

    print("\n正在获取股票数据...")

    stock_data = {}
    for code, name in stocks.items():
        print(f"\n获取 {name}({code}) 数据...")
        data = get_stock_data(code, start_date=start_date, end_date=end_date)
        if data is not None and len(data) > 50:
            stock_data[code] = {'data': data, 'name': name}
        else:
            print(f"警告: {name}({code}) 数据不足")

    if not stock_data:
        print("错误: 无法获取任何股票数据")
        return

    print(f"\n成功获取 {len(stock_data)} 只股票数据")

    # 回测参数
    initial_cash = 100000
    commission = 0.001

    results_summary = []

    # 对每只股票进行回测
    for code, stock_info in stock_data.items():
        data = stock_info['data']
        name = stock_info['name']

        print(f"\n{'='*60}")
        print(f"开始回测 {name}({code})")
        print(f"数据时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        try:
            cerebro, _, strat = run_backtest(
                data, MartingaleStrategy,
                cash=initial_cash,
                commission=commission,
                name="马丁格尔策略"
            )

            # 可视化结果
            plot_martingale_results(data, strat, "马丁格尔策略", f"{name}({code})")

            # 分析交易
            analyze_martingale_trades(strat, "马丁格尔策略")

            # 记录结果
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - initial_cash) / initial_cash * 100
            results_summary.append({
                'stock': f"{name}({code})",
                'return': total_return,
                'final_value': final_value
            })

        except Exception as e:
            print(f"回测失败: {e}")
            import traceback
            traceback.print_exc()

    # 结果汇总
    if results_summary:
        print(f"\n{'='*60}")
        print("回测结果汇总")
        print(f"{'='*60}")

        results_df = pd.DataFrame(results_summary)
        print("\n详细结果:")
        print(results_df.to_string(index=False, formatters={
            'return': '{:.2f}%'.format,
            'final_value': '{:,.0f}'.format
        }))

        # 绘制对比图
        plt.figure(figsize=(12, 8))
        stocks_list = [r['stock'] for r in results_summary]
        returns = [r['return'] for r in results_summary]

        bars = plt.bar(range(len(stocks_list)), returns,
                       alpha=0.8, color='steelblue', edgecolor='navy', linewidth=1.5)

        # 标注数值
        for i, ret in enumerate(returns):
            plt.text(i, ret + (max(returns) * 0.01 if max(returns) > 0 else -abs(min(returns)) * 0.01),
                    f'{ret:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.ylabel('收益率 (%)', fontsize=12)
        plt.title('马丁格尔策略 - 不同股票收益率对比', fontsize=16, fontweight='bold')
        plt.xticks(range(len(stocks_list)), [s.split('(')[0] for s in stocks_list], fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        plt.tight_layout()
        plt.show()

        # 统计分析
        avg_return = np.mean(returns)
        print(f"\n策略表现:")
        print(f"  - 平均收益率: {avg_return:.2f}%")
        print(f"  - 最高收益率: {max(returns):.2f}%")
        print(f"  - 最低收益率: {min(returns):.2f}%")
        print(f"  - 收益波动性: {np.std(returns):.2f}%")

        print(f"\n⚠️  风险提示:")
        print(f"  - 马丁格尔策略在下跌趋势中风险极高")
        print(f"  - 需要严格控制最大加仓次数")
        print(f"  - 建议仅在震荡市或上升趋势中使用")
        print(f"  - 务必设置止损，避免巨额亏损")
    else:
        print("没有成功的回测结果")

# 运行主程序
if __name__ == "__main__":
    main()
