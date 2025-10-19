# 基于Backtrader的多股票综合评分回测系统
# 数据源：本地CSV文件 (从 stock_data 目录读取)
# 策略：综合评分>=阈值时建仓，达到卖出条件时全部清仓
# 特点：只要有资金就持续寻找买入机会，支持多只股票同时持仓
# 使用前：请先运行 download_stock_data_baostock.py 下载数据

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import glob

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiStockStrategy(bt.Strategy):
    """
    多股票综合评分轮动策略

    核心逻辑：
    1. 每日扫描所有股票，计算买入和卖出综合评分
    2. 持仓股票评分>=卖出阈值时全部清仓
    3. 非持仓股票评分>=买入阈值且有资金时建仓
    4. 按评分从高到低买入，充分利用资金
    """

    params = (
        # 评分阈值
        ('buy_threshold', 50),        # 买入评分阈值
        ('sell_threshold', 50),       # 卖出评分阈值

        # 仓位管理
        ('max_positions', 3),         # 最大持仓股票数
        ('position_ratio', 0.35),     # 单只股票最大仓位比例

        # 技术指标参数
        ('rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
        ('ma_short', 5),
        ('ma_medium', 20),
        ('ma_long', 60),
        ('volume_ma_period', 20),
        ('volume_ratio_threshold', 1.2),
    )

    def __init__(self):
        """初始化策略"""
        # 为每个数据源创建指标
        self.indicators = {}

        for i, d in enumerate(self.datas):
            # RSI
            rsi = bt.indicators.RSI(d.close, period=self.params.rsi_period)

            # 多周期均线
            ma5 = bt.indicators.SMA(d.close, period=self.params.ma_short)
            ma20 = bt.indicators.SMA(d.close, period=self.params.ma_medium)
            ma60 = bt.indicators.SMA(d.close, period=self.params.ma_long)

            # 成交量均线
            volume_ma = bt.indicators.SMA(d.volume, period=self.params.volume_ma_period)

            self.indicators[d._name] = {
                'rsi': rsi,
                'ma5': ma5,
                'ma20': ma20,
                'ma60': ma60,
                'volume_ma': volume_ma,
                'data': d
            }

        # 记录交易信号
        self.buy_signals = []
        self.sell_signals = []

        # 记录每日账户价值
        self.portfolio_values = []
        self.dates = []

        # 订单管理
        self.orders = {}

    def calculate_buy_score(self, data_name):
        """
        计算买入信号评分（满分100分）
        """
        indicators = self.indicators[data_name]
        data = indicators['data']

        score = 0
        details = {}

        # === 过滤器1: RSI超卖 ===
        rsi = indicators['rsi'][0]
        if rsi < 50:
            rsi_score = (50 - rsi) * 1.2  # 连续评分
        else:
            rsi_score = 0
        details['RSI'] = rsi
        details['RSI得分'] = rsi_score
        score+=rsi_score

        # === 过滤器2: 多周期均线趋势 ===
        ma_score = 0
        ma5, ma20, ma60 = indicators['ma5'][0], indicators['ma20'][0], indicators['ma60'][0]
        slope20 = indicators['ma20'][0] - indicators['ma20'][-5]
        if ma5 > ma20:
            ma_score += 10
        if slope20 > 0:
            ma_score += min(slope20 / ma20 * 1000, 10)
        score += ma_score
        details['均线'] = f"+{ma_score:.1f}"

        # === 过滤器3: 成交量放量 ===
        volume_ratio = data.volume[0] / indicators['volume_ma'][0]
        volume_score = max(0, min((volume_ratio - 1) * 20, 25))
        score += volume_score
        details['量比'] = volume_ratio
        details['成交量得分'] = volume_score

        return score, details

    def calculate_sell_score(self, data_name):
        """
        卖出信号评分（0-100分）
        优化目标：
        - 平滑化评分函数
        - 引入趋势与动量斜率
        - 加强缩量出货识别
        """
        indicators = self.indicators[data_name]
        data = indicators['data']

        score = 0
        details = {}

        # === 过滤器1: RSI止盈信号（权重60）===
        rsi = indicators['rsi'][0]
        # 在 RSI=60-100 区间内线性映射
        if rsi > 60:
            rsi_score = min((rsi - 60)*1.8, 60)
        else:
            rsi_score = 0
        score += rsi_score
        details['RSI'] = round(rsi, 2)
        details['RSI得分'] = round(rsi_score, 2)

        # === 过滤器2: 均线趋势转弱（权重30）===
        ma5 = indicators['ma5'][0]
        ma5_prev = indicators['ma5'][-1]
        ma20 = indicators['ma20'][0]
        close_price = data.close[0]

        # 趋势跟踪-ma
        # 趋势反转-rsi
        # (1) MA5斜率得分：越下行得分越高
        ma5_slope = (ma5 - ma5_prev) / ma5_prev
        slope_score = min(max(-ma5_slope * 4000, 0), 15)  # 下行2%以上给满分20

        # (2) 价格偏离MA20得分
        deviation = (ma20 - close_price) / ma20
        deviation_score = min(max(deviation * 100, 0), 10)  # 跌破2%给15分

        # (3) MA5下穿MA20加分
        cross_score = 5 if ma5 < ma20 else 0

        ma_score = slope_score + deviation_score + cross_score
        score += ma_score
        details['均线斜率'] = round(ma5_slope * 100, 2)
        details['偏离度'] = round(deviation * 100, 2)
        details['均线得分'] = round(ma_score, 2)

        # === 过滤器3: 成交量信号（权重10）===
        volume_score = 0
        volume_ratio = data.volume[0] / indicators['volume_ma'][0] if indicators['volume_ma'][0] > 0 else 1

        # 放量出货：高位成交放大
        if volume_ratio >= 1.5:
            volume_score = 10
        # 缩量滞涨：价格弱势但缩量
        elif 0.5 < volume_ratio < 0.9:
            volume_score = 8
        # 持平则中性
        elif 0.9 <= volume_ratio <= 1.2:
            volume_score = 4

        score += volume_score
        details['成交量得分'] = round(volume_score, 2)
        details['量比'] = round(volume_ratio, 2)

        # === 总分修正（防止超出100）===
        score = min(score, 100)
        details['总分'] = round(score, 2)

        return score, details


    def calculate_position_size(self, data, score):
        """
        根据综合评分计算买入数量
        单只股票不超过总资金的position_ratio
        """
        current_price = data.close[0]
        total_value = self.broker.getvalue()

        # 单只股票最大投资金额
        max_investment = total_value * self.params.position_ratio

        # 可用资金
        available_cash = self.broker.getcash()

        # 实际投资金额
        investment = min(available_cash, max_investment)

        # 计算股数（向下取整到100股）
        shares = int(investment / current_price / 100) * 100

        return max(shares, 100) if shares >= 100 else 0

    def next(self):
        """策略主逻辑 - 每个交易日执行"""
        # 记录每日账户价值
        self.dates.append(self.datas[0].datetime.date(0))
        self.portfolio_values.append(self.broker.getvalue())

        current_date = self.datas[0].datetime.date(0)

        # === 步骤1: 检查卖出信号（先卖出，释放资金）===
        for data_name, indicators in self.indicators.items():
            data = indicators['data']
            position = self.getposition(data)

            # 如果持有该股票
            if position.size > 0:
                sell_score, sell_details = self.calculate_sell_score(data_name)

                # 达到卖出阈值，全部清仓
                if sell_score >= self.params.sell_threshold:
                    self.close(data=data)

                    self.sell_signals.append({
                        'date': current_date,
                        'stock_name': data_name,
                        'price': data.close[0],
                        'size': position.size,
                        'score': sell_score,
                        'details': sell_details
                    })

        # === 步骤2: 扫描买入机会并按评分排序 ===
        buy_candidates = []

        for data_name, indicators in self.indicators.items():
            data = indicators['data']
            position = self.getposition(data)

            # 跳过已持仓的股票
            if position.size > 0:
                continue

            # 计算买入评分
            buy_score, buy_details = self.calculate_buy_score(data_name)

            # 评分达标
            if buy_score >= self.params.buy_threshold:
                buy_candidates.append({
                    'data_name': data_name,
                    'data': data,
                    'score': buy_score,
                    'details': buy_details,
                    'price': data.close[0]
                })

        # === 步骤3: 按评分排序，执行买入 ===
        if buy_candidates:
            # 按评分降序排序
            buy_candidates.sort(key=lambda x: x['score'], reverse=True)

            # 统计当前持仓数量
            current_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)

            for candidate in buy_candidates:
                # 检查持仓数量限制
                if current_positions >= self.params.max_positions:
                    break

                # 检查是否有足够资金
                size = self.calculate_position_size(candidate['data'], candidate['score'])

                if size >= 100:
                    # 买入
                    self.buy(data=candidate['data'], size=size)

                    self.buy_signals.append({
                        'date': current_date,
                        'stock_name': candidate['data_name'],
                        'price': candidate['price'],
                        'size': size,
                        'score': candidate['score'],
                        'details': candidate['details']
                    })

                    current_positions += 1

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                pass  # 买入完成
            elif order.issell():
                pass  # 卖出完成

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass  # 订单被取消/拒绝


def get_stock_data(code, name, csv_dir='stock_data'):
    """
    从本地CSV文件读取股票数据

    参数:
        code: 股票代码
        name: 股票名称
        csv_dir: CSV文件目录

    返回:
        DataFrame or None
    """
    try:
        # CSV文件名格式: 代码_名称.csv
        csv_file = os.path.join(csv_dir, f"{code}_{name}.csv")

        if not os.path.exists(csv_file):
            print(f"  ✗ {name}({code}): 文件不存在 {csv_file}")
            return None

        # 读取CSV
        df = pd.read_csv(csv_file)

        # 标准化列名
        df.columns = [col.lower() for col in df.columns]

        # 确保有date列
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            print(f"  ✗ {name}({code}): CSV缺少日期列")
            return None

        df.set_index('datetime', inplace=True)

        # 检查必需列
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            print(f"  ✗ {name}({code}): CSV缺少必需列")
            return None

        # 确保数据类型正确
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除空值
        df.dropna(subset=required, inplace=True)

        return df

    except Exception as e:
        print(f"  ✗ {name}({code}): 读取失败 - {str(e)[:60]}")
        return None


def run_backtest(stock_pool, start_date, end_date,
                 initial_cash=100000,
                 buy_threshold=60,
                 sell_threshold=60,
                 max_positions=3,
                 position_ratio=0.35,
                 commission=0.001):
    """
    运行多股票回测

    Parameters:
    -----------
    stock_pool : dict
        股票池 {code: name}
    start_date : str
        开始日期 'YYYYMMDD'
    end_date : str
        结束日期 'YYYYMMDD'
    initial_cash : float
        初始资金
    buy_threshold : float
        买入评分阈值
    sell_threshold : float
        卖出评分阈值
    max_positions : int
        最大持仓数量
    position_ratio : float
        单只股票最大仓位比例
    commission : float
        手续费率
    """

    print(f"\n{'='*60}")
    print(f"开始回测...")
    print(f"股票池: {len(stock_pool)} 只股票")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"买入阈值: {buy_threshold}分")
    print(f"卖出阈值: {sell_threshold}分")
    print(f"最大持仓: {max_positions}只")
    print(f"单只仓位: {position_ratio*100:.0f}%")
    print(f"{'='*60}\n")

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(
        MultiStockStrategy,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        position_ratio=position_ratio
    )

    # 从本地CSV加载股票数据
    print("正在从CSV文件加载股票数据...")
    loaded_count = 0

    for code, name in stock_pool.items():
        df = get_stock_data(code, name, csv_dir='stock_data')

        if df is not None and len(df) >= 70:  # 至少需要60天数据（MA60）
            data_feed = bt.feeds.PandasData(dataname=df, name=name)
            cerebro.adddata(data_feed)
            loaded_count += 1
            print(f"  ✓ {name}({code}): {len(df)}条数据")
        else:
            if df is None:
                pass  # 错误信息已在get_stock_data中打印
            else:
                print(f"  ✗ {name}({code}): 数据不足({len(df)}条)，跳过")

    if loaded_count == 0:
        print("\n错误: 没有成功加载任何股票数据")
        print("提示: 请先运行 download_stock_data_baostock.py 下载数据到 stock_data 目录")
        return None, None

    print(f"\n成功加载 {loaded_count} 只股票\n")

    # 设置初始资金
    cerebro.broker.setcash(initial_cash)

    # 设置手续费
    cerebro.broker.setcommission(commission=commission)

    # 设置股票交易规则（每手100股）
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f"开始回测...\n")

    # 运行回测
    results = cerebro.run()
    strat = results[0]

    # 获取最终价值
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    print(f"\n{'='*60}")
    print(f"回测完成！")
    print(f"{'='*60}")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"最终资金: ¥{final_value:,.2f}")
    print(f"总收益: ¥{final_value - initial_cash:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"{'='*60}\n")

    return cerebro, strat


def print_trade_statistics(strat):
    """打印交易统计"""
    print(f"\n{'='*60}")
    print(f"交易统计")
    print(f"{'='*60}")
    print(f"买入次数: {len(strat.buy_signals)}")
    print(f"卖出次数: {len(strat.sell_signals)}")

    if strat.buy_signals:
        avg_buy_score = np.mean([s['score'] for s in strat.buy_signals])
        print(f"平均买入评分: {avg_buy_score:.2f}分")

    if strat.sell_signals:
        avg_sell_score = np.mean([s['score'] for s in strat.sell_signals])
        print(f"平均卖出评分: {avg_sell_score:.2f}分")

    print(f"{'='*60}\n")

    # 显示前10笔买入信号
    if strat.buy_signals:
        print(f"前10笔买入信号:")
        for i, signal in enumerate(strat.buy_signals[:10], 1):
            print(f"  {i}. {signal['date']} {signal['stock_name']}: "
                  f"¥{signal['price']:.2f}, {signal['size']}股, 评分{signal['score']:.1f}")

    print()

    # 显示前10笔卖出信号
    if strat.sell_signals:
        print(f"前10笔卖出信号:")
        for i, signal in enumerate(strat.sell_signals[:10], 1):
            print(f"  {i}. {signal['date']} {signal['stock_name']}: "
                  f"¥{signal['price']:.2f}, {signal['size']}股, 评分{signal['score']:.1f}")


def plot_portfolio_value(strat):
    """绘制账户价值曲线"""
    if not strat.portfolio_values:
        print("没有数据可绘制")
        return

    dates = strat.dates
    values = strat.portfolio_values

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, values, label='账户总值', linewidth=2, color='navy')
    ax.axhline(y=strat.broker.startingcash, color='red', linestyle='--',
               alpha=0.7, label=f'初始资金 ¥{strat.broker.startingcash:,.0f}')

    # 标注买入点
    for signal in strat.buy_signals:
        idx = dates.index(signal['date']) if signal['date'] in dates else None
        if idx is not None:
            ax.scatter(signal['date'], values[idx], color='red', marker='^',
                      s=80, alpha=0.6, zorder=5)

    # 标注卖出点
    for signal in strat.sell_signals:
        idx = dates.index(signal['date']) if signal['date'] in dates else None
        if idx is not None:
            ax.scatter(signal['date'], values[idx], color='green', marker='v',
                      s=80, alpha=0.6, zorder=5)

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('账户价值 (元)', fontsize=11)
    ax.set_title('多股票轮动策略 - 账户价值变化', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'portfolio_value_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {filename}")
    plt.show()


def plot_stock_trades(strat, stock_pool):
    """为每只有交易的股票绘制价格、成交量和买卖点"""
    if not strat.buy_signals and not strat.sell_signals:
        print("没有交易记录，无法绘制")
        return

    # 找出所有有交易的股票
    traded_stocks = set()
    for signal in strat.buy_signals:
        traded_stocks.add(signal['stock_name'])
    for signal in strat.sell_signals:
        traded_stocks.add(signal['stock_name'])

    if not traded_stocks:
        print("没有交易的股票")
        return

    print(f"\n为 {len(traded_stocks)} 只股票绘制交易图表...")

    # 反向映射：股票名称 -> 代码
    name_to_code = {v: k for k, v in stock_pool.items()}

    for stock_name in traded_stocks:
        code = name_to_code.get(stock_name)
        if not code:
            continue

        # 从CSV获取该股票数据
        df = get_stock_data(code, stock_name, csv_dir='stock_data')
        if df is None or df.empty:
            continue

        # 该股票的买卖信号
        buy_signals = [s for s in strat.buy_signals if s['stock_name'] == stock_name]
        sell_signals = [s for s in strat.sell_signals if s['stock_name'] == stock_name]

        # 创建图表：价格 + 成交量
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # === 子图1: 价格和买卖点 ===
        ax1.plot(df.index, df['close'], label='收盘价', linewidth=1.5, color='black', alpha=0.7)

        # 绘制均线
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        ma60 = df['close'].rolling(60).mean()
        ax1.plot(df.index, ma5, label='MA5', linewidth=1, alpha=0.6, color='orange')
        ax1.plot(df.index, ma20, label='MA20', linewidth=1, alpha=0.6, color='blue')
        ax1.plot(df.index, ma60, label='MA60', linewidth=1, alpha=0.6, color='purple')

        # 标注买入点
        for signal in buy_signals:
            ax1.scatter(signal['date'], signal['price'], color='red', marker='^',
                       s=200, alpha=0.8, zorder=5, edgecolors='darkred', linewidths=2)
            ax1.annotate(f"买\n¥{signal['price']:.2f}\n{signal['score']:.0f}分",
                        xy=(signal['date'], signal['price']),
                        xytext=(0, 20), textcoords='offset points',
                        fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))

        # 标注卖出点
        for signal in sell_signals:
            ax1.scatter(signal['date'], signal['price'], color='green', marker='v',
                       s=200, alpha=0.8, zorder=5, edgecolors='darkgreen', linewidths=2)
            ax1.annotate(f"卖\n¥{signal['price']:.2f}\n{signal['score']:.0f}分",
                        xy=(signal['date'], signal['price']),
                        xytext=(0, -30), textcoords='offset points',
                        fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1))

        ax1.set_ylabel('价格 (元)', fontsize=11)
        ax1.set_title(f'{stock_name}({code}) - 交易信号分析', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # === 子图2: 成交量 ===
        colors = ['red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green'
                  for i in range(len(df))]
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.5, width=0.8)

        # 成交量均线
        vol_ma20 = df['volume'].rolling(20).mean()
        ax2.plot(df.index, vol_ma20, label='成交量MA20', linewidth=1.5, color='blue', alpha=0.7)

        # 在成交量图上也标注买卖点
        for signal in buy_signals:
            idx = df.index.get_loc(signal['date']) if signal['date'] in df.index else None
            if idx is not None:
                ax2.scatter(signal['date'], df['volume'].iloc[idx],
                           color='red', marker='^', s=150, alpha=0.8, zorder=5)

        for signal in sell_signals:
            idx = df.index.get_loc(signal['date']) if signal['date'] in df.index else None
            if idx is not None:
                ax2.scatter(signal['date'], df['volume'].iloc[idx],
                           color='green', marker='v', s=150, alpha=0.8, zorder=5)

        ax2.set_xlabel('日期', fontsize=11)
        ax2.set_ylabel('成交量', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'trade_{code}_{stock_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ {stock_name}({code}): {filename}")
        plt.close()

    print(f"\n所有交易图表已保存！")


def save_trade_records(strat):
    """保存交易记录"""
    if not strat.buy_signals and not strat.sell_signals:
        print("没有交易记录")
        return

    # 保存买入记录
    if strat.buy_signals:
        df_buy = pd.DataFrame(strat.buy_signals)
        filename = f'buy_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_buy.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"买入记录已保存: {filename}")

    # 保存卖出记录
    if strat.sell_signals:
        df_sell = pd.DataFrame(strat.sell_signals)
        filename = f'sell_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_sell.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"卖出记录已保存: {filename}")


def main():
    """主函数"""
    print("=== 基于Backtrader的多股票综合评分回测系统 ===")
    print("数据源: stock_data 目录下的CSV文件")
    print("策略: 评分>=阈值时建仓，达到卖出条件全部清仓")
    print("特点: 只要有资金就持续寻找买入机会\n")

    # 定义股票池
    # 注意：需要先运行 download_stock_data_baostock.py 下载这些股票的数据
    stock_pool = {
        '000001': '平安银行',
        '000002': '万科A',
        '600036': '招商银行',
        '600519': '贵州茅台',
        '000858': '五粮液',
        '601318': '中国平安',
        '600276': '恒瑞医药',
        '000333': '美的集团',
        '600887': '伊利股份',
        '601166': '兴业银行',
        '000651': '格力电器',
        '601888': '中国中免',
        '600000': '浦发银行',
        '600030': '中信证券',
        '601398': '工商银行',
        '601988': '中国银行',
        '601328': '交通银行',
        '600016': '民生银行',
        '000725': '京东方A',
        '601012': '隆基绿能',
        '002475': '立讯精密',
        '300750': '宁德时代',
        '002594': '比亚迪',
        '000568': '泸州老窖',
        '000596': '古井贡酒',
        '603288': '海天味业',
        '600690': '海尔智家',
        '000338': '潍柴动力',
        '600031': '三一重工',
        '601688': '华泰证券',
        '601166': '兴业银行',
        '600585': '海螺水泥',
        '600009': '上海机场',
        '600048': '保利发展',
        '000876': '新希望',
        '002271': '东方雨虹',
        '002714': '牧原股份',
        '600809': '山西汾酒',
        '603259': '药明康德',
        '300059': '东方财富',
        '600104': '上汽集团',
        '601633': '长城汽车',
        '002142': '宁波银行',
        '601288': '农业银行',
        '600196': '复星医药',
        '600588': '用友网络',
        '002230': '科大讯飞',
        '000063': '中兴通讯',
        '600050': '中国联通',
        '600019': '宝钢股份',
        '601857': '中国石油',
        '601088': '中国神华',
        '600028': '中国石化',
        '000876': '新希望',
        '002508': '老板电器',
        '600183': '生益科技',
        '600118': '中国卫星',
        '002460': '赣锋锂业',
        '002049': '紫光国微',
        '688981': '中芯国际',
        '002415': '海康威视',
        '300124': '汇川技术',
        '300014': '亿纬锂能',
        '688599': '天合光能',
        '600801': '华新水泥',
        '601919': '中远海控',
        '600845': '宝信软件',
        '000069': '华侨城A',
        '000100': 'TCL科技',
        '600340': '华夏幸福',
        '000166': '申万宏源',
        '600999': '招商证券',
        '601901': '方正证券',
        '601211': '国泰君安',
        '002352': '顺丰控股',
        '002027': '分众传媒',
        '600760': '中航沈飞',
        '002410': '广联达',
        '300760': '迈瑞医疗',
        '688111': '金山办公',
        '300015': '爱尔眼科',
        '600406': '国电南瑞',
        '002129': 'TCL中环',
        '600893': '航发动力',
        '601899': '紫金矿业',
        '601336': '新华保险',
        '601601': '中国太保',
        '601628': '中国人寿',
        '002736': '国信证券',
        '600109': '国金证券',
        '002648': '卫星化学',
        '601601': '中国太保',
        '600346': '恒力石化',
        '000799': '酒鬼酒',
        '600132': '重庆啤酒',
        '600872': '中炬高新'
    }

    # 回测参数
    initial_cash = 100000      # 初始资金
    buy_threshold = 60         # 买入评分阈值
    sell_threshold = 60        # 卖出评分阈值
    max_positions = 3          # 最大持仓数量
    position_ratio = 0.35      # 单只股票最大仓位比例
    commission = 0.001         # 手续费

    # 回测时间
    start_date = "20230101"
    end_date = "20251018"

    # 运行回测
    cerebro, strat = run_backtest(
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        position_ratio=position_ratio,
        commission=commission
    )

    if cerebro is None or strat is None:
        print("回测失败")
        return

    # 打印统计信息
    print_trade_statistics(strat)

    # 绘制图表
    plot_portfolio_value(strat)

    # 绘制每只股票的交易图表
    plot_stock_trades(strat, stock_pool)

    # 保存交易记录
    save_trade_records(strat)

    print("\n回测完成！")


if __name__ == "__main__":
    main()
