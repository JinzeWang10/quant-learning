"""
基于Backtrader的多股票综合评分回测系统 (本地CSV数据版)
数据源: 读取 stock_data 目录下的所有CSV文件
特点: 离线回测，速度快，不依赖网络
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import os
import glob

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiStockStrategy(bt.Strategy):
    """多股票综合评分轮动策略"""

    params = (
        ('buy_threshold', 60),
        ('sell_threshold', 55),
        ('max_positions', 3),
        ('position_ratio', 0.35),
        ('rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
        ('ma_short', 5),
        ('ma_medium', 20),
        ('ma_long', 60),
        ('volume_ma_period', 20),
    )

    def __init__(self):
        """初始化策略"""
        self.indicators = {}

        for i, d in enumerate(self.datas):
            rsi = bt.indicators.RSI(d.close, period=self.params.rsi_period)
            ma5 = bt.indicators.SMA(d.close, period=self.params.ma_short)
            ma20 = bt.indicators.SMA(d.close, period=self.params.ma_medium)
            ma60 = bt.indicators.SMA(d.close, period=self.params.ma_long)
            volume_ma = bt.indicators.SMA(d.volume, period=self.params.volume_ma_period)

            self.indicators[d._name] = {
                'rsi': rsi,
                'ma5': ma5,
                'ma20': ma20,
                'ma60': ma60,
                'volume_ma': volume_ma,
                'data': d
            }

        self.buy_signals = []
        self.sell_signals = []
        self.portfolio_values = []
        self.dates = []
        self.orders = {}

    def calculate_buy_score(self, data_name):
        """计算买入评分 (满分100分)"""
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

        # === 过滤器1: RSI止盈信号（权重40）===
        rsi = indicators['rsi'][0]
        # 在 RSI=50-80 区间内线性映射
        if rsi > 50:
            rsi_score = min((rsi - 50) / 30 * 40, 40)
        else:
            rsi_score = 0
        score += rsi_score
        details['RSI'] = round(rsi, 2)
        details['RSI得分'] = round(rsi_score, 2)

        # === 过滤器2: 均线趋势转弱（权重40）===
        ma5 = indicators['ma5'][0]
        ma5_prev = indicators['ma5'][-1]
        ma20 = indicators['ma20'][0]
        close_price = data.close[0]

        # (1) MA5斜率得分：越下行得分越高
        ma5_slope = (ma5 - ma5_prev) / ma5_prev
        slope_score = min(max(-ma5_slope * 4000, 0), 20)  # 下行2%以上给满分20

        # (2) 价格偏离MA20得分
        deviation = (ma20 - close_price) / ma20
        deviation_score = min(max(deviation * 100, 0), 15)  # 跌破2%给15分

        # (3) MA5下穿MA20加分
        cross_score = 5 if ma5 < ma20 else 0

        ma_score = slope_score + deviation_score + cross_score
        score += ma_score
        details['均线斜率'] = round(ma5_slope * 100, 2)
        details['偏离度'] = round(deviation * 100, 2)
        details['均线得分'] = round(ma_score, 2)

        # === 过滤器3: 成交量信号（权重20）===
        volume_score = 0
        volume_ratio = data.volume[0] / indicators['volume_ma'][0] if indicators['volume_ma'][0] > 0 else 1

        # 放量出货：高位成交放大
        if volume_ratio >= 1.5:
            volume_score = 20
        # 缩量滞涨：价格弱势但缩量
        elif 0.5 < volume_ratio < 0.9:
            volume_score = 12
        # 持平则中性
        elif 0.9 <= volume_ratio <= 1.2:
            volume_score = 8

        score += volume_score
        details['成交量得分'] = round(volume_score, 2)
        details['量比'] = round(volume_ratio, 2)

        # === 总分修正（防止超出100）===
        score = min(score, 100)
        details['总分'] = round(score, 2)

        return score, details

    def calculate_position_size(self, data, score):
        """计算买入数量"""
        current_price = data.close[0]
        total_value = self.broker.getvalue()
        max_investment = total_value * self.params.position_ratio
        available_cash = self.broker.getcash()
        investment = min(available_cash, max_investment)
        shares = int(investment / current_price / 100) * 100
        return max(shares, 100) if shares >= 100 else 0

    def next(self):
        """策略主逻辑"""
        self.dates.append(self.datas[0].datetime.date(0))
        self.portfolio_values.append(self.broker.getvalue())
        current_date = self.datas[0].datetime.date(0)

        # 卖出检查
        for data_name, indicators in self.indicators.items():
            data = indicators['data']
            position = self.getposition(data)

            if position.size > 0:
                sell_score, sell_details = self.calculate_sell_score(data_name)
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

        # 买入检查
        buy_candidates = []
        for data_name, indicators in self.indicators.items():
            data = indicators['data']
            position = self.getposition(data)

            if position.size > 0:
                continue

            buy_score, buy_details = self.calculate_buy_score(data_name)
            if buy_score >= self.params.buy_threshold:
                buy_candidates.append({
                    'data_name': data_name,
                    'data': data,
                    'score': buy_score,
                    'details': buy_details,
                    'price': data.close[0]
                })

        if buy_candidates:
            buy_candidates.sort(key=lambda x: x['score'], reverse=True)
            current_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)

            for candidate in buy_candidates:
                if current_positions >= self.params.max_positions:
                    break

                size = self.calculate_position_size(candidate['data'], candidate['score'])
                if size >= 100:
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
        """订单通知"""
        pass


def load_stock_data_from_csv(csv_dir='stock_data'):
    """
    从CSV目录加载所有股票数据

    返回:
        dict: {股票名称: (代码, DataFrame)}
    """
    if not os.path.exists(csv_dir):
        print(f"错误: 目录 {csv_dir} 不存在")
        print(f"请先运行 download_stock_data_baostock.py 下载数据")
        return {}

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    if not csv_files:
        print(f"错误: {csv_dir} 目录中没有CSV文件")
        print(f"请先运行 download_stock_data_baostock.py 下载数据")
        return {}

    stock_data = {}
    print(f"\n正在从 {csv_dir} 目录加载数据...")

    for csv_file in csv_files:
        try:
            # 从文件名提取信息: 000001_平安银行.csv
            filename = os.path.basename(csv_file)
            code_name = filename.replace('.csv', '')
            parts = code_name.split('_', 1)

            if len(parts) >= 2:
                code = parts[0]
                name = parts[1]
            else:
                code = code_name
                name = code_name

            # 读取CSV
            df = pd.read_csv(csv_file)

            # 标准化列名
            df.columns = [col.lower() for col in df.columns]

            # 确保有date列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
            else:
                print(f"  ✗ {filename}: 缺少日期列")
                continue

            # 设置索引
            df.set_index('date', inplace=True)

            # 检查必需列
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                print(f"  ✗ {filename}: 缺少必需列 {required}")
                continue

            # 数据清洗
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)

            if len(df) < 70:
                print(f"  ✗ {name}({code}): 数据不足 ({len(df)}条)")
                continue

            stock_data[name] = (code, df)
            print(f"  ✓ {name}({code}): {len(df)}条数据")

        except Exception as e:
            print(f"  ✗ {csv_file}: {str(e)[:60]}")

    return stock_data


def run_backtest(csv_dir='stock_data',
                 initial_cash=100000,
                 buy_threshold=50,
                 sell_threshold=55,
                 max_positions=3,
                 position_ratio=0.35,
                 commission=0.001):
    """运行回测"""

    print(f"\n{'='*70}")
    print(f"多股票综合评分回测系统 (本地CSV数据)")
    print(f"{'='*70}")
    print(f"数据源: {csv_dir} 目录")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"买入阈值: {buy_threshold}分")
    print(f"卖出阈值: {sell_threshold}分")
    print(f"最大持仓: {max_positions}只")
    print(f"单只仓位: {position_ratio*100:.0f}%")
    print(f"{'='*70}")

    # 加载CSV数据
    stock_data = load_stock_data_from_csv(csv_dir)

    if not stock_data:
        return None, None

    print(f"\n成功加载 {len(stock_data)} 只股票")

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

    # 添加数据
    print("\n添加数据到回测引擎...")
    for name, (code, df) in stock_data.items():
        data_feed = bt.feeds.PandasData(dataname=df, name=name)
        cerebro.adddata(data_feed)

    # 设置初始资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f"\n开始回测...")
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    print(f"\n{'='*70}")
    print(f"回测完成！")
    print(f"{'='*70}")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"最终资金: ¥{final_value:,.2f}")
    print(f"总收益: ¥{final_value - initial_cash:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"{'='*70}")

    return cerebro, strat


def print_trade_statistics(strat):
    """打印交易统计"""
    print(f"\n{'='*70}")
    print(f"交易统计")
    print(f"{'='*70}")
    print(f"买入次数: {len(strat.buy_signals)}")
    print(f"卖出次数: {len(strat.sell_signals)}")

    if strat.buy_signals:
        avg_buy_score = np.mean([s['score'] for s in strat.buy_signals])
        print(f"平均买入评分: {avg_buy_score:.2f}分")

    if strat.sell_signals:
        avg_sell_score = np.mean([s['score'] for s in strat.sell_signals])
        print(f"平均卖出评分: {avg_sell_score:.2f}分")

    print(f"{'='*70}")

    if strat.buy_signals:
        print(f"\n前10笔买入信号:")
        for i, signal in enumerate(strat.buy_signals[:10], 1):
            print(f"  {i}. {signal['date']} {signal['stock_name']}: "
                  f"¥{signal['price']:.2f}, {signal['size']}股, 评分{signal['score']:.1f}")

    if strat.sell_signals:
        print(f"\n前10笔卖出信号:")
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
    ax.set_title('多股票轮动策略 - 账户价值变化 (本地CSV数据)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'portfolio_csv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {filename}")
    plt.show()


def save_trade_records(strat):
    """保存交易记录"""
    if not strat.buy_signals and not strat.sell_signals:
        print("\n没有交易记录")
        return

    # 保存买入记录
    if strat.buy_signals:
        df_buy = pd.DataFrame(strat.buy_signals)
        filename = f'buy_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_buy.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n买入记录已保存: {filename}")

    # 保存卖出记录
    if strat.sell_signals:
        df_sell = pd.DataFrame(strat.sell_signals)
        filename = f'sell_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_sell.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"卖出记录已保存: {filename}")


def main():
    """主函数"""
    print("\n=== 多股票综合评分回测系统 (本地CSV版) ===")
    print("数据源: stock_data 目录下的CSV文件")
    print("特点: 离线回测，速度快，可重复运行\n")

    # 回测参数
    csv_dir = "stock_data"
    initial_cash = 100000
    buy_threshold = 50      # 买入阈值
    sell_threshold = 55     # 卖出阈值
    max_positions = 3       # 最大持仓数
    position_ratio = 0.35   # 单只仓位
    commission = 0.001      # 手续费

    # 运行回测
    cerebro, strat = run_backtest(
        csv_dir=csv_dir,
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        position_ratio=position_ratio,
        commission=commission
    )

    if cerebro is None or strat is None:
        print("\n提示: 请先运行 download_stock_data_baostock.py 下载数据")
        return

    # 打印统计
    print_trade_statistics(strat)

    # 绘制图表
    plot_portfolio_value(strat)

    # 保存交易记录
    save_trade_records(strat)

    print("\n回测完成！")


if __name__ == "__main__":
    main()
