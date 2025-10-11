# A股量化交易回测系统
# 使用RSI策略进行回测分析

import akshare as ak
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import quantstats as qs
warnings.filterwarnings('ignore')

# 扩展QuantStats
qs.extend_pandas()

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

# 2. RSI多重滤波器策略
class RSIStrategy(bt.Strategy):
    """
    RSI三指标策略：
    采用评分系统，RSI + 均线 + 成交量 三个核心指标

    过滤器体系（3个核心指标）：
    1. RSI超卖/超买（主信号）
    2. 多周期均线趋势（MA5, MA20, MA60）
    3. 成交量放量确认

    买入条件：评分>=阈值
    """

    params = (
        ('rsi_period', 14),              # RSI周期
        ('rsi_lower', 30),               # RSI下轨
        ('rsi_upper', 70),               # RSI上轨
        ('ma_short', 5),                 # 短期均线
        ('ma_medium', 20),               # 中期均线
        ('ma_long', 60),                 # 长期均线
        ('volume_ma_period', 20),        # 成交量均线周期
        ('volume_ratio_threshold', 1.2), # 放量阈值
        ('buy_score_threshold', 45),     # 买入评分阈值（满分100）
        ('sell_score_threshold', 45),    # 卖出评分阈值（满分100）
    )

    def __init__(self):
        # === 技术指标初始化 ===

        # 1. RSI（相对强弱指标）
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # 2. 多周期均线
        self.ma5 = bt.indicators.SMA(self.data.close, period=self.params.ma_short)
        self.ma20 = bt.indicators.SMA(self.data.close, period=self.params.ma_medium)
        self.ma60 = bt.indicators.SMA(self.data.close, period=self.params.ma_long)

        # 3. 成交量指标
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma_period)

        # 用于记录交易信号和技术指标
        self.buy_signals = []
        self.sell_signals = []

        # 记录每日账户价值用于QuantStats分析
        self.portfolio_values = []
        self.dates = []

        # 记录每日综合评分
        self.daily_buy_scores = []
        self.daily_sell_scores = []


    def calculate_position_ratio_from_score(self, score, is_buy=True):
        """
        根据综合评分计算仓位比例
        score: 综合评分 (0-100)
        is_buy: True表示买入，False表示卖出

        返回: 仓位比例 (0-1之间)
        """
        if is_buy:
            # 买入逻辑：评分越高，买入比例越大
            # 60分：20%仓位
            # 70分：35%仓位
            # 80分：50%仓位
            # 90分：70%仓位
            # 100分：90%仓位
            if score >= 90:
                ratio = 0.7 + (score - 90) * 0.02  # 90分=70%, 100分=90%
            elif score >= 80:
                ratio = 0.5 + (score - 80) * 0.02  # 80分=50%, 90分=70%
            elif score >= 70:
                ratio = 0.35 + (score - 70) * 0.015  # 70分=35%, 80分=50%
            elif score >= 60:
                ratio = 0.2 + (score - 60) * 0.015  # 60分=20%, 70分=35%
            else:
                ratio = 0.1  # 兜底最小仓位
        else:
            # 卖出逻辑：评分越高，卖出比例越大
            # 60分：30%仓位
            # 70分：50%仓位
            # 80分：70%仓位
            # 90分：100%仓位（清仓）
            if score >= 90:
                ratio = 1.0  # 90分以上全部卖出
            elif score >= 80:
                ratio = 0.7 + (score - 80) * 0.03  # 80分=70%, 90分=100%
            elif score >= 70:
                ratio = 0.5 + (score - 70) * 0.02  # 70分=50%, 80分=70%
            elif score >= 60:
                ratio = 0.3 + (score - 60) * 0.02  # 60分=30%, 70分=50%
            else:
                ratio = 0.2  # 兜底最小卖出比例

        return min(max(ratio, 0.1), 1.0)  # 限制在10%-100%之间

    def calculate_position_size(self, buy_or_sell, score):
        """
        根据综合评分计算买入/卖出数量
        buy_or_sell: True表示买入，False表示卖出
        score: 综合评分
        """
        current_price = self.data.close[0]

        if buy_or_sell:
            # 买入：根据可用资金和评分计算
            available_cash = self.broker.get_cash()
            position_ratio = self.calculate_position_ratio_from_score(score, is_buy=True)
            target_value = available_cash * position_ratio
            shares = int(target_value / current_price / 100) * 100  # 向下取整到100股
        else:
            # 卖出：根据当前持仓和评分计算
            current_position = self.position.size
            position_ratio = self.calculate_position_ratio_from_score(score, is_buy=False)
            shares = int(current_position * position_ratio / 100) * 100  # 向下取整到100股

        return max(shares, 100) if shares >= 100 else 0  # 至少100股，否则不交易

    def calculate_buy_score(self):
        """
        计算买入信号评分（满分100分）
        3个核心指标：RSI + 均线 + 成交量 
        """
        score = 0
        details = {}

        # === 过滤器1: RSI超卖（权重50分） ===
        rsi_score=0
        if self.rsi[0] < self.params.rsi_lower:
            if self.rsi[0] > 25:
                rsi_score=10
            elif self.rsi[0] >20:
                rsi_score=25
            elif self.rsi[0] >15:
                rsi_score=30
            elif self.rsi[0] >10:
                rsi_score=40
            elif self.rsi[0] >0:
                rsi_score=50
            # RSI越低，分数越高
            details['RSI'] = f"+{rsi_score:.1f} (RSI={self.rsi[0]:.1f})"
        else:
            details['RSI'] = f"+0 (RSI={self.rsi[0]:.1f})"
        score += rsi_score

        # === 过滤器2: 多周期均线趋势（权重25分） ===
        ma_score = 0
        # 短期趋势：价格>MA5
        if self.ma5[0] >= self.ma20[0] and self.ma5[-1] <= self.ma20[-1]:
            ma_score += 15
        # 长期趋势：MA20>MA60或MA60斜率>=0 
        if self.ma20[0] > self.ma60[0] and self.ma20[-1] <= self.ma60[-1]:
            ma_score += 10
        score += ma_score
        details['均线'] = f"+{ma_score:.1f}"

        # === 过滤器3: 成交量放量（权重25分） ===
        volume_score = 0
        volume_ratio = self.data.volume[0] / self.volume_ma[0] if self.volume_ma[0] > 0 else 0

        if volume_ratio >= 1.5:  # 放量50%以上
            volume_score = 25
        elif volume_ratio >= 1.2:  # 放量20%以上
            volume_score = 18
        elif volume_ratio >= 1.0:  # 持平或略增
            volume_score = 10

        score += volume_score
        details['成交量'] = f"+{volume_score:.1f} (量比={volume_ratio:.2f})"

        return score, details, volume_ratio

    def calculate_sell_score(self):
        """
        计算卖出信号评分（满分100分）
        3个核心指标：RSI + 均线 + 成交量
        卖出逻辑：止盈(RSI高位) 或 趋势转弱 或 破位
        """
        score = 0
        details = {}

        # === 过滤器1: RSI止盈信号（权重40分） ===
        # 不要求必须>70，只要RSI回落到合理区间就开始评分
        if self.rsi[0] > 60:  # RSI进入高位区域(60-100)
            # RSI越高，分数越高
            rsi_score = 50 * (self.rsi[0] - 60) / 40  # 60分得0分，100分得40分
            score += rsi_score
            details['RSI'] = f"+{rsi_score:.1f} (RSI={self.rsi[0]:.1f})"
        elif self.rsi[0] > 50:  # RSI在50-60之间，给少量分数
            rsi_score = 20 * (self.rsi[0] - 50) / 10
            score += rsi_score
            details['RSI'] = f"+{rsi_score:.1f} (RSI={self.rsi[0]:.1f})"
        else:
            details['RSI'] = f"+0 (RSI={self.rsi[0]:.1f})"

        # === 过滤器2: 均线趋势转弱（权重25分） ===
        ma_score = 0
        # 信号2: MA5下降 (+15分) - 短期趋势转向
        if self.ma5[0] < self.ma20[0] and self.ma5[-1] >= self.ma20[-1]:
            ma_score += 15

        # 信号3: 价格接近或跌破MA20 (+10分) - 中期支撑失守
        if self.ma20[0] < self.ma60[0] and self.ma20[-1] >= self.ma60[-1]:
            ma_score += 10
        score += ma_score
        details['均线'] = f"+{ma_score:.1f}"

        # === 过滤器3: 成交量信号（权重20分） ===
        volume_score = 0
        volume_ratio = self.data.volume[0] / self.volume_ma[0] if self.volume_ma[0] > 0 else 0
        if volume_ratio >= 1.5:  # 放量下跌（恐慌盘）
            volume_score = 20
        elif volume_ratio >= 1.2:  # 温和放量
            volume_score = 12
        elif volume_ratio < 0.8:  # 缩量（上涨动能不足）
            volume_score = 8

        score += volume_score
        details['成交量'] = f"+{volume_score:.1f} (量比={volume_ratio:.2f})"

        return score, details

    def next(self):
        # 记录每日账户价值
        self.dates.append(self.data.datetime.date(0))
        self.portfolio_values.append(self.broker.getvalue())

        # === 计算并记录每日综合评分 ===
        buy_score, buy_details, volume_ratio = self.calculate_buy_score()
        sell_score, sell_details = self.calculate_sell_score()

        self.daily_buy_scores.append(buy_score)
        self.daily_sell_scores.append(sell_score)

        # === 买入逻辑：基于多重过滤器评分（支持加仓）===
        # 当评分达到阈值时买入（无论是否持仓都可以买）
        if buy_score >= self.params.buy_score_threshold:
            size = self.calculate_position_size(True, buy_score)  # 传入评分
            if size >= 100:
                self.buy(size=size)
                self.buy_signals.append({
                    'date': self.data.datetime.date(0),
                    'price': self.data.close[0],
                    'rsi': self.rsi[0],
                    'score': buy_score,
                    'details': buy_details,
                    'volume_ratio': volume_ratio,
                    'size': size
                })

        # === 卖出逻辑：持仓时基于多重过滤器评分 ===
        # 只有在持仓且卖出评分达标时才卖出
        if self.position:
            # 当评分达到阈值时卖出
            if sell_score >= self.params.sell_score_threshold:
                size = self.calculate_position_size(False, sell_score)  # 传入评分
                if size >= 100:
                    self.sell(size=size)  # 启用实际卖出

                    # 计算当前量比
                    volume_ratio = self.data.volume[0] / self.volume_ma[0] if self.volume_ma[0] > 0 else 0

                    self.sell_signals.append({
                        'date': self.data.datetime.date(0),
                        'price': self.data.close[0],
                        'rsi': self.rsi[0],
                        'score': sell_score,
                        'details': sell_details,
                        'volume_ratio': volume_ratio,
                        'size': size
                    })

# 4. 回测函数（使用QuantStats分析）
def run_backtest(data, strategy_class, cash=100000, commission=0.001, name="策略"):
    """
    运行回测并使用QuantStats进行分析
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

    print(f'{name} - 初始资金: {cash:,.0f}')

    # 运行回测
    results = cerebro.run()
    strat = results[0]

    # 从策略对象中获取每日账户价值
    portfolio_series = pd.Series(strat.portfolio_values, index=pd.DatetimeIndex(strat.dates))

    # 计算每日收益率
    returns = portfolio_series.pct_change().dropna()

    # 使用QuantStats生成完整报告
    print(f"\n{'='*60}")
    print(f"{name} - QuantStats 分析报告")
    print(f"{'='*60}\n")

    # 显示关键指标
    qs.reports.metrics(returns, mode='full', display=True)

    # 生成可视化报告（HTML格式）
    output_file = f"quantstats_report_{name.replace(' ', '_')}.html"
    qs.reports.html(returns, output=output_file, title=f"{name} 回测报告")
    print(f"\n完整HTML报告已保存至: {output_file}")

    return cerebro, results, strat

# 5. 简洁的买卖点可视化
def plot_trades(data, strategy, stock_name):
    """
    可视化买卖点、RSI、综合评分和量比
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # 子图1: 股价和买卖点
    ax1.plot(data.index, data['close'], label='收盘价', linewidth=1.5, color='navy', alpha=0.7)

    # 标注买入点
    for signal in strategy.buy_signals:
        date, price = signal['date'], signal['price']
        score = signal.get('score', 0)
        details = signal.get('details', {})

        ax1.scatter(date, price, color='red', marker='^', s=180, zorder=5, edgecolors='darkred', linewidth=2.5)
        ax1.axvline(x=date, color='red', linestyle='--', alpha=0.3, linewidth=1)

        # 构建详细信息字符串
        detail_str = '\n'.join([f"{k}:{v}" for k, v in list(details.items())[:3]])

        # 标注信息（包含评分）
        ax1.annotate(f'买入 [{score:.0f}分]\n{detail_str}',
                   xy=(date, price), xytext=(8, 15),
                   textcoords='offset points', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.85),
                   color='white', fontweight='bold')

    # 标注卖出点
    for signal in strategy.sell_signals:
        date, price = signal['date'], signal['price']
        score = signal.get('score', 0)
        details = signal.get('details', {})

        ax1.scatter(date, price, color='green', marker='v', s=180, zorder=5, edgecolors='darkgreen', linewidth=2.5)
        ax1.axvline(x=date, color='green', linestyle='--', alpha=0.3, linewidth=1)

        # 构建详细信息字符串
        detail_str = '\n'.join([f"{k}:{v}" for k, v in list(details.items())[:2]])

        # 标注信息（包含评分）
        ax1.annotate(f'卖出 [{score:.0f}分]\n{detail_str}',
                   xy=(date, price), xytext=(8, -30),
                   textcoords='offset points', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.85),
                   color='white', fontweight='bold')

    ax1.set_ylabel('价格 (元)', fontsize=11)

    # 计算平均评分
    avg_buy_score = np.mean([s.get('score', 0) for s in strategy.buy_signals]) if strategy.buy_signals else 0
    avg_sell_score = np.mean([s.get('score', 0) for s in strategy.sell_signals]) if strategy.sell_signals else 0

    title = f'{stock_name} - 多重滤波器交易分析\n'
    title += f'买入:{len(strategy.buy_signals)}次(均分{avg_buy_score:.0f}) | 卖出:{len(strategy.sell_signals)}次(均分{avg_sell_score:.0f})'
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 子图2: RSI
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

    ax2.plot(data.index, rsi_values, label='RSI', color='purple', linewidth=1.5)
    ax2.axhline(y=70, color='red', linestyle='-', alpha=0.7, linewidth=2, label='超买线(70)')
    ax2.axhline(y=30, color='green', linestyle='-', alpha=0.7, linewidth=2, label='超卖线(30)')
    ax2.fill_between(data.index, 70, 100, alpha=0.15, color='red')
    ax2.fill_between(data.index, 0, 30, alpha=0.15, color='green')

    # 标注买卖点的RSI值
    for signal in strategy.buy_signals:
        ax2.axvline(x=signal['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
    for signal in strategy.sell_signals:
        ax2.axvline(x=signal['date'], color='green', linestyle='--', alpha=0.3, linewidth=1)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 子图3: 综合评分（买入和卖出）
    # 对齐日期：策略开始运行后才有评分数据
    score_dates = strategy.dates
    buy_scores = strategy.daily_buy_scores
    sell_scores = strategy.daily_sell_scores

    ax3.plot(score_dates, buy_scores, label='买入综合评分', color='red', linewidth=2, alpha=0.8)
    # ax3.plot(score_dates, sell_scores, label='卖出综合评分', color='green', linewidth=2, alpha=0.8)

    # 添加阈值线
    ax3.axhline(y=strategy.params.buy_score_threshold, color='red', linestyle='--',
                alpha=0.5, linewidth=1.5, label=f'买入阈值({strategy.params.buy_score_threshold}分)')
    ax3.axhline(y=strategy.params.sell_score_threshold, color='green', linestyle='--',
                alpha=0.5, linewidth=1.5, label=f'卖出阈值({strategy.params.sell_score_threshold}分)')

    # 填充达标区域
    ax3.fill_between(score_dates, strategy.params.buy_score_threshold, 100,
                     alpha=0.1, color='red', label='买入信号区')
    ax3.fill_between(score_dates, strategy.params.sell_score_threshold, 100,
                     alpha=0.1, color='green', label='卖出信号区')

    # 标注实际买卖点
    for signal in strategy.buy_signals:
        ax3.axvline(x=signal['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
        # 标注买入点的评分
        if signal['date'] in score_dates:
            idx = score_dates.index(signal['date'])
            ax3.scatter(signal['date'], buy_scores[idx], color='red',
                       marker='^', s=120, zorder=5, edgecolors='darkred', linewidth=2)

    for signal in strategy.sell_signals:
        ax3.axvline(x=signal['date'], color='green', linestyle='--', alpha=0.3, linewidth=1)
        # 标注卖出点的评分
        if signal['date'] in score_dates:
            idx = score_dates.index(signal['date'])
            ax3.scatter(signal['date'], sell_scores[idx], color='green',
                       marker='v', s=120, zorder=5, edgecolors='darkgreen', linewidth=2)

    ax3.set_ylim(0, 100)
    ax3.set_ylabel('综合评分', fontsize=11)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 子图4: 成交量和量比
    volume_ma = data['volume'].rolling(window=20).mean()
    colors = ['red' if data['close'].iloc[i] >= data['open'].iloc[i] else 'green'
              for i in range(len(data))]

    ax4.bar(data.index, data['volume'], color=colors, alpha=0.5, width=0.8)
    ax4.plot(data.index, volume_ma, color='blue', linewidth=2, alpha=0.8, label='成交量MA20')
    ax4.plot(data.index, volume_ma * 1.2, color='orange', linewidth=1.5,
             linestyle='--', alpha=0.7, label='放量阈值(1.2x)')

    # 标注买卖点
    for signal in strategy.buy_signals:
        ax4.axvline(x=signal['date'], color='red', linestyle='--', alpha=0.3, linewidth=1)
    for signal in strategy.sell_signals:
        ax4.axvline(x=signal['date'], color='green', linestyle='--', alpha=0.3, linewidth=1)

    ax4.set_ylabel('成交量', fontsize=11)
    ax4.set_xlabel('日期', fontsize=11)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'trades_{stock_name}.png', dpi=150, bbox_inches='tight')
    print(f"买卖点图表已保存: trades_{stock_name}.png")
    plt.show()

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

    # 对每只股票进行RSI策略回测
    for code, stock_info in stock_data.items():
        data = stock_info['data']
        name = stock_info['name']

        print(f"\n{'='*60}")
        print(f"开始回测 {name}({code})")
        print(f"数据时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        # RSI策略回测
        print(f"\n--- {name} RSI策略回测 ---")
        try:
            cerebro, _, strat = run_backtest(
                data, RSIStrategy,
                cash=initial_cash,
                commission=commission,
                name=f"RSI策略_{name}({code})"
            )

            # 简单的交易统计
            print(f"\n交易统计:")
            print(f"  买入次数: {len(strat.buy_signals)}")
            print(f"  卖出次数: {len(strat.sell_signals)}")

            final_value = cerebro.broker.getvalue()
            total_return = (final_value - initial_cash) / initial_cash * 100
            print(f"  最终收益率: {total_return:.2f}%")

            # 可视化买卖点
            plot_trades(data, strat, f"{name}({code})")

        except Exception as e:
            print(f"RSI策略回测失败: {e}")

    print(f"\n{'='*80}")
    print("所有回测完成！请查看生成的HTML报告文件。")
    print(f"{'='*80}")


# 运行主程序
if __name__ == "__main__":    
    # 运行回测
    main()