# 通用批量回测框架
# 支持任意策略对多只股票进行批量回测

import akshare as ak
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== 通用批量回测系统 ===\n")

# ============== 数据获取 ==============
def get_stock_data(symbol, start_date="20230101", end_date=None):
    """获取单只股票数据"""
    try:
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

        df.rename(columns={
            '日期': 'datetime', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        }, inplace=True)

        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        print(f"  ✗ {symbol} 获取失败: {e}")
        return None

def get_stock_list(index='沪深300', max_stocks=50):
    """获取指数成分股列表"""
    try:
        if index == '沪深300':
            df = ak.index_stock_cons_csindex(symbol="000300")
            stock_codes = df['成分券代码'].head(max_stocks).tolist()
        elif index == '上证50':
            df = ak.index_stock_cons_csindex(symbol="000016")
            stock_codes = df['成分券代码'].head(max_stocks).tolist()
        elif index == '中证500':
            df = ak.index_stock_cons_csindex(symbol="000905")
            stock_codes = df['成分券代码'].head(max_stocks).tolist()
        else:
            stock_codes = []

        return stock_codes
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

# ============== 通用回测引擎 ==============
def backtest_single_stock(data, strategy_class, cash=100000, commission=0.001):
    """对单只股票进行回测"""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    results = cerebro.run()
    strat = results[0]

    # 提取分析结果
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - cash) / cash * 100

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown.get('max', {}).get('drawdown', 0)

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'final_value': final_value,
        'return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate
    }

def batch_backtest(stock_codes, strategy_class, start_date="20230101",
                   cash=100000, commission=0.001, show_progress=True):
    """批量回测多只股票"""
    results = []

    print(f"开始批量回测 {len(stock_codes)} 只股票...")
    print(f"策略: {strategy_class.__name__}")
    print(f"初始资金: {cash:,}, 手续费: {commission*100}%")
    print(f"数据时间: {start_date} ~ {datetime.now().strftime('%Y%m%d')}")
    print("-" * 70)

    success_count = 0
    fail_count = 0

    for i, code in enumerate(stock_codes):
        if show_progress:
            print(f"[{i+1}/{len(stock_codes)}] {code}...", end=' ')

        # 获取数据
        data = get_stock_data(code, start_date=start_date)

        if data is None or len(data) < 100:
            if show_progress:
                print("✗ 数据不足")
            fail_count += 1
            continue

        # 回测
        try:
            result = backtest_single_stock(data, strategy_class, cash, commission)
            result['code'] = code
            result['name'] = code  # 可以后续添加股票名称
            results.append(result)

            if show_progress:
                print(f"✓ 收益: {result['return']:>7.2f}%, 交易: {result['total_trades']:>3}次")

            success_count += 1
        except Exception as e:
            if show_progress:
                print(f"✗ 回测失败: {e}")
            fail_count += 1

    print("-" * 70)
    print(f"回测完成: 成功 {success_count}, 失败 {fail_count}")

    return pd.DataFrame(results)

# ============== 结果分析 ==============
def analyze_results(results_df):
    """分析回测结果"""
    if results_df.empty:
        print("没有回测结果")
        return

    print("\n" + "="*70)
    print("批量回测结果分析")
    print("="*70)

    # 基础统计
    print(f"\n总样本数: {len(results_df)}")
    print(f"\n收益率统计:")
    print(f"  平均收益率: {results_df['return'].mean():>8.2f}%")
    print(f"  中位数收益率: {results_df['return'].median():>8.2f}%")
    print(f"  最高收益率: {results_df['return'].max():>8.2f}%")
    print(f"  最低收益率: {results_df['return'].min():>8.2f}%")
    print(f"  收益率标准差: {results_df['return'].std():>8.2f}%")

    positive = len(results_df[results_df['return'] > 0])
    negative = len(results_df[results_df['return'] <= 0])
    print(f"\n盈亏分布:")
    print(f"  盈利股票数: {positive} ({positive/len(results_df)*100:.1f}%)")
    print(f"  亏损股票数: {negative} ({negative/len(results_df)*100:.1f}%)")

    print(f"\n交易统计:")
    print(f"  平均交易次数: {results_df['total_trades'].mean():.1f}")
    print(f"  平均胜率: {results_df['win_rate'].mean():.1f}%")

    print(f"\n风险指标:")
    print(f"  平均最大回撤: {results_df['max_drawdown'].mean():.2f}%")
    print(f"  平均夏普比率: {results_df['sharpe'].mean():.2f}")

    # Top & Bottom
    print(f"\n表现最好的5只股票:")
    top5 = results_df.nlargest(5, 'return')[['code', 'return', 'total_trades', 'win_rate', 'max_drawdown']]
    for idx, row in top5.iterrows():
        print(f"  {row['code']}: {row['return']:>7.2f}%, 交易{row['total_trades']}次, 胜率{row['win_rate']:.1f}%, 回撤{row['max_drawdown']:.1f}%")

    print(f"\n表现最差的5只股票:")
    bottom5 = results_df.nsmallest(5, 'return')[['code', 'return', 'total_trades', 'win_rate', 'max_drawdown']]
    for idx, row in bottom5.iterrows():
        print(f"  {row['code']}: {row['return']:>7.2f}%, 交易{row['total_trades']}次, 胜率{row['win_rate']:.1f}%, 回撤{row['max_drawdown']:.1f}%")

def plot_results(results_df, strategy_name="策略"):
    """可视化回测结果"""
    if results_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 收益率分布直方图
    ax1 = axes[0, 0]
    ax1.hist(results_df['return'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(results_df['return'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均值: {results_df["return"].mean():.2f}%')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('收益率 (%)')
    ax1.set_ylabel('股票数量')
    ax1.set_title(f'{strategy_name} - 收益率分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 收益率排序图
    ax2 = axes[0, 1]
    sorted_returns = results_df['return'].sort_values(ascending=False)
    colors = ['green' if x > 0 else 'red' for x in sorted_returns]
    ax2.bar(range(len(sorted_returns)), sorted_returns, color=colors, alpha=0.6)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('股票排名')
    ax2.set_ylabel('收益率 (%)')
    ax2.set_title(f'{strategy_name} - 收益率排名')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 收益率 vs 交易次数
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['total_trades'], results_df['return'],
                          alpha=0.6, c=results_df['win_rate'], cmap='RdYlGn', s=50)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('交易次数')
    ax3.set_ylabel('收益率 (%)')
    ax3.set_title(f'{strategy_name} - 收益率 vs 交易次数')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='胜率 (%)')

    # 4. 收益率 vs 最大回撤
    ax4 = axes[1, 1]
    ax4.scatter(results_df['max_drawdown'], results_df['return'], alpha=0.6, color='purple', s=50)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('最大回撤 (%)')
    ax4.set_ylabel('收益率 (%)')
    ax4.set_title(f'{strategy_name} - 收益率 vs 最大回撤')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============== 示例策略（可替换） ==============
class RSIStrategy(bt.Strategy):
    """RSI策略示例"""
    params = (('rsi_period', 14), ('rsi_lower', 30), ('rsi_upper', 70))

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)

    def next(self):
        if not self.position:
            if self.rsi < self.params.rsi_lower:
                self.buy(size=int(self.broker.get_cash() / self.data.close[0] / 100) * 100)
        elif self.rsi > self.params.rsi_upper:
            self.close()

class MartingaleStrategy(bt.Strategy):
    """马丁格尔策略示例"""
    params = (
        ('base_size', 100), ('drop_pct', 0.05),
        ('multiplier', 2), ('max_level', 4), ('profit_pct', 0.08)
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=14)
        self.level = 0
        self.last_price = 0

    def next(self):
        if not self.position:
            if self.rsi < 30:
                self.buy(size=self.params.base_size)
                self.level = 1
                self.last_price = self.data.close[0]
        else:
            avg_price = self.broker.getvalue() / (self.position.size + 1e-6)
            profit = (self.data.close[0] - avg_price) / avg_price

            if profit >= self.params.profit_pct:
                self.close()
                self.level = 0
                return

            drop = (self.data.close[0] - self.last_price) / self.last_price
            if drop <= -self.params.drop_pct and self.level < self.params.max_level:
                size = self.params.base_size * (self.params.multiplier ** self.level)
                self.buy(size=size)
                self.level += 1
                self.last_price = self.data.close[0]

# ============== 主程序 ==============
def main():
    # 选择股票池
    print("正在获取股票列表...")

    # 方式1: 使用指数成分股
    stock_codes = get_stock_list('沪深300', max_stocks=50)

    # 方式2: 手动指定股票
    # stock_codes = ['000001', '000002', '600036', '600519', '000858',
    #                '601318', '600276', '000333', '002475', '300750']

    if not stock_codes:
        print("未能获取股票列表，使用默认列表")
        stock_codes = ['000001', '000002', '600036', '600519', '000858']

    print(f"共选择 {len(stock_codes)} 只股票\n")

    # 选择策略
    strategy = MartingaleStrategy  # 可改为 MartingaleStrategy 或其他策略

    # 批量回测
    results_df = batch_backtest(
        stock_codes=stock_codes,
        strategy_class=strategy,
        start_date="20230101",
        cash=100000,
        commission=0.001
    )

    # 分析结果
    if not results_df.empty:
        analyze_results(results_df)

        # 保存结果
        results_df.to_csv(f'backtest_results_{strategy.__name__}_{datetime.now().strftime("%Y%m%d")}.csv',
                         index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 CSV 文件")

        # 可视化
        plot_results(results_df, strategy.__name__)
    else:
        print("没有有效的回测结果")

if __name__ == "__main__":
    main()
