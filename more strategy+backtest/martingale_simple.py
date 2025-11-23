# 马丁格尔策略 - 简洁版
import akshare as ak
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取数据
def get_data(symbol, start="20230101", end=None):
    end = end or datetime.now().strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    df.rename(columns={'日期': 'datetime', '开盘': 'open', '收盘': 'close',
                       '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

# 马丁格尔策略
class Martingale(bt.Strategy):
    params = dict(
        base_size=100,      # 基础仓位
        drop_pct=0.05,      # 下跌5%加仓
        multiplier=2,       # 加倍买入
        max_level=4,        # 最多4次
        profit_pct=0.08     # 盈利8%止盈
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=14)
        self.level = 0
        self.cost = 0
        self.last_price = 0

    def next(self):
        if not self.position:
            # 无仓位：RSI超卖时买入
            if self.rsi < 30:
                self.buy(size=self.p.base_size)
                self.level = 1
                self.cost = self.data.close[0]
                self.last_price = self.data.close[0]
        else:
            # 有仓位：检查止盈或加仓
            avg_price = self.broker.getvalue() / (self.position.size + 1e-6)
            profit = (self.data.close[0] - avg_price) / avg_price

            # 止盈
            if profit >= self.p.profit_pct:
                self.close()
                self.level = 0
                return

            # 加仓：价格下跌且未达上限
            drop = (self.data.close[0] - self.last_price) / self.last_price
            if drop <= -self.p.drop_pct and self.level < self.p.max_level:
                size = self.p.base_size * (self.p.multiplier ** self.level)
                self.buy(size=size)
                self.level += 1
                self.last_price = self.data.close[0]

# 回测
def backtest(data, cash=100000):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Martingale)
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(0.001)

    print(f'初始资金: {cash:,.0f}')
    cerebro.run()
    final = cerebro.broker.getvalue()
    ret = (final - cash) / cash * 100
    print(f'最终资金: {final:,.0f}')
    print(f'收益率: {ret:.2f}%')

    cerebro.plot(style='candlestick', volume=False)

# 主程序
if __name__ == '__main__':
    print("=== 马丁格尔策略回测 ===\n")

    # 获取平安银行数据
    data = get_data('000001')
    print(f"数据: {len(data)}条\n")

    # 回测
    backtest(data)
