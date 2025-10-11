# 多股票综合评分回测系统
# 策略：综合评分>=70分时建仓，达到卖出条件时全部清仓
# 特点：只要有资金就持续寻找买入机会，支持多只股票轮动

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiStockStrategy:
    """
    多股票轮动策略
    核心逻辑：
    1. 每日扫描所有股票，计算综合评分
    2. 评分>=买入阈值且有资金时建仓
    3. 持仓股票评分达到卖出条件时全部清仓
    4. 支持同时持有多只股票
    """

    def __init__(self, initial_cash=100000, buy_threshold=70, sell_threshold=60,
                 max_positions=5, single_position_ratio=0.3,
                 rsi_period=14, rsi_lower=30, rsi_upper=70,
                 ma_short=5, ma_medium=20, ma_long=60,
                 volume_ma_period=20, commission=0.001):
        """
        初始化策略参数

        Parameters:
        -----------
        initial_cash : float
            初始资金
        buy_threshold : float
            买入评分阈值（>=此分数才买入）
        sell_threshold : float
            卖出评分阈值（>=此分数全部卖出）
        max_positions : int
            最大持仓股票数量
        single_position_ratio : float
            单只股票最大仓位比例（相对于总资金）
        commission : float
            交易手续费率
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_positions = max_positions
        self.single_position_ratio = single_position_ratio
        self.commission = commission

        # 技术指标参数
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.volume_ma_period = volume_ma_period

        # 持仓记录
        self.positions = {}  # {stock_code: {'shares': 100, 'cost': 10.0, 'buy_date': date}}

        # 交易记录
        self.trades = []

        # 每日账户价值记录
        self.daily_values = []

    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        if len(prices) < period + 1:
            return 50

        deltas = np.diff(prices[-period-1:])
        gains = deltas[deltas >= 0].sum() if len(deltas[deltas >= 0]) > 0 else 0
        losses = -deltas[deltas < 0].sum() if len(deltas[deltas < 0]) > 0 else 0

        if losses == 0:
            return 100
        if gains == 0:
            return 0

        rs = (gains / period) / (losses / period)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_score(self, df, idx):
        """
        计算指定日期的综合评分
        idx: 数据索引位置
        """
        if idx < self.ma_long:
            return 0, {}

        # 获取当前数据
        close_price = df.iloc[idx]['close']
        volume = df.iloc[idx]['volume']

        # 计算RSI
        close_prices = df.iloc[:idx+1]['close'].values
        rsi = self.calculate_rsi(close_prices, self.rsi_period)

        # 计算均线
        ma5 = df.iloc[max(0, idx-self.ma_short+1):idx+1]['close'].mean()
        ma20 = df.iloc[max(0, idx-self.ma_medium+1):idx+1]['close'].mean()
        ma60 = df.iloc[max(0, idx-self.ma_long+1):idx+1]['close'].mean()
        ma60_prev = df.iloc[max(0, idx-self.ma_long):idx]['close'].mean() if idx > self.ma_long else ma60

        # 计算成交量比
        volume_ma = df.iloc[max(0, idx-self.volume_ma_period+1):idx+1]['volume'].mean()
        volume_ratio = volume / volume_ma if volume_ma > 0 else 0

        # === 买入评分系统 ===
        score = 0
        details = {}

        # 1. RSI超卖（权重40分）
        rsi_score = 0
        if rsi < self.rsi_lower:
            rsi_score = 40 * (self.rsi_lower - rsi) / self.rsi_lower
        score += rsi_score

        # 2. 均线趋势（权重35分）
        ma_score = 0
        if close_price > ma5:
            ma_score += 12
        if ma5 > ma20:
            ma_score += 12
        if ma20 > ma60 or ma60 >= ma60_prev:
            ma_score += 11
        score += ma_score

        # 3. 成交量放量（权重25分）
        volume_score = 0
        if volume_ratio >= 1.5:
            volume_score = 25
        elif volume_ratio >= 1.2:
            volume_score = 18
        elif volume_ratio >= 1.0:
            volume_score = 10
        score += volume_score

        details = {
            'RSI': rsi,
            'RSI得分': rsi_score,
            'MA5': ma5,
            'MA20': ma20,
            'MA60': ma60,
            '均线得分': ma_score,
            '量比': volume_ratio,
            '成交量得分': volume_score
        }

        return score, details

    def calculate_sell_score(self, df, idx):
        """
        计算卖出评分
        """
        if idx < self.ma_long:
            return 0, {}

        close_price = df.iloc[idx]['close']
        volume = df.iloc[idx]['volume']

        # 计算RSI
        close_prices = df.iloc[:idx+1]['close'].values
        rsi = self.calculate_rsi(close_prices, self.rsi_period)

        # 计算均线
        ma5 = df.iloc[max(0, idx-self.ma_short+1):idx+1]['close'].mean()
        ma5_prev = df.iloc[max(0, idx-self.ma_short):idx]['close'].mean() if idx > self.ma_short else ma5
        ma20 = df.iloc[max(0, idx-self.ma_medium+1):idx+1]['close'].mean()

        # 计算成交量比
        volume_ma = df.iloc[max(0, idx-self.volume_ma_period+1):idx+1]['volume'].mean()
        volume_ratio = volume / volume_ma if volume_ma > 0 else 0

        # === 卖出评分系统 ===
        score = 0
        details = {}

        # 1. RSI止盈信号（权重40分）
        rsi_score = 0
        if rsi > 60:
            rsi_score = 40 * (rsi - 60) / 40
        elif rsi > 50:
            rsi_score = 10 * (rsi - 50) / 10
        score += rsi_score

        # 2. 均线趋势转弱（权重40分）
        ma_score = 0
        if close_price < ma5:
            ma_score += 15
        if ma5 < ma5_prev:
            ma_score += 15
        if close_price < ma20 * 1.02:
            ma_score += 10
        score += ma_score

        # 3. 成交量信号（权重20分）
        volume_score = 0
        if volume_ratio >= 1.5:
            volume_score = 20
        elif volume_ratio >= 1.2:
            volume_score = 12
        elif volume_ratio < 0.8:
            volume_score = 8
        score += volume_score

        details = {
            'RSI': rsi,
            'RSI得分': rsi_score,
            '均线得分': ma_score,
            '成交量得分': volume_score
        }

        return score, details

    def buy_stock(self, stock_code, stock_name, price, date, score, details):
        """
        买入股票
        """
        # 计算可用资金（单只股票不超过总资金的single_position_ratio）
        max_investment = self.initial_cash * self.single_position_ratio

        # 实际投入金额
        investment = min(self.cash, max_investment)

        if investment < price * 100:  # 至少买100股
            return False

        # 计算购买股数（向下取整到100股）
        shares = int(investment / price / 100) * 100

        if shares < 100:
            return False

        # 计算总成本（含手续费）
        cost = shares * price * (1 + self.commission)

        if cost > self.cash:
            return False

        # 扣除资金
        self.cash -= cost

        # 记录持仓
        self.positions[stock_code] = {
            'name': stock_name,
            'shares': shares,
            'cost_price': price,
            'buy_date': date,
            'buy_score': score
        }

        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'amount': cost,
            'score': score,
            'details': details,
            'cash_after': self.cash
        })

        return True

    def sell_stock(self, stock_code, price, date, score, details):
        """
        卖出股票（全部清仓）
        """
        if stock_code not in self.positions:
            return False

        position = self.positions[stock_code]
        shares = position['shares']

        # 计算卖出收入（扣除手续费）
        revenue = shares * price * (1 - self.commission)

        # 增加资金
        self.cash += revenue

        # 计算收益
        cost = shares * position['cost_price'] * (1 + self.commission)
        profit = revenue - cost
        profit_rate = (profit / cost) * 100

        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'stock_name': position['name'],
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'amount': revenue,
            'profit': profit,
            'profit_rate': profit_rate,
            'hold_days': (date - position['buy_date']).days,
            'buy_price': position['cost_price'],
            'buy_score': position['buy_score'],
            'sell_score': score,
            'details': details,
            'cash_after': self.cash
        })

        # 删除持仓
        del self.positions[stock_code]

        return True

    def get_portfolio_value(self, stock_prices):
        """
        计算当前账户总价值
        stock_prices: {stock_code: current_price}
        """
        stock_value = 0
        for code, position in self.positions.items():
            if code in stock_prices:
                stock_value += position['shares'] * stock_prices[code]

        return self.cash + stock_value

    def run_backtest(self, stock_pool, start_date, end_date):
        """
        运行回测

        Parameters:
        -----------
        stock_pool : dict
            股票池 {code: name}
        start_date : str
            开始日期 'YYYYMMDD'
        end_date : str
            结束日期 'YYYYMMDD'
        """
        print(f"\n{'='*60}")
        print(f"开始回测...")
        print(f"股票池: {len(stock_pool)} 只股票")
        print(f"时间范围: {start_date} ~ {end_date}")
        print(f"初始资金: ¥{self.initial_cash:,.2f}")
        print(f"买入阈值: {self.buy_threshold}分")
        print(f"卖出阈值: {self.sell_threshold}分")
        print(f"最大持仓: {self.max_positions}只")
        print(f"单只仓位: {self.single_position_ratio*100:.0f}%")
        print(f"{'='*60}\n")

        # 1. 获取所有股票数据
        print("正在获取股票数据...")
        stock_data = {}

        for code, name in tqdm(stock_pool.items(), desc="加载数据"):
            try:
                df = self.get_stock_data(code, start_date, end_date)
                if df is not None and len(df) >= self.ma_long + 10:
                    stock_data[code] = {
                        'name': name,
                        'data': df
                    }
            except:
                continue

        if not stock_data:
            print("无法获取股票数据")
            return

        print(f"成功加载 {len(stock_data)} 只股票数据\n")

        # 2. 获取所有交易日期
        all_dates = set()
        for stock_info in stock_data.values():
            all_dates.update(stock_info['data']['date'].tolist())

        trading_dates = sorted(list(all_dates))

        if not trading_dates:
            print("没有交易日期数据")
            return

        print(f"回测交易日数: {len(trading_dates)}\n")
        print("开始模拟交易...\n")

        # 3. 逐日回测
        for current_date in tqdm(trading_dates, desc="回测进度"):

            # 当日所有股票的价格（用于计算账户价值）
            daily_prices = {}

            # 当日所有股票的买入评分
            buy_candidates = []

            # === 步骤1: 检查卖出信号（先卖出，释放资金）===
            for stock_code in list(self.positions.keys()):
                stock_info = stock_data.get(stock_code)
                if not stock_info:
                    continue

                df = stock_info['data']
                date_match = df[df['date'] == current_date]

                if date_match.empty:
                    continue

                idx = df[df['date'] == current_date].index[0]
                current_price = df.iloc[idx]['close']
                daily_prices[stock_code] = current_price

                # 计算卖出评分
                sell_score, sell_details = self.calculate_sell_score(df, idx)

                # 如果达到卖出条件，全部清仓
                if sell_score >= self.sell_threshold:
                    self.sell_stock(stock_code, current_price, current_date, sell_score, sell_details)

            # === 步骤2: 扫描所有股票的买入机会 ===
            for stock_code, stock_info in stock_data.items():
                # 跳过已持仓的股票
                if stock_code in self.positions:
                    df = stock_info['data']
                    date_match = df[df['date'] == current_date]
                    if not date_match.empty:
                        idx = date_match.index[0]
                        daily_prices[stock_code] = df.iloc[idx]['close']
                    continue

                df = stock_info['data']
                date_match = df[df['date'] == current_date]

                if date_match.empty:
                    continue

                idx = date_match.index[0]
                current_price = df.iloc[idx]['close']
                daily_prices[stock_code] = current_price

                # 计算买入评分
                buy_score, buy_details = self.calculate_score(df, idx)

                # 如果评分达标，加入候选列表
                if buy_score >= self.buy_threshold:
                    buy_candidates.append({
                        'code': stock_code,
                        'name': stock_info['name'],
                        'price': current_price,
                        'score': buy_score,
                        'details': buy_details
                    })

            # === 步骤3: 执行买入（按评分排序，优先买入高分股票）===
            if buy_candidates:
                # 按评分降序排序
                buy_candidates.sort(key=lambda x: x['score'], reverse=True)

                for candidate in buy_candidates:
                    # 检查是否还有资金和持仓空间
                    if len(self.positions) >= self.max_positions:
                        break

                    if self.cash < candidate['price'] * 100:
                        break

                    # 买入
                    self.buy_stock(
                        candidate['code'],
                        candidate['name'],
                        candidate['price'],
                        current_date,
                        candidate['score'],
                        candidate['details']
                    )

            # === 记录每日账户价值 ===
            portfolio_value = self.get_portfolio_value(daily_prices)
            self.daily_values.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        print("\n回测完成！\n")

    def get_stock_data(self, code, start_date, end_date):
        """获取股票数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df is None or df.empty:
                return None

            df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            }, inplace=True)

            df['date'] = pd.to_datetime(df['date'])
            return df

        except:
            return None

    def print_summary(self):
        """打印回测摘要"""
        if not self.daily_values:
            print("没有回测数据")
            return

        final_value = self.daily_values[-1]['value']
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        # 统计交易次数
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        # 统计盈亏
        if sell_trades:
            profits = [t['profit'] for t in sell_trades]
            profit_rates = [t['profit_rate'] for t in sell_trades]
            win_trades = [t for t in sell_trades if t['profit'] > 0]
            win_rate = len(win_trades) / len(sell_trades) * 100

            avg_profit = np.mean(profits)
            avg_profit_rate = np.mean(profit_rates)
            max_profit = max(profits)
            max_loss = min(profits)
        else:
            win_rate = 0
            avg_profit = 0
            avg_profit_rate = 0
            max_profit = 0
            max_loss = 0

        # 计算最大回撤
        values = [d['value'] for d in self.daily_values]
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        print(f"\n{'='*60}")
        print(f"{'回测摘要':^50}")
        print(f"{'='*60}")
        print(f"初始资金: ¥{self.initial_cash:,.2f}")
        print(f"最终资金: ¥{final_value:,.2f}")
        print(f"总收益: ¥{final_value - self.initial_cash:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"\n{'='*60}")
        print(f"交易统计:")
        print(f"  买入次数: {len(buy_trades)}")
        print(f"  卖出次数: {len(sell_trades)}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  平均收益: ¥{avg_profit:,.2f}")
        print(f"  平均收益率: {avg_profit_rate:.2f}%")
        print(f"  最大盈利: ¥{max_profit:,.2f}")
        print(f"  最大亏损: ¥{max_loss:,.2f}")
        print(f"{'='*60}\n")

        # 打印持仓明细
        if self.positions:
            print(f"当前持仓 ({len(self.positions)}只):")
            for code, pos in self.positions.items():
                print(f"  {pos['name']}({code}): {pos['shares']}股, 成本¥{pos['cost_price']:.2f}")
        else:
            print("当前无持仓")

    def print_trade_details(self, top_n=10):
        """打印交易明细"""
        if not self.trades:
            print("没有交易记录")
            return

        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        if not sell_trades:
            print("还没有完成的交易")
            return

        print(f"\n{'='*80}")
        print(f"交易明细 (显示前{top_n}笔完成的交易)")
        print(f"{'='*80}\n")

        for i, trade in enumerate(sell_trades[:top_n], 1):
            print(f"第{i}笔: {trade['stock_name']}({trade['stock_code']})")
            print(f"  买入评分: {trade['buy_score']:.1f}分 | 卖出评分: {trade['sell_score']:.1f}分")
            print(f"  买入价: ¥{trade['buy_price']:.2f} | 卖出价: ¥{trade['price']:.2f}")
            print(f"  持有天数: {trade['hold_days']}天")
            print(f"  收益: ¥{trade['profit']:,.2f} ({trade['profit_rate']:.2f}%)")
            print()

    def plot_results(self):
        """绘制回测结果"""
        if not self.daily_values:
            print("没有数据可绘制")
            return

        df = pd.DataFrame(self.daily_values)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 子图1: 账户价值曲线
        ax1.plot(df['date'], df['value'], label='账户总值', linewidth=2, color='navy')
        ax1.axhline(y=self.initial_cash, color='red', linestyle='--', alpha=0.7, label='初始资金')

        # 标注买卖点
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        for trade in buy_trades:
            date_match = df[df['date'] == trade['date']]
            if not date_match.empty:
                value = date_match.iloc[0]['value']
                ax1.scatter(trade['date'], value, color='red', marker='^', s=50, alpha=0.6)

        for trade in sell_trades:
            date_match = df[df['date'] == trade['date']]
            if not date_match.empty:
                value = date_match.iloc[0]['value']
                ax1.scatter(trade['date'], value, color='green', marker='v', s=50, alpha=0.6)

        ax1.set_ylabel('账户价值 (元)', fontsize=11)
        ax1.set_title('账户价值变化', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 子图2: 持仓数量
        ax2.plot(df['date'], df['positions'], label='持仓股票数', linewidth=2, color='orange')
        ax2.fill_between(df['date'], 0, df['positions'], alpha=0.3, color='orange')
        ax2.set_ylabel('持仓数量', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.set_title('持仓股票数量变化', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'backtest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {filename}")
        plt.show()

    def save_results(self):
        """保存回测结果"""
        if not self.trades:
            print("没有交易记录")
            return

        # 保存交易记录
        df_trades = pd.DataFrame(self.trades)
        filename = f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_trades.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"交易记录已保存: {filename}")

        # 保存每日价值
        df_daily = pd.DataFrame(self.daily_values)
        filename2 = f'daily_values_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_daily.to_csv(filename2, index=False, encoding='utf-8-sig')
        print(f"每日价值已保存: {filename2}")


def main():
    """主函数"""
    print("=== 多股票综合评分回测系统 ===")
    print("策略: 评分>=阈值时建仓，达到卖出条件全部清仓\n")

    # 定义股票池（可以自定义）
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
        '601166': '兴业银行'
    }

    # 回测参数
    initial_cash = 100000  # 初始资金10万
    buy_threshold = 70     # 买入评分阈值
    sell_threshold = 60    # 卖出评分阈值
    max_positions = 3      # 最大持仓3只
    single_position_ratio = 0.35  # 单只股票最大35%仓位

    # 回测时间
    start_date = "20230101"
    end_date = "20241231"

    # 创建策略实例
    strategy = MultiStockStrategy(
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        single_position_ratio=single_position_ratio,
        commission=0.001
    )

    # 运行回测
    strategy.run_backtest(stock_pool, start_date, end_date)

    # 显示结果
    strategy.print_summary()
    strategy.print_trade_details(top_n=10)

    # 绘制图表
    strategy.plot_results()

    # 保存结果
    strategy.save_results()


if __name__ == "__main__":
    main()
