"""
基于随机森林的最优选股回测策略
==============================

策略核心思想:
1. 使用随机森林机器学习模型预测股票未来5日上涨概率
2. 每日从股票池中选择预测概率最高的单只股票买入（概率需>60%）
3. 当持仓股票概率降至<50%时卖出止损
4. 采用单股票集中持仓，追求高确定性而非分散

关键参数:
- 买入阈值: 60% (只买概率>60%的股票)
- 卖出阈值: 50% (概率<50%时止损)
- 仓位比例: 95% (保留5%现金应对手续费)
- 初始资金: 100,000元

数据划分:
- 训练期: 2022-01-01 至 2025-06-30 (训练模型)
- 回测期: 2025-07-01 至 2025-11-21 (验证策略)

"""

import pandas as pd
import numpy as np
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from hs300_stocks import get_hs300_stock_pool  # 导入沪深300成分股

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号显示

warnings.filterwarnings('ignore')

# ==================== 数据处理模块 ====================

def get_stock_data(code, name, csv_dir='stock_data'):
    """
    从本地CSV文件读取股票历史数据

    参数:
        code: 股票代码，如 '000001'
        name: 股票名称，如 '平安银行'
        csv_dir: CSV文件目录，默认 'stock_data'

    返回:
        DataFrame: 包含 date, open, high, low, close, volume 列
        None: 如果文件不存在或数据不足

    CSV文件命名格式: {code}_{name}.csv
    例如: 000001_平安银行.csv
    """
    try:
        csv_file = os.path.join(csv_dir, f"{code}_{name}.csv")
        if not os.path.exists(csv_file):
            return None

        df = pd.read_csv(csv_file)
        df.columns = [col.lower() for col in df.columns]

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
        else:
            return None

        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=required, inplace=True)

        if len(df) < 100:
            return None

        return df

    except Exception:
        return None


def calculate_features(df, for_training=True):
    """
    计算技术指标特征（共10个特征）

    - 历史特征：使用截至昨日收盘的数据（开盘时已知）
    - 开盘特征：使用当日开盘价（开盘时已知）
    - 确保所有特征在开盘时都可获得，无时间穿越

    参数:
        df: 包含OHLCV数据的DataFrame
        for_training: 是否用于训练（True时计算标签，False时不计算）

    返回:
        df: 添加了特征列和标签列的DataFrame

    特征说明（前8个为历史特征，后2个为开盘特征）:
        历史特征（基于昨日及之前数据）:
        1. return_1d_prev: 昨日收益率（短期动量）
        2. return_5d_prev: 5日收益率截至昨日（中期动量）
        3. return_10d_prev: 10日收益率截至昨日（长期动量）
        4. ma_ratio_5_20_prev: 昨日5日均线/20日均线（均线位置）
        5. volume_ratio_prev: 昨日成交量/5日均量（量能变化）
        6. rsi_prev: 昨日RSI指标（超买超卖）
        7. volatility_prev: 昨日波动率（风险水平）
        8. bb_position_prev: 昨日布林带位置（价格相对位置）

        开盘特征（基于当日开盘价）:
        9. open_gap: 开盘跳空 = 今开盘/昨收盘 - 1（隔夜变化）
        10. open_vs_ma5: 开盘价/昨日5日均线 - 1（开盘强度）

    标签定义（仅训练时计算）:
        label=1: 未来5日涨幅>3% (正样本)
        label=0: 未来5日涨幅≤3% (负样本)
    """
    # === 第一部分：计算原始指标（基于收盘价） ===

    # 价格收益率
    return_1d = df['close'].pct_change(1)
    return_5d = df['close'].pct_change(5)
    return_10d = df['close'].pct_change(10)

    # 均线
    ma5 = df['close'].rolling(5).mean()
    ma10 = df['close'].rolling(10).mean()
    ma20 = df['close'].rolling(20).mean()
    ma_ratio_5_20 = ma5 / ma20

    # 成交量
    volume_ma5 = df['volume'].rolling(5).mean()
    volume_ratio = df['volume'] / volume_ma5

    # RSI指标
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # 波动率
    volatility = return_1d.rolling(20).std()

    # 布林带
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # === 第二部分：向后偏移1天（让特征对齐到"截至昨日"） ===

    df['return_1d_prev'] = return_1d.shift(1)      # 昨日收益率
    df['return_5d_prev'] = return_5d.shift(1)      # 5日收益率（截至昨日）
    df['return_10d_prev'] = return_10d.shift(1)    # 10日收益率（截至昨日）
    df['ma_ratio_5_20_prev'] = ma_ratio_5_20.shift(1)  # 昨日均线比
    df['volume_ratio_prev'] = volume_ratio.shift(1)    # 昨日量比
    df['rsi_prev'] = rsi.shift(1)                  # 昨日RSI
    df['volatility_prev'] = volatility.shift(1)    # 昨日波动率
    df['bb_position_prev'] = bb_position.shift(1)  # 昨日布林带位置

    # === 第三部分：计算开盘价特征（当日开盘时可获得） ===

    df['open_gap'] = df['open'] / df['close'].shift(1) - 1  # 开盘跳空
    df['open_vs_ma5'] = df['open'] / ma5.shift(1) - 1       # 开盘价相对均线位置

    # === 第四部分：标签（未来5日涨幅） - 仅训练时计算 ===

    if for_training:
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['label'] = (df['future_return'] > 0.03).astype(int)

    return df


def prepare_training_data(stock_pool, train_start, train_end):
    """
    准备训练数据（严格按时间范围，避免数据泄露）

    参数:
        stock_pool: 股票池字典 {代码: 名称}
        train_start: 训练开始日期，如 '2022-01-01'
        train_end: 训练结束日期，如 '2025-06-30'

    返回:
        combined: 合并后的DataFrame，包含所有股票的特征数据
        feature_cols: 特征列名列表

    重要: 训练数据必须早于回测数据，严格时间序列分割
    """
    all_data = []

    print(f"\n正在加载训练数据 ({train_start} 至 {train_end})...")
    for code, name in stock_pool.items():
        df = get_stock_data(code, name, csv_dir='stock_data')
        if df is not None:
            # 过滤训练时间段
            df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            if len(df) < 100:
                continue

            df = calculate_features(df)
            df['stock_code'] = code
            df['stock_name'] = name
            all_data.append(df)
            print(f"  + {name}({code}): {len(df)}条")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna()

    # 10个特征（8个历史特征 + 2个开盘特征）
    feature_cols = ['return_1d_prev', 'return_5d_prev', 'return_10d_prev',
                   'ma_ratio_5_20_prev', 'volume_ratio_prev', 'rsi_prev',
                   'volatility_prev', 'bb_position_prev',
                   'open_gap', 'open_vs_ma5']

    return combined, feature_cols


def train_rf_model(combined, feature_cols):
    """
    训练随机森林分类模型

    参数:
        combined: 包含特征和标签的DataFrame
        feature_cols: 特征列名列表

    返回:
        rf: 训练好的RandomForestClassifier模型
        feature_cols: 特征列名（用于后续预测）

    模型参数:
        n_estimators=100: 使用100棵决策树
        max_depth=10: 每棵树最大深度10（防止过拟合）
        random_state=42: 随机种子（保证结果可复现）
        class_weight='balanced': 自动平衡正负样本权重
    """
    X = combined[feature_cols]
    y = combined['label']

    print(f"\n训练数据: {len(combined)}条")
    print(f"日期范围: {combined['date'].min()} 至 {combined['date'].max()}")
    print(f"正样本比例: {y.mean():.2%}")

    print("\n训练随机森林模型...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X, y)

    # 训练集准确率
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"训练集准确率: {acc:.2%}")

    return rf, feature_cols


# ==================== Backtrader策略模块 ====================

class RFBestStockStrategy(bt.Strategy):
    """
    基于随机森林的最优选股策略（Backtrader策略类）

    交易逻辑:
        买入规则:
            1. 当前无持仓
            2. 计算股票池中所有股票的上涨概率
            3. 筛选出概率 > buy_threshold (60%) 的候选股票
            4. 选择概率最高的那只股票
            5. 使用95%资金买入，按100股整数倍

        卖出规则:
            1. 持有股票时，每日计算其上涨概率
            2. 当概率 < sell_threshold (50%) 时卖出
            3. 全部清仓，等待下次买入机会

    参数说明:
        buy_threshold: 买入概率阈值（默认0.60，即60%）
        sell_threshold: 卖出概率阈值（默认0.50，即50%）
        position_ratio: 单只股票仓位比例（默认0.95，保留5%现金）
        model: 训练好的随机森林模型
        feature_cols: 特征列名列表
        stock_pool: 股票池字典
    """

    params = (
        ('buy_threshold', 0.60),    # 买入概率阈值：只买概率>60%的股票
        ('sell_threshold', 0.50),   # 卖出概率阈值：概率<50%时止损
        ('position_ratio', 0.95),   # 仓位比例：使用95%资金，保留5%现金
        ('model', None),            # RF模型对象
        ('feature_cols', None),     # 特征列名列表
        ('stock_pool', None),       # 股票池字典 {代码: 名称}
    )

    def __init__(self):
        """初始化策略变量"""
        self.order = None           # 当前订单
        self.buy_price = None       # 买入价格
        self.buy_date = None        # 买入日期
        self.buy_size = None        # 买入数量
        self.current_stock = None   # 当前持仓股票代码
        self.trade_log = []         # 交易记录列表

    def log(self, txt, dt=None):
        """日志输出"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """
        订单状态通知（Backtrader回调函数）

        功能:
            1. 记录买入/卖出的执行价格、数量、手续费
            2. 计算每笔交易的盈亏和持仓天数
            3. 保存交易记录到trade_log
        """
        # 订单提交或接受状态，不做处理
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_date = self.datas[0].datetime.date(0)
                self.buy_size = order.executed.size  # 记录买入数量
                self.log(f'买入执行: {order.data._name}, 开盘价: {order.executed.price:.2f}, '
                        f'数量: {int(order.executed.size)}, 手续费: {order.executed.comm:.2f}')
            elif order.issell():
                # 使用买入时记录的数量
                sell_size = self.buy_size
                profit = (order.executed.price - self.buy_price) * sell_size - order.executed.comm
                profit_pct = (order.executed.price / self.buy_price - 1) * 100
                days_held = (self.datas[0].datetime.date(0) - self.buy_date).days

                self.log(f'卖出执行: {order.data._name}, 开盘价: {order.executed.price:.2f}, '
                        f'数量: {int(sell_size)}, 收益: {profit:.2f} ({profit_pct:+.2f}%), '
                        f'持有天数: {days_held}')

                # 记录交易
                self.trade_log.append({
                    'stock': order.data._name,
                    'buy_date': self.buy_date,
                    'sell_date': self.datas[0].datetime.date(0),
                    'buy_price': self.buy_price,
                    'sell_price': order.executed.price,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'days': days_held
                })

                self.current_stock = None
                self.buy_price = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单取消/拒绝: {order.data._name}')

        self.order = None

    def get_stock_probability(self, data, stock_code):
        """
        计算股票的上涨概率（核心预测函数）

        参数:
            data: Backtrader数据对象
            stock_code: 股票代码

        返回:
            float: 上涨概率 [0, 1]，越高表示越可能上涨

        流程:
            1. 读取股票历史数据
            2. 计算10个技术指标特征（8个历史+2个开盘）
            3. 找到当前日期对应的数据行
            4. 使用RF模型预测上涨概率

        注意：
            - 预测时不计算future_return（那是要预测的目标）
            - 只计算10个特征用于预测
            - 特征已对齐到开盘时可获得的数据，无时间穿越
        """
        try:
            # 获取该股票的历史数据
            name = self.p.stock_pool[stock_code]
            df = get_stock_data(stock_code, name, csv_dir='stock_data')
            if df is None:
                return 0.0

            # 计算特征（预测时不需要标签）
            df = calculate_features(df, for_training=False)
            # 只删除特征列中的NaN，不删除future_return的NaN（因为没计算）
            df = df.dropna(subset=self.p.feature_cols)
            if len(df) == 0:
                return 0.0

            # 找到当前日期对应的数据
            current_date = data.datetime.date(0)
            df_current = df[df['date'] == pd.Timestamp(current_date)]

            if len(df_current) == 0:
                # 使用最新数据
                df_current = df.iloc[-1:]

            # 预测
            X = df_current[self.p.feature_cols].values
            prob = self.p.model.predict_proba(X)[0][1]
            return prob

        except Exception as e:
            return 0.0

    def next_open(self):
        """
        每日开盘前执行函数（使用cheat_on_open模式）

        在每个交易日开盘前调用，可以看到当日开盘价并立即在开盘价成交
        执行逻辑:
            1. 使用截至昨日的历史特征（8个）
            2. 使用当日开盘价特征（2个：open_gap, open_vs_ma5）
            3. 模型预测，当日开盘立即执行交易

        注意：这是合理的，因为开盘价在开盘时就已知
        """
        # ===== 卖出逻辑 =====
        # 如果有持仓，检查是否需要卖出
        if self.current_stock is not None:
            data = self.getdatabyname(self.current_stock)
            prob = self.get_stock_probability(data, self.current_stock)

            # 卖出条件：概率低于阈值
            if prob < self.p.sell_threshold:
                position_size = self.getposition(data).size
                if position_size > 0:
                    self.log(f'卖出信号: {self.current_stock}, 概率: {prob:.2%} < {self.p.sell_threshold:.0%}')
                    # 明确指定卖出全部持仓
                    self.order = self.sell(data=data, size=position_size)
            else:
                # 持仓中，概率仍然>=50%，继续持有
                if self.buy_price:
                    current_price = data.open[0]
                    unrealized_pnl = (current_price - self.buy_price) / self.buy_price * 100
                    self.log(f'持仓中: {self.current_stock}, 概率: {prob:.2%}, 浮盈: {unrealized_pnl:+.2f}%')

        # ===== 买入逻辑 =====
        # 如果无持仓，寻找买入机会
        else:
            # 第一步：计算股票池中所有股票的上涨概率
            candidates = []  # 候选股票列表
            all_probs = []  # 记录所有概率用于调试

            for stock_code in self.p.stock_pool.keys():
                try:
                    data = self.getdatabyname(stock_code)
                    prob = self.get_stock_probability(data, stock_code)
                    all_probs.append((stock_code, prob))

                    if prob > self.p.buy_threshold:
                        candidates.append({
                            'code': stock_code,
                            'prob': prob,
                            'data': data
                        })
                except:
                    continue

            # 第二步：从候选股票中选择概率最高的那只
            if candidates:
                best = max(candidates, key=lambda x: x['prob'])  # 最优股票

                # 第三步：检查是否有其他持仓（调试用）
                total_positions_value = 0
                for d in self.datas:
                    pos = self.getposition(d)
                    if pos.size > 0:
                        pos_value = pos.size * d.open[0]
                        total_positions_value += pos_value
                        self.log(f'  [调试] 持仓: {d._name}, 数量: {pos.size}, 市值: {pos_value:.2f}')

                # 第四步：计算买入数量（按100股整数倍）
                # 使用账户总值而不是现金，避免cheat_on_open模式下资金未结算的问题
                cash = self.broker.getcash()
                value = self.broker.getvalue()         # 账户总值
                price = best['data'].open[0]           # 当前开盘价
                size = int((value * self.p.position_ratio) / price / 100) * 100  # 向下取整到100股

                # 第五步：执行买入（至少100股才买）
                if size >= 100:
                    self.log(f'买入信号: {best["code"]}, 概率: {best["prob"]:.2%}, '
                            f'开盘价: {price:.2f}, 数量: {size} (总值: {value:.2f})')
                    # 使用市价单，当日开盘成交
                    self.order = self.buy(data=best['data'], size=size)
                    self.current_stock = best['code']
            else:
                # 没有符合条件的股票，输出当天最高概率
                if all_probs:
                    max_stock, max_prob = max(all_probs, key=lambda x: x[1])
                    self.log(f'无买入机会，最高概率: {max_stock} {max_prob:.2%} < {self.p.buy_threshold:.0%}')
                else:
                    self.log(f'警告: 无法获取任何股票数据')


class PandasData(bt.feeds.PandasData):
    """
    自定义Pandas数据源（适配Backtrader）

    将Pandas DataFrame转换为Backtrader可识别的数据格式

    列映射:
        datetime -> index: 日期索引
        open -> open: 开盘价
        high -> high: 最高价
        low -> low: 最低价
        close -> close: 收盘价
        volume -> volume: 成交量
        openinterest -> None: 不使用持仓量
    """
    params = (
        ('datetime', None),        # 使用索引作为日期
        ('open', 'open'),          # 开盘价列名
        ('high', 'high'),          # 最高价列名
        ('low', 'low'),            # 最低价列名
        ('close', 'close'),        # 收盘价列名
        ('volume', 'volume'),      # 成交量列名
        ('openinterest', None),    # 不使用持仓量
    )


def run_backtest(model, feature_cols, stock_pool, backtest_start, backtest_end, initial_cash=100000):
    """
    运行回测（使用Backtrader框架）

    参数:
        model: 训练好的RF模型
        feature_cols: 特征列名列表
        stock_pool: 股票池字典
        backtest_start: 回测开始日期，如 '2025-07-01'
        backtest_end: 回测结束日期，如 '2025-11-21'
        initial_cash: 初始资金，默认100,000元

    返回:
        final_value: 最终资金
        total_return: 总收益率(%)

    流程:
        1. 创建Cerebro引擎
        2. 添加策略和参数
        3. 加载股票数据
        4. 设置初始资金和手续费
        5. 运行回测
        6. 统计交易结果
        7. 绘制收益曲线
    """
    print(f"\n{'='*80}")
    print(f"开始回测 ({backtest_start} 至 {backtest_end})")
    print(f"{'='*80}")

    # 第一步：创建Backtrader回测引擎
    # cheat_on_open=True: 允许在开盘时看到开盘价并立即成交
    # 这是合理的，因为我们的特征使用的是：截至昨日的历史数据 + 当日开盘价
    cerebro = bt.Cerebro(cheat_on_open=True)

    # 第二步：添加策略及参数
    cerebro.addstrategy(
        RFBestStockStrategy,
        buy_threshold=0.60,
        sell_threshold=0.50,
        position_ratio=0.95,
        model=model,
        feature_cols=feature_cols,
        stock_pool=stock_pool
    )

    # 第三步：加载股票数据到Cerebro
    data_count = 0
    for code, name in stock_pool.items():
        df = get_stock_data(code, name, csv_dir='stock_data')
        if df is None:
            continue

        # 过滤回测时间段（向前扩展以计算指标）
        df_backtest = df[df['date'] >= pd.Timestamp(backtest_start) - pd.Timedelta(days=60)]
        if len(df_backtest) < 50:
            continue

        # 设置日期为索引
        df_backtest = df_backtest.set_index('date')
        df_backtest = df_backtest[['open', 'high', 'low', 'close', 'volume']]

        # 添加到cerebro
        data = PandasData(
            dataname=df_backtest,
            fromdate=datetime.strptime(backtest_start, '%Y-%m-%d'),
            todate=datetime.strptime(backtest_end, '%Y-%m-%d')
        )
        cerebro.adddata(data, name=code)
        data_count += 1

    print(f"已加载 {data_count} 只股票数据")

    # 第四步：设置初始资金
    cerebro.broker.setcash(initial_cash)

    # 第五步：设置手续费（0.1%）
    cerebro.broker.setcommission(commission=0.001)

    # 第六步：运行回测
    print(f"\n初始资金: {initial_cash:,.2f}")
    strategies = cerebro.run()
    final_value = cerebro.broker.getvalue()

    # 第七步：计算收益指标
    total_return = (final_value - initial_cash) / initial_cash * 100

    print(f"\n{'='*80}")
    print(f"回测结果")
    print(f"{'='*80}")
    print(f"最终资金: {final_value:,.2f}")
    print(f"总收益: {final_value - initial_cash:+,.2f}")
    print(f"收益率: {total_return:+.2f}%")

    # 第八步：检查是否有未平仓持仓
    strategy = strategies[0]
    if strategy.current_stock:
        print(f"\n警告: 回测结束时仍持有股票 {strategy.current_stock}")
        print(f"  买入价格: {strategy.buy_price:.2f}")
        print(f"  买入日期: {strategy.buy_date}")
        print(f"  未计入交易记录！")

    # 第九步：统计交易明细
    if strategy.trade_log:
        trades_df = pd.DataFrame(strategy.trade_log)
        print(f"\n交易统计:")
        print(f"  交易次数: {len(trades_df)}")
        print(f"  盈利次数: {(trades_df['profit'] > 0).sum()}")
        print(f"  亏损次数: {(trades_df['profit'] < 0).sum()}")
        print(f"  胜率: {(trades_df['profit'] > 0).mean():.2%}")
        print(f"  平均收益率: {trades_df['profit_pct'].mean():+.2f}%")
        print(f"  平均持仓天数: {trades_df['days'].mean():.1f}天")

        print(f"\n前10笔交易明细:")
        print(trades_df.head(10).to_string(index=False))
    else:
        print(f"\n无交易记录")

    # 第九步：绘制自定义可视化图表
    plot_backtest_results(strategy, initial_cash, final_value, backtest_start, backtest_end)

    return final_value, total_return


def get_index_data(index_code='sh.000001', csv_dir='stock_data'):
    """
    获取上证指数数据

    参数:
        index_code: 指数代码，默认'sh.000001'（上证指数）
        csv_dir: CSV文件目录

    返回:
        DataFrame: 包含日期和收盘价的指数数据
        None: 如果文件不存在
    """
    try:
        # 尝试多种可能的文件名
        possible_files = [
            f"{index_code}_上证指数.csv",
            f"sh.000001_上证指数.csv",
            f"000001_上证指数.csv",
            f"上证指数.csv"
        ]

        df = None
        for filename in possible_files:
            filepath = os.path.join(csv_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                break

        if df is None:
            print(f"  警告: 未找到上证指数数据文件")
            return None

        # 标准化列名
        df.columns = [col.lower() for col in df.columns]

        # 处理日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
        else:
            return None

        # 确保有收盘价
        if 'close' not in df.columns:
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna()

        return df

    except Exception as e:
        print(f"  警告: 读取上证指数数据失败: {e}")
        return None


def plot_backtest_results(strategy, initial_cash, final_value, start_date, end_date):
    """
    绘制回测结果可视化图表

    包含两个子图：
    1. 资金曲线与上证指数对比（双Y轴）
    2. 交易标记（在时间轴上标注买卖点）
    """
    if not strategy.trade_log:
        print("\n无交易记录，跳过可视化")
        return

    # 准备交易数据
    trades_df = pd.DataFrame(strategy.trade_log)
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])

    # 构建每日资金曲线
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    portfolio_values = []

    current_value = initial_cash
    for date in date_range:
        # 查找该日期之前完成的所有交易
        completed_trades = trades_df[trades_df['sell_date'] <= date]
        if len(completed_trades) > 0:
            total_profit = completed_trades['profit'].sum()
            current_value = initial_cash + total_profit
        else:
            current_value = initial_cash

        portfolio_values.append(current_value)

    # 读取上证指数数据
    index_df = get_index_data(csv_dir='stock_data')

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('回测结果可视化', fontsize=16, fontweight='bold')

    # === 子图1：资金曲线 vs 上证指数（双Y轴） ===
    ax1 = axes[0]

    # 左轴：策略资金曲线
    color1 = '#2E86AB'
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('策略资金 (元)', fontsize=12, color=color1)
    line1 = ax1.plot(date_range, portfolio_values, linewidth=2.5, color=color1,
                     label='策略资金曲线', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(date_range[0], date_range[-1])

    # 右轴：上证指数
    if index_df is not None:
        ax1_right = ax1.twinx()
        color2 = '#E63946'
        ax1_right.set_ylabel('上证指数', fontsize=12, color=color2)

        # 过滤指数数据到回测时间范围
        index_filtered = index_df[
            (index_df['date'] >= pd.Timestamp(start_date)) &
            (index_df['date'] <= pd.Timestamp(end_date))
        ].copy()

        if len(index_filtered) > 0:
            line2 = ax1_right.plot(index_filtered['date'], index_filtered['close'],
                                  linewidth=2, color=color2, alpha=0.7,
                                  label='上证指数', linestyle='--', zorder=2)
            ax1_right.tick_params(axis='y', labelcolor=color2)

            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=11)
        else:
            ax1.legend(loc='upper left', fontsize=11)
    else:
        ax1.legend(loc='upper left', fontsize=11)

    ax1.set_title('策略资金曲线 vs 上证指数', fontsize=13, pad=10, fontweight='bold')

    # === 子图2：交易标记 ===
    ax2 = axes[1]

    # 绘制买入点和卖出点
    for idx, trade in trades_df.iterrows():
        # 买入标记（绿色向上三角）
        ax2.scatter(trade['buy_date'], 1, color='green', marker='^', s=200,
                   edgecolors='darkgreen', linewidths=2, zorder=3,
                   label='买入' if idx == 0 else '')
        # 卖出标记（红色向下三角）
        ax2.scatter(trade['sell_date'], 1, color='red', marker='v', s=200,
                   edgecolors='darkred', linewidths=2, zorder=3,
                   label='卖出' if idx == 0 else '')
        # 连线显示持仓期
        ax2.plot([trade['buy_date'], trade['sell_date']], [1, 1],
                color='blue', linewidth=3, alpha=0.3, zorder=1)

        # 标注股票代码和收益
        mid_date = trade['buy_date'] + (trade['sell_date'] - trade['buy_date']) / 2
        profit_text = f"{trade['stock']}\n{trade['profit_pct']:+.1f}%"
        color = 'green' if trade['profit'] > 0 else 'red'
        ax2.text(mid_date, 1.15, profit_text, ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold')

    ax2.set_ylim(0.5, 1.5)
    ax2.set_yticks([])
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('交易时间轴', fontsize=13, pad=10)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(date_range[0], date_range[-1])

    # 统一格式
    import matplotlib.dates as mdates
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=10)
        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # 保存图表
    output_file = 'backtest_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化图表已保存: {output_file}")
    plt.close()  # 关闭图表，不显示窗口


# ==================== 主程序 ====================

def main():
    """
    主程序入口

    完整流程:
        1. 定义股票池和时间划分
        2. 训练随机森林模型（或加载已有模型）
        3. 运行回测验证策略
        4. 输出收益和交易统计
    """
    print("="*80)
    print("基于随机森林的最优选股回测策略")
    print("="*80)

    # ===== 第一步：定义股票池 =====
    # 使用沪深300成分股（300只A股核心股票）
    # 沪深300：A股市场最具代表性的大中盘指数
    # 特点：
    #   - 300只股票，样本充足，训练数据更丰富
    #   - 覆盖沪深两市，行业分布均衡
    #   - 市值覆盖面广：既有千亿市值龙头，也有百亿成长股
    #   - 流动性好，数据质量高
    stock_pool = get_hs300_stock_pool()

    print(f"股票池: 沪深300成分股")
    print(f"股票数量: {len(stock_pool)}只")
    print(f"覆盖行业: 金融(60)、消费(50)、科技(50)、医药(40)、新能源(30)、周期(40)、能源(20)、工业(30)、地产(10)、传媒(10)")

    # ===== 第二步：定义时间划分（严格时间序列） =====
    # 训练期：用于训练模型，不能包含回测期数据
    TRAIN_START = '2022-01-01'   # 训练开始日期
    TRAIN_END = '2025-06-30'     # 训练结束日期

    # 回测期：模拟真实交易，验证策略收益
    BACKTEST_START = '2025-07-01'  # 回测开始日期（必须晚于训练期）
    BACKTEST_END = '2025-11-21'    # 回测结束日期

    # ===== 第三步：训练模型 =====
    print("\n开始训练模型...")

    # 3.1 准备训练数据
    result = prepare_training_data(stock_pool, TRAIN_START, TRAIN_END)
    if result is None:
        print("\n错误: 无法加载训练数据")
        print("请先运行 download_stock_data_baostock.py 下载数据")
        return

    combined, feature_cols = result

    # 3.2 训练随机森林模型
    model, feature_cols = train_rf_model(combined, feature_cols)

    # 3.3 保存模型供每日预测使用
    print(f"\n{'='*80}")
    print("保存模型...")
    print(f"{'='*80}")

    import pickle
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'stock_pool': stock_pool,
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'train_start': TRAIN_START,
        'train_end': TRAIN_END
    }

    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("✓ 模型已保存到: rf_model.pkl")
    print(f"  - 股票池: {len(stock_pool)} 只")
    print(f"  - 特征数: {len(feature_cols)} 个")
    print(f"  - 训练时间: {model_data['train_date']}")

    # ===== 第四步：运行回测 =====
    final_value, total_return = run_backtest(
        model=model,
        feature_cols=feature_cols,
        stock_pool=stock_pool,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        initial_cash=100000
    )

    print(f"\n{'='*80}")
    print("回测完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
