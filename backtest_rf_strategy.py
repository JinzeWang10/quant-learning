"""
基于随机森林模型的量化交易回测系统
策略: 使用机器学习预测上涨概率，概率>60%买入，<40%卖出
数据源: 本地CSV文件
"""

import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import os
import pickle

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RFPredictor:
    """随机森林预测器 - 封装模型训练和预测"""

    def __init__(self):
        self.model = None
        self.feature_cols = None

    def calculate_features(self, df):
        """计算技术指标特征"""
        # 确保有足够数据
        if len(df) < 30:
            return None

        # 价格特征
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)

        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma_ratio_5_20'] = df['ma5'] / df['ma20']

        # 成交量
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 波动率
        df['volatility'] = df['return_1d'].rolling(20).std()

        # 布林带位置
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def train_model(self, stock_pool, csv_dir='stock_data',
                    train_start='2023-01-01', train_end='2024-06-30',
                    test_start='2024-07-01', test_end='2024-10-18'):
        """
        训练随机森林模型（按时间顺序划分训练集和测试集）

        参数:
            stock_pool: 股票池 {code: name}
            csv_dir: CSV目录
            train_start: 训练集开始日期
            train_end: 训练集结束日期
            test_start: 测试集开始日期
            test_end: 测试集结束日期
        """
        print("\n" + "="*70)
        print("训练随机森林模型（时间序列划分）")
        print("="*70)
        print(f"训练期: {train_start} ~ {train_end}")
        print(f"测试期: {test_start} ~ {test_end}")
        print("="*70)

        train_data = []
        test_data = []

        # 转换日期
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)

        # 加载所有股票数据
        print("\n正在加载数据...")
        for code, name in stock_pool.items():
            csv_file = os.path.join(csv_dir, f"{code}_{name}.csv")
            if not os.path.exists(csv_file):
                continue

            try:
                df = pd.read_csv(csv_file)
                df.columns = [col.lower() for col in df.columns]

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    continue

                # 计算特征
                df = self.calculate_features(df)
                if df is None:
                    continue

                # 计算标签：未来5日涨幅>3%为正样本
                df['future_return'] = df['close'].shift(-5) / df['close'] - 1
                df['label'] = (df['future_return'] > 0.03).astype(int)

                df['stock_code'] = code

                # 按时间划分训练集和测试集
                train_df = df[(df['date'] >= train_start_dt) & (df['date'] <= train_end_dt)].copy()
                test_df = df[(df['date'] >= test_start_dt) & (df['date'] <= test_end_dt)].copy()

                if len(train_df) > 0:
                    train_data.append(train_df)
                if len(test_df) > 0:
                    test_data.append(test_df)

                print(f"  ✓ {name}({code}): 训练{len(train_df)}条, 测试{len(test_df)}条")

            except Exception as e:
                print(f"  ✗ {name}({code}): {str(e)[:50]}")

        if not train_data:
            print("错误: 没有加载到训练数据")
            return False

        # 合并训练集
        train_combined = pd.concat(train_data, ignore_index=True)
        train_combined = train_combined.dropna()

        # 特征列
        self.feature_cols = ['return_1d', 'return_5d', 'return_10d', 'ma_ratio_5_20',
                             'volume_ratio', 'rsi', 'volatility', 'bb_position']

        X_train = train_combined[self.feature_cols]
        y_train = train_combined['label']

        print(f"\n训练集大小: {len(X_train)} 条记录")
        print(f"训练集正样本比例: {y_train.mean():.2%}")

        # 如果有测试集，也处理
        if test_data:
            test_combined = pd.concat(test_data, ignore_index=True)
            test_combined = test_combined.dropna()
            X_test = test_combined[self.feature_cols]
            y_test = test_combined['label']
            print(f"测试集大小: {len(X_test)} 条记录")
            print(f"测试集正样本比例: {y_test.mean():.2%}")
        else:
            X_test = None
            y_test = None
            print("警告: 没有测试集数据")

        # 训练模型
        print("\n训练随机森林...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # 评估
        train_score = self.model.score(X_train, y_train)
        print(f"训练集准确率: {train_score:.2%}")

        if X_test is not None:
            test_score = self.model.score(X_test, y_test)
            print(f"测试集准确率: {test_score:.2%}")
        else:
            print("测试集准确率: N/A")

        # 特征重要性
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n特征重要性:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']:<20} {row['importance']:.4f}")

        print("\n" + "="*70)
        print("模型训练完成！")
        print("="*70)

        return True

    def predict_probability(self, df):
        """
        预测上涨概率

        参数:
            df: 包含历史数据的DataFrame (至少30条)

        返回:
            float: 上涨概率 (0-1)，失败返回None
        """
        if self.model is None or self.feature_cols is None:
            return None

        try:
            # 计算特征
            df_features = self.calculate_features(df.copy())
            if df_features is None:
                return None

            df_features = df_features.dropna()
            if len(df_features) == 0:
                return None

            # 使用最新数据预测
            latest = df_features[self.feature_cols].iloc[-1:].values
            prob = self.model.predict_proba(latest)[0][1]  # 上涨概率

            return prob

        except Exception:
            return None

    def save_model(self, filepath='rf_model.pkl'):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.feature_cols}, f)
        print(f"模型已保存: {filepath}")

    def load_model(self, filepath='rf_model.pkl'):
        """加载模型"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_cols = data['features']
            print(f"模型已加载: {filepath}")
            return True
        return False


class RFStrategy(bt.Strategy):
    """
    基于随机森林的交易策略

    规则:
    - 上涨概率 > 60% → 买入
    - 上涨概率 < 40% → 卖出
    - 单只股票最大仓位 35%
    """

    params = (
        ('buy_threshold', 0.60),      # 买入概率阈值
        ('sell_threshold', 0.40),     # 卖出概率阈值
        ('max_positions', 3),         # 最大持仓数
        ('position_ratio', 0.35),     # 单只仓位比例
        ('predictor', None),          # RF预测器
    )

    def __init__(self):
        self.predictions = {}  # 存储每只股票的预测概率
        self.buy_signals = []
        self.sell_signals = []
        self.portfolio_values = []
        self.dates = []

    def next(self):
        """策略主逻辑"""
        current_date = self.datas[0].datetime.date(0)
        self.dates.append(current_date)
        self.portfolio_values.append(self.broker.getvalue())

        # 调试：每10个交易日打印一次
        if len(self.dates) % 10 == 1:
            print(f"  回测进度: {current_date}, 已运行{len(self.dates)}天")

        # === 卖出检查 ===
        for d in self.datas:
            position = self.getposition(d)
            if position.size > 0:
                # 预测概率
                prob = self._predict_stock(d)

                if prob is not None and prob < self.params.sell_threshold:
                    self.close(data=d)
                    self.sell_signals.append({
                        'date': current_date,
                        'stock_name': d._name,
                        'price': d.close[0],
                        'size': position.size,
                        'probability': prob
                    })

        # === 买入检查 ===
        buy_candidates = []

        for d in self.datas:
            position = self.getposition(d)
            if position.size > 0:
                continue

            # 预测概率
            prob = self._predict_stock(d)

            # 调试：打印预测结果（仅前5天）
            if len(self.dates) <= 5 and prob is not None:
                print(f"    {d._name}: 概率={prob:.2%}, 阈值={self.params.buy_threshold:.0%}")

            if prob is not None and prob >= self.params.buy_threshold:
                buy_candidates.append({
                    'data': d,
                    'name': d._name,
                    'probability': prob,
                    'price': d.close[0]
                })
                print(f"  [买入候选] {d._name}: 概率={prob:.2%}")

        # 按概率排序
        if buy_candidates:
            buy_candidates.sort(key=lambda x: x['probability'], reverse=True)

            current_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)

            for candidate in buy_candidates:
                if current_positions >= self.params.max_positions:
                    break

                size = self._calculate_position_size(candidate['data'])
                if size >= 100:
                    self.buy(data=candidate['data'], size=size)
                    self.buy_signals.append({
                        'date': current_date,
                        'stock_name': candidate['name'],
                        'price': candidate['price'],
                        'size': size,
                        'probability': candidate['probability']
                    })
                    current_positions += 1

    def _predict_stock(self, data):
        """
        预测单只股票的上涨概率

        参数:
            data: backtrader数据源对象
        """
        if self.params.predictor is None:
            return None

        try:
            # 获取历史数据（使用backtrader的buflen获取可用数据长度）
            lookback = min(100, len(data))  # 最多取100条历史数据

            if lookback < 30:
                return None

            # 构建DataFrame
            hist_data = []
            for i in range(-lookback + 1, 1):  # 从-99到0
                hist_data.append({
                    'date': data.datetime.date(i),
                    'open': data.open[i],
                    'high': data.high[i],
                    'low': data.low[i],
                    'close': data.close[i],
                    'volume': data.volume[i]
                })

            df = pd.DataFrame(hist_data)

            # 预测
            prob = self.params.predictor.predict_probability(df)
            return prob

        except Exception:
            # 预测失败，返回None
            return None

    def _calculate_position_size(self, data):
        """计算买入数量"""
        current_price = data.close[0]
        total_value = self.broker.getvalue()
        max_investment = total_value * self.params.position_ratio
        available_cash = self.broker.getcash()
        investment = min(available_cash, max_investment)
        shares = int(investment / current_price / 100) * 100
        return max(shares, 100) if shares >= 100 else 0


def load_stock_data(stock_pool, csv_dir='stock_data',
                    backtest_start=None, backtest_end=None):
    """
    加载股票数据（可指定回测时间范围）

    参数:
        stock_pool: 股票池
        csv_dir: CSV目录
        backtest_start: 回测开始日期（可选）
        backtest_end: 回测结束日期（可选）
    """
    stock_data = {}

    print("\n正在加载回测数据...")
    if backtest_start and backtest_end:
        print(f"回测期: {backtest_start} ~ {backtest_end}")

    for code, name in stock_pool.items():
        csv_file = os.path.join(csv_dir, f"{code}_{name}.csv")
        if not os.path.exists(csv_file):
            print(f"  ✗ {name}({code}): 文件不存在")
            continue

        try:
            df = pd.read_csv(csv_file)
            df.columns = [col.lower() for col in df.columns]

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                continue

            # 如果指定了回测时间范围，需要包含之前的数据用于特征计算
            if backtest_start and backtest_end:
                # 向前扩展60天，确保有足够的历史数据计算MA60等指标
                backtest_start_dt = pd.to_datetime(backtest_start)
                extended_start = backtest_start_dt - pd.Timedelta(days=100)  # 多取100天历史数据

                df = df[(df['date'] >= extended_start) &
                       (df['date'] <= pd.to_datetime(backtest_end))].copy()

            df.set_index('date', inplace=True)

            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                continue

            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=required, inplace=True)

            if len(df) >= 30:  # 降低要求，回测期数据可能较短
                stock_data[name] = (code, df)
                print(f"  ✓ {name}({code}): {len(df)}条数据")
            else:
                print(f"  ✗ {name}({code}): 数据不足({len(df)}条)")

        except Exception as e:
            print(f"  ✗ {name}({code}): {str(e)[:50]}")

    return stock_data


def run_backtest(stock_pool, predictor,
                 initial_cash=100000,
                 buy_threshold=0.60,
                 sell_threshold=0.40,
                 max_positions=3,
                 position_ratio=0.35,
                 commission=0.001,
                 backtest_start='2024-07-01',
                 backtest_end='2024-10-18'):
    """
    运行回测

    参数:
        backtest_start: 回测开始日期（应与测试集一致）
        backtest_end: 回测结束日期（应与测试集一致）
    """

    print("\n" + "="*70)
    print("随机森林策略回测")
    print("="*70)
    print(f"回测期: {backtest_start} ~ {backtest_end}")
    print(f"买入阈值: 概率 ≥ {buy_threshold:.0%}")
    print(f"卖出阈值: 概率 < {sell_threshold:.0%}")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"最大持仓: {max_positions}只")
    print(f"单只仓位: {position_ratio*100:.0f}%")
    print("="*70)

    # 加载数据（仅加载回测期数据）
    stock_data = load_stock_data(stock_pool, csv_dir='stock_data',
                                 backtest_start=backtest_start,
                                 backtest_end=backtest_end)

    if not stock_data:
        print("\n错误: 没有加载到任何数据")
        return None, None

    print(f"\n成功加载 {len(stock_data)} 只股票")

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(
        RFStrategy,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        position_ratio=position_ratio,
        predictor=predictor
    )

    # 添加数据
    for name, (code, df) in stock_data.items():
        data_feed = bt.feeds.PandasData(dataname=df, name=name)
        cerebro.adddata(data_feed)

    # 设置参数
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print("\n开始回测...\n")
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    print("\n" + "="*70)
    print("回测完成！")
    print("="*70)
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"最终资金: ¥{final_value:,.2f}")
    print(f"总收益: ¥{final_value - initial_cash:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print("="*70)

    return cerebro, strat


def print_trade_statistics(strat):
    """打印交易统计"""
    print("\n" + "="*70)
    print("交易统计")
    print("="*70)
    print(f"买入次数: {len(strat.buy_signals)}")
    print(f"卖出次数: {len(strat.sell_signals)}")

    if strat.buy_signals:
        avg_buy_prob = np.mean([s['probability'] for s in strat.buy_signals])
        print(f"平均买入概率: {avg_buy_prob:.2%}")

        print(f"\n前10笔买入信号:")
        for i, s in enumerate(strat.buy_signals[:10], 1):
            print(f"  {i}. {s['date']} {s['stock_name']}: "
                  f"¥{s['price']:.2f}, {s['size']}股, 概率{s['probability']:.1%}")

    if strat.sell_signals:
        avg_sell_prob = np.mean([s['probability'] for s in strat.sell_signals])
        print(f"\n平均卖出概率: {avg_sell_prob:.2%}")

        print(f"\n前10笔卖出信号:")
        for i, s in enumerate(strat.sell_signals[:10], 1):
            print(f"  {i}. {s['date']} {s['stock_name']}: "
                  f"¥{s['price']:.2f}, {s['size']}股, 概率{s['probability']:.1%}")

    print("="*70)


def plot_results(strat):
    """绘制回测结果"""
    if not strat.portfolio_values:
        print("没有数据可绘制")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # === 子图1: 账户价值曲线 ===
    dates = strat.dates
    values = strat.portfolio_values

    ax1.plot(dates, values, label='账户总值', linewidth=2, color='navy')
    ax1.axhline(y=strat.broker.startingcash, color='red', linestyle='--',
                alpha=0.7, label=f'初始资金 ¥{strat.broker.startingcash:,.0f}')

    # 标注买入点
    for signal in strat.buy_signals:
        idx = dates.index(signal['date']) if signal['date'] in dates else None
        if idx is not None:
            ax1.scatter(signal['date'], values[idx], color='red', marker='^',
                       s=100, alpha=0.7, zorder=5)

    # 标注卖出点
    for signal in strat.sell_signals:
        idx = dates.index(signal['date']) if signal['date'] in dates else None
        if idx is not None:
            ax1.scatter(signal['date'], values[idx], color='green', marker='v',
                       s=100, alpha=0.7, zorder=5)

    ax1.set_ylabel('账户价值 (元)', fontsize=11)
    ax1.set_title('随机森林策略 - 账户价值变化', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # === 子图2: 收益率曲线 ===
    returns = [(v - strat.broker.startingcash) / strat.broker.startingcash * 100
               for v in values]
    ax2.plot(dates, returns, linewidth=2, color='darkgreen')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(dates, returns, 0, alpha=0.3, color='green', where=[r >= 0 for r in returns])
    ax2.fill_between(dates, returns, 0, alpha=0.3, color='red', where=[r < 0 for r in returns])

    ax2.set_xlabel('日期', fontsize=11)
    ax2.set_ylabel('收益率 (%)', fontsize=11)
    ax2.set_title('累计收益率', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'rf_strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {filename}")
    plt.show()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("基于随机森林的量化交易回测系统")
    print("="*70)
    print("数据源: stock_data 目录")
    print("策略: 概率>60%买入，<40%卖出")
    print("="*70)

    # 定义时间划分（防止数据泄露）
    TRAIN_START = '2023-01-01'
    TRAIN_END = '2024-06-30'
    TEST_START = '2024-07-01'
    TEST_END = '2024-10-18'

    print(f"\n时间划分:")
    print(f"  训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  测试期: {TEST_START} ~ {TEST_END}")
    print(f"  回测期: {TEST_START} ~ {TEST_END} (与测试期一致)")
    print("="*70)

    # 股票池（精简版，用于快速测试）
    stock_pool = {
        '000001': '平安银行',
        '600036': '招商银行',
        '600519': '贵州茅台',
        '000858': '五粮液',
        '601318': '中国平安',
        '000333': '美的集团',
        '000651': '格力电器',
        '600887': '伊利股份',
        '600030': '中信证券',
        '002142': '宁波银行',
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


    # 1. 创建并训练RF预测器
    predictor = RFPredictor()

    # 训练模型（使用训练集和测试集）
    print("\n训练新模型（使用时间序列划分）...")
    if not predictor.train_model(
        stock_pool,
        csv_dir='stock_data',
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END
    ):
        print("模型训练失败")
        return

    # 保存模型
    predictor.save_model('rf_model_ts.pkl')

    # 2. 运行回测（仅在测试期回测）
    cerebro, strat = run_backtest(
        stock_pool=stock_pool,
        predictor=predictor,
        initial_cash=100000,
        buy_threshold=0.60,
        sell_threshold=0.40,
        max_positions=3,
        position_ratio=0.35,
        commission=0.001,
        backtest_start=TEST_START,  # 回测期 = 测试期
        backtest_end=TEST_END
    )

    if cerebro is None or strat is None:
        print("回测失败")
        return

    # 3. 统计和可视化
    print_trade_statistics(strat)
    plot_results(strat)

    print("\n回测完成！")


if __name__ == "__main__":
    main()
