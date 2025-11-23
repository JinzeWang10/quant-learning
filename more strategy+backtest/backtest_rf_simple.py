"""
基于随机森林的量化交易回测系统 (简化版 - 能产生交易)
注意: 此版本使用全部数据训练，存在数据泄露，仅用于验证策略逻辑
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
    """随机森林预测器"""

    def __init__(self):
        self.model = None
        self.feature_cols = None

    def calculate_features(self, df):
        """计算技术指标特征"""
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

    def train_model(self, stock_pool, csv_dir='stock_data'):
        """训练模型（使用全部数据）"""
        print("\n" + "="*70)
        print("训练随机森林模型 (使用全部数据)")
        print("="*70)

        all_data = []

        print("正在加载数据...")
        for code, name in stock_pool.items():
            csv_file = os.path.join(csv_dir, f"{code}_{name}.csv")
            if not os.path.exists(csv_file):
                continue

            try:
                df = pd.read_csv(csv_file)
                df.columns = [col.lower() for col in df.columns]
                df['date'] = pd.to_datetime(df['date'])

                # 计算特征
                df = self.calculate_features(df)
                if df is None:
                    continue

                # 标签：未来5日涨幅>3%
                df['future_return'] = df['close'].shift(-5) / df['close'] - 1
                df['label'] = (df['future_return'] > 0.03).astype(int)

                all_data.append(df)
                print(f"  ✓ {name}({code}): {len(df)}条")

            except Exception as e:
                print(f"  ✗ {name}({code}): {str(e)[:50]}")

        if not all_data:
            return False

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.dropna()

        self.feature_cols = ['return_1d', 'return_5d', 'return_10d', 'ma_ratio_5_20',
                             'volume_ratio', 'rsi', 'volatility', 'bb_position']

        X = combined[self.feature_cols]
        y = combined['label']

        print(f"\n训练集大小: {len(X)} 条")
        print(f"正样本比例: {y.mean():.2%}")

        self.model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=42, n_jobs=-1)
        self.model.fit(X, y)

        print(f"训练准确率: {self.model.score(X, y):.2%}")

        # 特征重要性
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n特征重要性:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']:<20} {row['importance']:.4f}")

        print("="*70)
        return True

    def predict_probability(self, df):
        """预测上涨概率"""
        if self.model is None:
            return None

        try:
            df_features = self.calculate_features(df.copy())
            if df_features is None:
                return None

            df_features = df_features.dropna()
            if len(df_features) == 0:
                return None

            latest = df_features[self.feature_cols].iloc[-1:].values
            prob = self.model.predict_proba(latest)[0][1]
            return prob

        except Exception:
            return None


class RFStrategy(bt.Strategy):
    """随机森林交易策略"""

    params = (
        ('buy_threshold', 0.60),
        ('sell_threshold', 0.40),
        ('max_positions', 3),
        ('position_ratio', 0.35),
        ('predictor', None),
    )

    def __init__(self):
        self.buy_signals = []
        self.sell_signals = []
        self.portfolio_values = []
        self.dates = []

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.dates.append(current_date)
        self.portfolio_values.append(self.broker.getvalue())

        # 卖出检查
        for d in self.datas:
            position = self.getposition(d)
            if position.size > 0:
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
                    print(f"  [卖出] {d._name}: {prob:.1%}")

        # 买入检查
        buy_candidates = []
        for d in self.datas:
            if self.getposition(d).size > 0:
                continue

            prob = self._predict_stock(d)
            if prob is not None and prob >= self.params.buy_threshold:
                buy_candidates.append({
                    'data': d,
                    'name': d._name,
                    'probability': prob,
                    'price': d.close[0]
                })

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
                    print(f"  [买入] {candidate['name']}: {candidate['probability']:.1%}")
                    current_positions += 1

    def _predict_stock(self, data):
        """预测股票"""
        if self.params.predictor is None:
            return None

        try:
            lookback = min(100, len(data))
            if lookback < 30:
                return None

            hist_data = []
            for i in range(-lookback + 1, 1):
                hist_data.append({
                    'date': data.datetime.date(i),
                    'open': data.open[i],
                    'high': data.high[i],
                    'low': data.low[i],
                    'close': data.close[i],
                    'volume': data.volume[i]
                })

            df = pd.DataFrame(hist_data)
            prob = self.params.predictor.predict_probability(df)
            return prob

        except Exception:
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


def load_stock_data(stock_pool, csv_dir='stock_data'):
    """加载股票数据"""
    stock_data = {}

    print("\n正在加载回测数据...")
    for code, name in stock_pool.items():
        csv_file = os.path.join(csv_dir, f"{code}_{name}.csv")
        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
            df.columns = [col.lower() for col in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            required = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required):
                for col in required:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df.dropna(subset=required, inplace=True)

                if len(df) >= 70:
                    stock_data[name] = (code, df)
                    print(f"  ✓ {name}({code}): {len(df)}条")

        except Exception as e:
            print(f"  ✗ {name}({code}): {str(e)[:50]}")

    return stock_data


def run_backtest(stock_pool, predictor,
                 initial_cash=100000,
                 buy_threshold=0.60,
                 sell_threshold=0.40):
    """运行回测"""

    print("\n" + "="*70)
    print("随机森林策略回测")
    print("="*70)

    stock_data = load_stock_data(stock_pool, csv_dir='stock_data')
    if not stock_data:
        return None, None

    print(f"\n成功加载 {len(stock_data)} 只股票")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(RFStrategy,
                       buy_threshold=buy_threshold,
                       sell_threshold=sell_threshold,
                       predictor=predictor)

    for name, (code, df) in stock_data.items():
        data_feed = bt.feeds.PandasData(dataname=df, name=name)
        cerebro.adddata(data_feed)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    print("\n开始回测...\n")
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    print("\n" + "="*70)
    print("回测完成！")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"最终资金: ¥{final_value:,.2f}")
    print(f"总收益: ¥{final_value - initial_cash:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"买入次数: {len(strat.buy_signals)}")
    print(f"卖出次数: {len(strat.sell_signals)}")
    print("="*70)

    return cerebro, strat


def plot_results(strat):
    """绘制结果"""
    if not strat.portfolio_values:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    dates = strat.dates
    values = strat.portfolio_values

    ax.plot(dates, values, label='账户总值', linewidth=2, color='navy')
    ax.axhline(y=strat.broker.startingcash, color='red', linestyle='--',
               alpha=0.7, label=f'初始资金 ¥{strat.broker.startingcash:,.0f}')

    for signal in strat.buy_signals:
        idx = dates.index(signal['date']) if signal['date'] in dates else None
        if idx:
            ax.scatter(signal['date'], values[idx], color='red', marker='^',
                      s=100, alpha=0.7, zorder=5)

    for signal in strat.sell_signals:
        idx = dates.index(signal['date']) if signal['date'] in dates else None
        if idx:
            ax.scatter(signal['date'], values[idx], color='green', marker='v',
                      s=100, alpha=0.7, zorder=5)

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('账户价值 (元)', fontsize=11)
    ax.set_title('随机森林策略回测结果', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'rf_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {filename}")
    plt.show()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("基于随机森林的量化交易回测系统 (简化版)")
    print("="*70)
    print("⚠️  警告: 此版本使用全部数据训练，存在数据泄露")
    print("⚠️  仅用于验证策略逻辑和调试")
    print("="*70)

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
    }

    # 1. 训练模型
    predictor = RFPredictor()
    if not predictor.train_model(stock_pool, csv_dir='stock_data'):
        print("模型训练失败")
        return

    # 2. 运行回测
    cerebro, strat = run_backtest(
        stock_pool=stock_pool,
        predictor=predictor,
        initial_cash=100000,
        buy_threshold=0.55,  # 降低阈值
        sell_threshold=0.45   # 提高卖出阈值
    )

    if cerebro is None or strat is None:
        return

    # 3. 可视化
    plot_results(strat)

    print("\n回测完成！")


if __name__ == "__main__":
    main()
