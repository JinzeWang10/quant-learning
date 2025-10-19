"""
调试脚本：检查随机森林策略为什么没有交易
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from backtest_rf_strategy import RFPredictor


def test_model_training():
    """测试模型训练"""
    print("="*70)
    print("测试1: 模型训练")
    print("="*70)

    stock_pool = {
        '000001': '平安银行',
        '600036': '招商银行',
    }

    predictor = RFPredictor()

    success = predictor.train_model(
        stock_pool,
        csv_dir='stock_data',
        train_start='2023-01-01',
        train_end='2024-06-30',
        test_start='2024-07-01',
        test_end='2024-10-18'
    )

    if success:
        print("\n✓ 模型训练成功")
        print(f"特征列: {predictor.feature_cols}")
        return predictor
    else:
        print("\n✗ 模型训练失败")
        return None


def test_prediction(predictor):
    """测试预测功能"""
    print("\n" + "="*70)
    print("测试2: 预测功能")
    print("="*70)

    # 加载一只股票的数据
    csv_file = 'stock_data/000001_平安银行.csv'
    if not os.path.exists(csv_file):
        print(f"✗ 文件不存在: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])

    # 测试不同日期的预测
    print("\n测试回测期的预测概率:")
    test_dates = pd.date_range('2024-07-01', '2024-07-10', freq='B')  # 工作日

    for date in test_dates:
        # 获取该日期之前的数据
        hist_df = df[df['date'] < date].tail(100).copy()

        if len(hist_df) < 30:
            continue

        prob = predictor.predict_probability(hist_df)

        if prob is not None:
            print(f"  {date.strftime('%Y-%m-%d')}: 上涨概率 = {prob:.2%}")
        else:
            print(f"  {date.strftime('%Y-%m-%d')}: 预测失败")


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*70)
    print("测试3: 回测期数据加载")
    print("="*70)

    csv_file = 'stock_data/000001_平安银行.csv'
    if not os.path.exists(csv_file):
        print(f"✗ 文件不存在: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])

    print(f"\n完整数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"完整数据条数: {len(df)}")

    # 筛选回测期数据
    backtest_start = pd.to_datetime('2024-07-01')
    backtest_end = pd.to_datetime('2024-10-18')

    backtest_df = df[(df['date'] >= backtest_start) & (df['date'] <= backtest_end)]

    print(f"\n回测期数据范围: {backtest_df['date'].min()} ~ {backtest_df['date'].max()}")
    print(f"回测期数据条数: {len(backtest_df)}")

    if len(backtest_df) == 0:
        print("\n✗ 警告: 回测期没有数据！")
    elif len(backtest_df) < 70:
        print(f"\n✗ 警告: 回测期数据不足70条，只有{len(backtest_df)}条")
    else:
        print("\n✓ 回测期数据正常")


def test_feature_calculation():
    """测试特征计算"""
    print("\n" + "="*70)
    print("测试4: 特征计算")
    print("="*70)

    csv_file = 'stock_data/000001_平安银行.csv'
    if not os.path.exists(csv_file):
        print(f"✗ 文件不存在: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])

    # 取最近100条数据
    df = df.tail(100).copy()

    predictor = RFPredictor()
    df_features = predictor.calculate_features(df)

    if df_features is None:
        print("✗ 特征计算失败")
        return

    print(f"\n原始数据: {len(df)} 条")
    print(f"计算特征后: {len(df_features)} 条")

    # 删除空值
    df_clean = df_features.dropna()
    print(f"删除空值后: {len(df_clean)} 条")

    if len(df_clean) > 0:
        print("\n最新5条特征:")
        feature_cols = ['return_1d', 'return_5d', 'return_10d', 'ma_ratio_5_20',
                       'volume_ratio', 'rsi', 'volatility', 'bb_position']

        print(df_clean[feature_cols].tail(5).to_string())
    else:
        print("\n✗ 警告: 所有数据都被过滤掉了")


def test_backtest_data_range():
    """测试回测数据是否足够"""
    print("\n" + "="*70)
    print("测试5: 检查回测期数据是否满足回测要求")
    print("="*70)

    stock_pool = {
        '000001': '平安银行',
        '600036': '招商银行',
    }

    backtest_start = pd.to_datetime('2024-07-01')
    backtest_end = pd.to_datetime('2024-10-18')

    for code, name in stock_pool.items():
        csv_file = f'stock_data/{code}_{name}.csv'
        if not os.path.exists(csv_file):
            print(f"✗ {name}: 文件不存在")
            continue

        df = pd.read_csv(csv_file)
        df.columns = [col.lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])

        # 筛选回测期
        backtest_df = df[(df['date'] >= backtest_start) & (df['date'] <= backtest_end)].copy()

        print(f"\n{name}({code}):")
        print(f"  回测期数据: {len(backtest_df)} 条")

        if len(backtest_df) >= 70:
            print(f"  ✓ 数据充足 (≥70条)")
        elif len(backtest_df) > 0:
            print(f"  ✗ 数据不足 (<70条)，backtrader可能跳过")
        else:
            print(f"  ✗ 没有数据")


def main():
    print("\n" + "="*70)
    print("随机森林策略调试工具")
    print("="*70)

    # 测试1: 数据加载
    test_data_loading()

    # 测试2: 回测数据范围
    test_backtest_data_range()

    # 测试3: 特征计算
    test_feature_calculation()

    # 测试4: 模型训练
    predictor = test_model_training()

    # 测试5: 预测功能
    if predictor:
        test_prediction(predictor)

    print("\n" + "="*70)
    print("调试完成")
    print("="*70)


if __name__ == "__main__":
    main()
