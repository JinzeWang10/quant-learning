"""
基于随机森林的量价选股模型
使用技术指标预测未来N日涨跌，选出高概率上涨股票
数据源：本地CSV文件 (从 stock_data 目录读取)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
import os

warnings.filterwarnings('ignore')


def get_stock_data(code, name, csv_dir='stock_data'):
    """
    从本地CSV文件读取股票历史数据

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
            return None

        # 读取CSV
        df = pd.read_csv(csv_file)

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

        # 检查必需列
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None

        # 确保数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除空值
        df.dropna(subset=required, inplace=True)

        if len(df) < 100:
            return None

        return df

    except Exception:
        return None


def calculate_features(df):
    """计算技术指标特征"""
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

    # 布林带
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # 标签：未来N天是否上涨
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['label'] = (df['future_return'] > 0.03).astype(int)  # 5日涨幅>3%为1

    return df


def prepare_training_data(stock_pool):
    """准备训练数据"""
    all_data = []

    print("正在从CSV加载训练数据...")
    for code, name in stock_pool.items():
        df = get_stock_data(code, name, csv_dir='stock_data')
        if df is not None:
            df = calculate_features(df)
            df['stock_code'] = code
            all_data.append(df)
            print(f"  ✓ {name}({code}): {len(df)}条")
        else:
            print(f"  ✗ {name}({code}): CSV文件不存在或数据不足")

    if not all_data:
        return None

    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna()

    # 特征列
    feature_cols = ['return_1d', 'return_5d', 'return_10d', 'ma_ratio_5_20',
                   'volume_ratio', 'rsi', 'volatility', 'bb_position']

    X = combined[feature_cols]
    y = combined['label']

    return X, y, feature_cols


def train_model(X, y):
    """训练随机森林模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n训练随机森林模型...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 评估
    y_pred = rf.predict(X_test)
    print(f"\n模型准确率: {accuracy_score(y_test, y_pred):.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不涨', '涨']))

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性:")
    print(importance.to_string(index=False))

    return rf


def select_stocks(model, stock_pool, feature_cols, top_n=10):
    """选股：预测并返回最有可能上涨的股票"""
    predictions = []

    print("\n正在从CSV分析股票...")
    for code, name in stock_pool.items():
        df = get_stock_data(code, name, csv_dir='stock_data')
        if df is None or len(df) < 50:
            continue

        df = calculate_features(df)
        df = df.dropna()

        if len(df) == 0:
            continue

        # 使用最新数据预测
        latest = df[feature_cols].iloc[-1:].values
        prob = model.predict_proba(latest)[0][1]  # 上涨概率

        predictions.append({
            'code': code,
            'name': name,
            'prob': prob,
            'close': df['close'].iloc[-1],
            'rsi': df['rsi'].iloc[-1],
            'volume_ratio': df['volume_ratio'].iloc[-1]
        })

    # 按概率排序
    predictions = sorted(predictions, key=lambda x: x['prob'], reverse=True)

    print(f"\n{'='*80}")
    print(f"Top {top_n} 推荐股票（按上涨概率排序）")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'股票代码':<8} {'股票名称':<10} {'上涨概率':<10} {'最新价':<8} {'RSI':<6} {'量比':<6}")
    print(f"{'-'*80}")

    for i, stock in enumerate(predictions[:top_n], 1):
        print(f"{i:<4} {stock['code']:<8} {stock['name']:<10} {stock['prob']:>8.2%} "
              f"{stock['close']:>8.2f} {stock['rsi']:>6.1f} {stock['volume_ratio']:>6.2f}")

    return predictions[:top_n]


def main():
    print("=== 基于随机森林的量价选股系统 ===")
    print("数据源: stock_data 目录下的CSV文件")
    print("使用前: 请先运行 download_stock_data_baostock.py 下载数据\n")

    # 股票池
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

    # 1. 准备训练数据
    result = prepare_training_data(stock_pool)
    if result is None:
        print("\n错误: 无法加载数据")
        print("提示: 请先运行 download_stock_data_baostock.py 下载数据到 stock_data 目录")
        return

    X, y, feature_cols = result
    print(f"\n训练集大小: {len(X)} 条记录")
    print(f"正样本比例: {y.mean():.2%}")

    # 2. 训练模型
    model = train_model(X, y)

    # 3. 选股
    top_stocks = select_stocks(model, stock_pool, feature_cols, top_n=10)

    print(f"\n{'='*80}")
    print("选股完成！建议关注上涨概率 > 60% 的股票")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
