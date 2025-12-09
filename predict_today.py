"""
每日预测脚本 - 增量更新数据并生成预测Excel

功能:
1. 增量更新股票数据（只获取最新数据追加到CSV）
2. 使用训练好的模型预测今日上涨概率
3. 生成Excel文件记录所有股票的预测结果

运行: python predict_today.py

数据源:
- 历史数据下载: baostock (download_stock_data_baostock.py)
- 增量更新: akshare (本脚本) - 支持盘中实时数据获取
"""

import akshare as ak
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')


# ak.stock_zh_a_hist.__globals__['url_stock_zh_a_hist'] = \
#     "http://push2his.eastmoney.com/api/qt/stock/kline/get"

# 全局变量：缓存实时行情数据
_spot_data_cache = None
_cache_time = None


def get_spot_data():
    """
    获取所有A股实时行情（使用缓存避免重复请求）

    返回:
        DataFrame: 包含所有股票实时行情的数据
    """
    global _spot_data_cache, _cache_time

    # 如果缓存存在且在5分钟内，直接返回缓存
    if _spot_data_cache is not None and _cache_time is not None:
        if (datetime.now() - _cache_time).total_seconds() < 300:
            return _spot_data_cache

    # 否则重新获取
    try:
        print("  📡 获取实时行情数据（新浪接口）...")
        _spot_data_cache = ak.stock_zh_a_spot_em()
        _cache_time = datetime.now()
        print(f"  ✓ 获取成功，共 {len(_spot_data_cache)} 只股票")
        return _spot_data_cache
    except Exception as e:
        print(f"  ✗ 获取实时行情失败: {str(e)[:80]}")
        return None


def update_stock_data_incremental(code, csv_path, spot_data=None, max_days=10):
    """
    使用新浪实时行情 + 历史数据接口增量更新股票数据

    策略:
    1. 如果CSV存在且最新日期是昨日 -> 从实时行情获取今日开盘数据，追加到CSV
    2. 如果CSV不存在或数据较老 -> 使用历史接口（会触发限流，但次数少）

    参数:
        code: 股票代码 (6位代码，如 '000001', '600036')
        csv_path: CSV文件路径
        spot_data: 实时行情DataFrame（提前获取，避免重复请求）
        max_days: 最多获取最近几天的数据

    返回:
        DataFrame: 更新后的完整数据

    优势:
        - 使用新浪实时行情接口（一次请求获取所有股票，避免频繁请求）
        - 只在必要时才调用历史接口
        - 支持盘中运行（9:30-15:00）获取当日实时数据
    """
    df_old = None

    try:
        # 1. 读取现有CSV数据
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_old['date'] = pd.to_datetime(df_old['date'])
            last_date = df_old['date'].max()
        else:
            df_old = None
            last_date = None

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # 2. 判断是否需要更新
        need_history_data = False
        need_today_data = False

        if last_date is None:
            # CSV不存在，需要获取历史数据
            need_history_data = True
            start_date = (datetime.now() - timedelta(days=max_days + 5)).strftime('%Y%m%d')
        elif last_date.date() < yesterday:
            # 数据较老，需要获取历史数据
            need_history_data = True
            start_date = last_date.strftime('%Y%m%d')
        elif last_date.date() == yesterday:
            # 最新数据是昨日，只需要获取今日实时数据
            need_today_data = True
        else:
            # 数据已是最新
            return df_old

        # 3. 获取今日实时数据（优先，避免频繁请求）
        if need_today_data and spot_data is not None:
            try:
                stock_info = spot_data[spot_data['代码'] == code]
                if len(stock_info) > 0:
                    row = stock_info.iloc[0]
                    # 提取今日数据
                    today_data = {
                        'date': pd.Timestamp(today),
                        'open': float(row['今开']) if pd.notna(row['今开']) and row['今开'] != '-' else None,
                        'close': float(row['最新价']) if pd.notna(row['最新价']) and row['最新价'] != '-' else None,
                        'high': float(row['最高']) if pd.notna(row['最高']) and row['最高'] != '-' else None,
                        'low': float(row['最低']) if pd.notna(row['最低']) and row['最低'] != '-' else None,
                        'volume': float(row['成交量']) if pd.notna(row['成交量']) and row['成交量'] != '-' else 0
                    }

                    # 处理盘中不完整数据
                    if today_data['open'] is not None and today_data['open'] > 0:
                        if today_data['close'] is None or today_data['close'] == 0:
                            today_data['close'] = today_data['open']
                        if today_data['high'] is None or today_data['high'] == 0:
                            today_data['high'] = today_data['open']
                        if today_data['low'] is None or today_data['low'] == 0:
                            today_data['low'] = today_data['open']
                        if today_data['volume'] == 0:
                            today_data['volume'] = 1

                        # 追加今日数据
                        df_today = pd.DataFrame([today_data])
                        df_merged = pd.concat([df_old, df_today], ignore_index=True)
                        df_merged = df_merged.sort_values('date').reset_index(drop=True)

                        # 保存
                        df_merged.to_csv(csv_path, index=False, encoding='utf-8-sig')
                        return df_merged
            except Exception as e:
                print(f"  ⚠️  {code} 实时数据获取失败，尝试历史接口: {str(e)[:50]}")
                need_history_data = True

        # 4. 如果需要历史数据，使用历史接口（会触发限流，但次数少）
        if need_history_data:
            end_date = datetime.now().strftime('%Y%m%d')

            # 添加延时
            time.sleep(0.3)

            df_new = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )

            if df_new is None or len(df_new) == 0:
                return df_old

            # 数据清洗：标准化列名
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            }

            df_new = df_new.rename(columns=column_mapping)
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df_new = df_new[required_cols].copy()
            df_new['date'] = pd.to_datetime(df_new['date'])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

            # 处理盘中不完整数据
            for idx, row in df_new.iterrows():
                if row['date'].date() == today and pd.notna(row['open']):
                    if pd.isna(row['close']) or row['close'] == 0:
                        df_new.loc[idx, 'close'] = row['open']
                    if pd.isna(row['high']) or row['high'] == 0:
                        df_new.loc[idx, 'high'] = row['open']
                    if pd.isna(row['low']) or row['low'] == 0:
                        df_new.loc[idx, 'low'] = row['open']
                    if pd.isna(row['volume']) or row['volume'] == 0:
                        df_new.loc[idx, 'volume'] = 1

            df_new = df_new.dropna(subset=['open', 'close'])
            df_new = df_new[df_new['open'] > 0].copy()

            if len(df_new) == 0:
                return df_old

            # 合并数据
            if df_old is not None:
                df_merged = pd.concat([df_old, df_new], ignore_index=True)
                df_merged = df_merged.drop_duplicates(subset=['date'], keep='last')
                df_merged = df_merged.sort_values('date').reset_index(drop=True)
            else:
                df_merged = df_new.sort_values('date').reset_index(drop=True)

            # 保存
            df_merged.to_csv(csv_path, index=False, encoding='utf-8-sig')
            return df_merged

        return df_old

    except Exception as e:
        print(f"  ✗ {code} 更新失败: {str(e)[:80]}")
        if df_old is not None:
            return df_old
        return None


def calculate_features(df, for_training=False):
    """
    计算技术指标特征（共10个特征，与final_strategy.py完全一致）

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


def predict_today():
    """主函数：增量更新数据并预测"""

    print(f"\n{'='*80}")
    print(f"📊 每日股票上涨概率预测")
    print(f"{'='*80}")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查并加载模型
    model_path = 'rf_model.pkl'
    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件不存在: {model_path}")
        print(f"请先运行 final_strategy.py 训练并保存模型")
        return

    print(f"\n[1/4] 📦 加载模型...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        stock_pool = model_data['stock_pool']

    print(f"  ✓ 模型加载成功")
    print(f"  ✓ 股票池: {len(stock_pool)} 只")
    print(f"  ✓ 特征数: {len(feature_cols)} 个")

    # 2. AkShare无需登录，直接开始更新
    print(f"\n[2/4] 🔌 使用AkShare数据源（支持盘中实时数据）...")
    print(f"  ✓ AkShare无需登录，直接获取数据")

    try:
        # 3. 增量更新所有股票数据
        print(f"\n[3/4] 📈 增量更新股票数据...")
        data_dir = 'stock_data'

        # 显示模型期望的特征列表
        print(f"\n  模型期望特征 ({len(feature_cols)} 个):")
        for i, feat in enumerate(feature_cols, 1):
            print(f"    {i}. {feat}")
        print()

        # 优先获取实时行情（一次性获取所有股票，避免频繁请求）
        spot_data = get_spot_data()

        predictions = []
        success_count = 0
        fail_count = 0
        first_error_shown = False  # 标记是否已显示第一个错误

        total = len(stock_pool)
        for idx, (code, name) in enumerate(stock_pool.items(), 1):
            # 使用与下载脚本一致的命名格式: {code}_{name}.csv
            csv_path = os.path.join(data_dir, f'{code}_{name}.csv')

            # 显示进度
            if idx % 10 == 0 or idx == 1:
                print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%) - {code} {name}")

            # 增量更新数据（传入实时行情数据，避免重复请求）
            df = update_stock_data_incremental(code, csv_path, spot_data=spot_data, max_days=10)

            if df is None or len(df) < 60:
                fail_count += 1
                if not first_error_shown:
                    print(f"\n  ⚠️  首个失败案例: {code} {name}")
                    print(f"      原因: 数据不足")
                    if df is None:
                        print(f"      详情: CSV文件读取失败或无数据")
                    else:
                        print(f"      详情: 数据行数 {len(df)} < 60 (不足以计算技术指标)")
                    print(f"      文件: {csv_path}")
                    first_error_shown = True
                continue

            # 计算特征
            df = calculate_features(df, for_training=False)

            # 检查特征列是否存在
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                fail_count += 1
                if not first_error_shown:
                    print(f"\n  ⚠️  首个失败案例: {code} {name}")
                    print(f"      原因: 缺少特征列")
                    print(f"      缺失特征: {missing_cols}")
                    print(f"      计算后特征数: {len(df.columns)}")
                    first_error_shown = True
                continue

            df = df.dropna(subset=feature_cols)

            if len(df) == 0:
                fail_count += 1
                if not first_error_shown:
                    print(f"\n  ⚠️  首个失败案例: {code} {name}")
                    print(f"      原因: 特征计算后所有行都包含NaN")
                    first_error_shown = True
                continue

            # 使用最新一行数据预测
            latest = df.iloc[-1]
            X = latest[feature_cols].values.reshape(1, -1)

            # 预测概率
            prob = model.predict_proba(X)[0][1]

            predictions.append({
                '股票代码': code,
                '股票名称': name,
                '预测概率': prob,
                '数据日期': latest['date'].strftime('%Y-%m-%d'),
                '收盘价': latest['close'],
                '开盘价': latest['open'],
                '昨日涨幅': latest.get('return_1d_prev', 0),
                '5日涨幅': latest.get('return_5d_prev', 0),
                '10日涨幅': latest.get('return_10d_prev', 0),
                '开盘跳空': latest.get('open_gap', 0),
                'RSI': latest.get('rsi_prev', 50),
                '量比': latest.get('volume_ratio_prev', 1),
                '布林带位置': latest.get('bb_position_prev', 0.5),
                '波动率': latest.get('volatility_prev', 0)
            })

            success_count += 1

            # 显示第一个成功案例的详细信息
            if success_count == 1:
                print(f"\n  ✅ 首个成功案例: {code} {name}")
                print(f"      CSV行数: {len(df)}")
                print(f"      最新日期: {latest['date'].strftime('%Y-%m-%d')}")
                print(f"      预测概率: {prob:.2%}\n")

        print(f"\n  ✓ 数据更新完成: 成功 {success_count}/{total}, 失败 {fail_count}")

    finally:
        pass  # AkShare无需登出

    if len(predictions) == 0:
        print(f"\n❌ 无有效预测数据")
        return

    # 4. 生成Excel文件
    print(f"\n[4/4] 📊 生成预测Excel...")

    df_predictions = pd.DataFrame(predictions)
    df_predictions = df_predictions.sort_values('预测概率', ascending=False)
    df_predictions['排名'] = range(1, len(df_predictions) + 1)

    # 调整列顺序
    cols = ['排名', '股票代码', '股票名称', '预测概率', '数据日期', '收盘价', '开盘价',
            '开盘跳空', '昨日涨幅', '5日涨幅', '10日涨幅', 'RSI', '量比', '布林带位置', '波动率']
    df_predictions = df_predictions[cols]

    # 保存到Excel
    output_file = f"stock_predictions_{datetime.now().strftime('%Y%m%d')}.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 写入完整预测结果
        df_predictions.to_excel(writer, sheet_name='完整预测', index=False)

        # 写入高概率股票（≥60%）
        df_high = df_predictions[df_predictions['预测概率'] >= 0.60]
        if len(df_high) > 0:
            df_high.to_excel(writer, sheet_name='高概率股票(≥60%)', index=False)

        # 写入中等概率股票（55%-60%）
        df_medium = df_predictions[(df_predictions['预测概率'] >= 0.55) &
                                   (df_predictions['预测概率'] < 0.60)]
        if len(df_medium) > 0:
            df_medium.to_excel(writer, sheet_name='中等概率股票(55-60%)', index=False)

    print(f"  ✓ Excel已生成: {output_file}")

    # 5. 显示摘要信息
    print(f"\n{'='*80}")
    print(f"📊 预测结果摘要")
    print(f"{'='*80}")
    print(f"总股票数: {len(df_predictions)}")
    print(f"平均预测概率: {df_predictions['预测概率'].mean():.2%}")
    print(f"")
    print(f"概率分布:")
    print(f"  ≥ 65%: {len(df_predictions[df_predictions['预测概率'] >= 0.65])} 只")
    print(f"  ≥ 60%: {len(df_predictions[df_predictions['预测概率'] >= 0.60])} 只")
    print(f"  ≥ 55%: {len(df_predictions[df_predictions['预测概率'] >= 0.55])} 只")
    print(f"  ≥ 50%: {len(df_predictions[df_predictions['预测概率'] >= 0.50])} 只")

    # 显示Top 10
    print(f"\n{'='*80}")
    print(f"🏆 Top 10 预测股票")
    print(f"{'='*80}")
    print(f"{'排名':<6} {'代码':<10} {'名称':<12} {'概率':<10} {'开盘价':<10} {'开盘跳空':<10}")
    print(f"{'-'*80}")

    for _, row in df_predictions.head(10).iterrows():
        print(f"{row['排名']:<6} {row['股票代码']:<10} {row['股票名称']:<12} "
              f"{row['预测概率']:>8.2%} {row['开盘价']:>9.2f} {row['开盘跳空']:>9.2%}")

    print(f"\n{'='*80}")
    print(f"✅ 预测完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    predict_today()
