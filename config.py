"""
========================================
量化交易系统 - 统一配置文件
========================================

📌 新手使用指南：
    1. 这是唯一需要修改的配置文件
    2. 所有参数都在这里，不需要去其他文件修改
    3. 每个参数都有详细说明，按需修改即可
    4. 修改后保存，然后运行其他脚本即可生效

📌 使用流程：
    1. 修改本文件的配置参数
    2. 运行 download_stock_data_baostock.py 下载数据
    3. 运行 final_strategy.py 训练模型和回测
    4. 运行 predict_today.py 每日预测

📌 快速开始（使用默认配置）：
    - 直接运行，无需修改任何参数
    - 系统会自动使用沪深300股票池
    - 默认参数已经过优化，适合大多数场景
"""

# ============================================================
# 第一部分：股票池配置（选择你要交易的股票范围）
# ============================================================

# 股票池选择
# 可选值: 'hs300'(沪深300), 'zz100'(中证100), 'custom'(自定义)
# 推荐: 'hs300' - 300只大中盘股票，数据质量好，流动性强
STOCK_POOL_TYPE = 'hs300'

# 自定义股票池（仅当 STOCK_POOL_TYPE='custom' 时生效）
# 格式: {'股票代码': '股票名称'}
# 示例: {'000001': '平安银行', '600036': '招商银行'}
CUSTOM_STOCK_POOL = {
    '000001': '平安银行',
    '600036': '招商银行',
    '600519': '贵州茅台',
    '000858': '五粮液',
    '601318': '中国平安',
}

# ============================================================
# 第二部分：数据下载配置
# ============================================================

# 数据下载时间范围
# 说明: 需要下载多长时间的历史数据
# 推荐: 至少3年数据，用于模型训练
DOWNLOAD_START_DATE = '2022-01-01'  # 下载开始日期
DOWNLOAD_END_DATE = '2025-11-21'    # 下载结束日期

# 数据存储目录
# 说明: 股票CSV文件保存的文件夹名称
DATA_DIR = 'stock_data'

# 是否下载指数数据
# 说明: 是否同时下载上证指数数据（用于对比）
DOWNLOAD_INDEX = True

# 下载时是否显示详细进度
# 说明: True=显示每只股票下载详情, False=只显示汇总信息
SHOW_DOWNLOAD_DETAILS = True

# ============================================================
# 第三部分：模型训练配置
# ============================================================

# 训练数据时间范围
# 说明: 用于训练模型的历史数据时间段
# 重要: 训练期必须早于回测期，避免数据泄露
TRAIN_START_DATE = '2022-01-01'  # 训练开始日期
TRAIN_END_DATE = '2025-06-30'    # 训练结束日期

# 随机森林模型参数
# n_estimators: 决策树数量，越多越准确但越慢（推荐: 100-200）
# max_depth: 树的最大深度，防止过拟合（推荐: 8-12）
# random_state: 随机种子，保证结果可复现（推荐: 42）
RF_N_ESTIMATORS = 100    # 决策树数量
RF_MAX_DEPTH = 10        # 最大深度
RF_RANDOM_STATE = 42     # 随机种子

# 模型保存路径
# 说明: 训练好的模型保存文件名
MODEL_SAVE_PATH = 'rf_model.pkl'

# ============================================================
# 第四部分：回测配置
# ============================================================

# 回测时间范围
# 说明: 模拟交易的时间段，用于验证策略效果
# 重要: 回测期必须晚于训练期
BACKTEST_START_DATE = '2025-07-01'  # 回测开始日期
BACKTEST_END_DATE = '2025-11-21'    # 回测结束日期

# 初始资金（元）
# 说明: 回测时的起始资金
INITIAL_CASH = 100000

# 交易手续费率
# 说明: 每笔交易的手续费比例（0.001 = 0.1%）
# 包含: 券商佣金 + 印花税 + 过户费
COMMISSION_RATE = 0.001

# ============================================================
# 第五部分：交易策略配置
# ============================================================

# 买入阈值（预测概率）
# 说明: 只买入预测上涨概率 > 此值的股票
# 推荐: 0.60 (60%) - 较高的确定性
# 范围: 0.50-0.70，越高越保守
BUY_THRESHOLD = 0.60

# 卖出阈值（预测概率）
# 说明: 当持仓股票概率 < 此值时卖出止损
# 推荐: 0.50 (50%) - 概率低于50%即止损
# 范围: 0.40-0.55，越低越保守
SELL_THRESHOLD = 0.50

# 仓位比例
# 说明: 买入时使用多少比例的资金
# 推荐: 0.95 (95%) - 保留5%现金应对手续费和滑点
# 范围: 0.80-0.99
POSITION_RATIO = 0.95

# ============================================================
# 第六部分：每日预测配置
# ============================================================

# 增量更新最大天数
# 说明: 每日预测时，最多获取最近几天的数据
# 推荐: 10天，足够计算技术指标
INCREMENTAL_MAX_DAYS = 10

# Excel输出配置
# 说明: 预测结果Excel文件的命名和内容设置

# 高概率股票阈值（用于Excel分类）
# 说明: 概率 >= 此值的股票会单独生成一个Sheet
HIGH_PROBABILITY_THRESHOLD = 0.60

# 中等概率股票阈值范围（用于Excel分类）
# 说明: 概率在此范围内的股票会单独生成一个Sheet
MEDIUM_PROBABILITY_MIN = 0.55
MEDIUM_PROBABILITY_MAX = 0.60

# 显示Top N股票
# 说明: 在终端显示预测概率最高的前N只股票
SHOW_TOP_N = 10

# ============================================================
# 第七部分：可视化配置
# ============================================================

# 回测结果图表保存路径
# 说明: 回测完成后，生成的可视化图表文件名
BACKTEST_CHART_PATH = 'backtest_visualization.png'

# 图表DPI（分辨率）
# 说明: 图表清晰度，越高越清晰但文件越大
# 推荐: 150（屏幕查看），300（打印输出）
CHART_DPI = 150

# ============================================================
# 第八部分：高级配置（一般无需修改）
# ============================================================

# 数据更新延时（秒）
# 说明: 下载数据时，每只股票之间的间隔时间
# 推荐: 0.2-0.5秒，避免请求过快被限流
DOWNLOAD_DELAY = 0.2

# 特征列名称（与模型一致，请勿修改）
# 说明: 技术指标特征的列名，修改会导致模型无法使用
FEATURE_COLS = [
    'return_1d_prev',      # 昨日收益率
    'return_5d_prev',      # 5日收益率
    'return_10d_prev',     # 10日收益率
    'ma_ratio_5_20_prev',  # 均线比率
    'volume_ratio_prev',   # 成交量比率
    'rsi_prev',            # RSI指标
    'volatility_prev',     # 波动率
    'bb_position_prev',    # 布林带位置
    'open_gap',            # 开盘跳空
    'open_vs_ma5'          # 开盘价相对均线
]

# 技术指标计算所需最小数据量
# 说明: 至少需要多少行数据才能计算技术指标
# 推荐: 60行（约3个月），用于计算20日均线等指标
MIN_DATA_ROWS = 60

# 警告消息开关
# 说明: 是否显示Python警告信息
# 推荐: False（隐藏警告，界面更简洁）
SHOW_WARNINGS = False

# ============================================================
# 第九部分：日志和调试配置
# ============================================================

# 是否显示详细日志
# 说明: True=显示所有操作细节, False=只显示关键信息
VERBOSE = True

# 是否保存交易日志到文件
# 说明: 是否将回测交易记录保存到CSV文件
SAVE_TRADE_LOG = True

# 交易日志文件名
TRADE_LOG_FILE = 'trade_log.csv'

# ============================================================
# 配置验证函数（自动检查配置是否合理）
# ============================================================

def validate_config():
    """
    验证配置参数是否合理
    如果发现问题，会打印警告信息
    """
    issues = []

    # 检查日期顺序
    if TRAIN_START_DATE >= TRAIN_END_DATE:
        issues.append("❌ 训练开始日期必须早于结束日期")

    if BACKTEST_START_DATE >= BACKTEST_END_DATE:
        issues.append("❌ 回测开始日期必须早于结束日期")

    if TRAIN_END_DATE >= BACKTEST_START_DATE:
        issues.append("⚠️  警告: 训练期和回测期有重叠，可能导致数据泄露")

    # 检查阈值范围
    if not (0 < BUY_THRESHOLD <= 1):
        issues.append("❌ 买入阈值必须在 0-1 之间")

    if not (0 < SELL_THRESHOLD <= 1):
        issues.append("❌ 卖出阈值必须在 0-1 之间")

    if SELL_THRESHOLD >= BUY_THRESHOLD:
        issues.append("⚠️  警告: 卖出阈值应该低于买入阈值")

    if not (0 < POSITION_RATIO <= 1):
        issues.append("❌ 仓位比例必须在 0-1 之间")

    # 检查资金
    if INITIAL_CASH < 10000:
        issues.append("⚠️  警告: 初始资金过低，可能无法有效交易")

    # 检查股票池
    if STOCK_POOL_TYPE == 'custom' and len(CUSTOM_STOCK_POOL) == 0:
        issues.append("❌ 自定义股票池不能为空")

    # 输出检查结果
    if issues:
        print("\n" + "="*70)
        print("配置检查发现以下问题：")
        print("="*70)
        for issue in issues:
            print(issue)
        print("="*70)
        print("请修改 config.py 文件后重新运行\n")
        return False
    else:
        print("\n✅ 配置检查通过！\n")
        return True


def print_config_summary():
    """
    打印当前配置摘要（便于确认参数）
    """
    print("\n" + "="*70)
    print("当前配置摘要")
    print("="*70)
    print(f"📊 股票池类型: {STOCK_POOL_TYPE}")
    print(f"📅 训练期: {TRAIN_START_DATE} ~ {TRAIN_END_DATE}")
    print(f"📅 回测期: {BACKTEST_START_DATE} ~ {BACKTEST_END_DATE}")
    print(f"💰 初始资金: {INITIAL_CASH:,} 元")
    print(f"📈 买入阈值: {BUY_THRESHOLD:.0%}")
    print(f"📉 卖出阈值: {SELL_THRESHOLD:.0%}")
    print(f"💼 仓位比例: {POSITION_RATIO:.0%}")
    print(f"🤖 模型参数: {RF_N_ESTIMATORS}棵树, 深度{RF_MAX_DEPTH}")
    print("="*70 + "\n")


def get_stock_pool():
    """
    根据配置返回股票池

    返回:
        dict: {股票代码: 股票名称}
    """
    if STOCK_POOL_TYPE == 'hs300':
        from hs300_stocks import get_hs300_stock_pool
        return get_hs300_stock_pool()
    elif STOCK_POOL_TYPE == 'zz100':
        # 使用download_stock_data_baostock.py中的中证100备用列表
        from download_stock_data_baostock import get_zz100_stocks_fallback
        return get_zz100_stocks_fallback()
    elif STOCK_POOL_TYPE == 'custom':
        return CUSTOM_STOCK_POOL
    else:
        raise ValueError(f"不支持的股票池类型: {STOCK_POOL_TYPE}")


# ============================================================
# 配置文件使用示例
# ============================================================

if __name__ == "__main__":
    print("""
    ========================================
    量化交易系统 - 配置文件
    ========================================

    这是系统的统一配置文件，包含所有可调参数。

    📌 快速开始：
        1. 查看并修改上方的配置参数
        2. 运行此文件检查配置: python config.py
        3. 确认无误后，运行其他脚本

    📌 推荐配置流程：
        Step 1: 选择股票池（STOCK_POOL_TYPE）
        Step 2: 设置数据下载范围（DOWNLOAD_START/END_DATE）
        Step 3: 设置训练和回测时间（TRAIN/BACKTEST日期）
        Step 4: 调整交易策略参数（BUY/SELL_THRESHOLD）
        Step 5: 运行下载、训练、预测脚本

    📌 常见问题：
        Q: 如何更换股票池？
        A: 修改 STOCK_POOL_TYPE，或填写 CUSTOM_STOCK_POOL

        Q: 如何提高策略收益？
        A: 调整 BUY_THRESHOLD（提高=更保守）

        Q: 如何减少交易次数？
        A: 提高 BUY_THRESHOLD，降低 SELL_THRESHOLD

    ========================================
    """)

    # 验证配置
    if validate_config():
        print_config_summary()

        # 显示股票池信息
        try:
            stock_pool = get_stock_pool()
            print(f"✅ 当前股票池包含 {len(stock_pool)} 只股票")
            print(f"   前5只: {list(stock_pool.items())[:5]}")
        except Exception as e:
            print(f"❌ 获取股票池失败: {e}")
