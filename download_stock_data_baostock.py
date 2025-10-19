"""
股票数据下载工具 (使用baostock数据源)
数据源: http://baostock.com/
优势: 免费、稳定、数据全面、无需token
"""

import baostock as bs
import pandas as pd
import os
from datetime import datetime
import time


def format_stock_code(code):
    """
    转换股票代码格式
    000001 -> sz.000001 (深圳)
    600036 -> sh.600036 (上海)
    """
    if code.startswith('6'):
        return f'sh.{code}'
    else:
        return f'sz.{code}'


def download_stock_with_baostock(code, name, start_date, end_date):
    """
    使用baostock下载单只股票数据

    参数:
        code: 股票代码 (如 '000001', '600036')
        name: 股票名称
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

    返回:
        DataFrame or None
    """
    try:
        # 转换代码格式
        bs_code = format_stock_code(code)

        # 查询日K线数据
        # frequency="d" 表示日K线
        # adjustflag="3" 表示前复权
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 前复权
        )

        if rs.error_code != '0':
            print(f"    ✗ 查询失败: {rs.error_msg}")
            return None

        # 获取数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            print(f"    ✗ 数据为空")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)

        # 数据清洗
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除空值
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

        # 过滤停牌日(成交量为0)
        df = df[df['volume'] > 0].copy()

        if df.empty:
            print(f"    ✗ 过滤后数据为空")
            return None

        return df

    except Exception as e:
        print(f"    ✗ 下载异常: {str(e)[:80]}")
        return None


def download_and_save(code, name, start_date, end_date, output_dir='stock_data'):
    """
    下载并保存单只股票数据

    返回:
        bool: 是否成功
    """
    print(f"\n下载 {name}({code})...")

    df = download_stock_with_baostock(code, name, start_date, end_date)

    if df is None or df.empty:
        print(f"  ✗ 下载失败")
        return False

    print(f"  ✓ 成功获取 {len(df)} 条数据")
    print(f"  日期范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存CSV (只保留回测需要的列)
    save_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df_save = df[save_cols].copy()

    filename = os.path.join(output_dir, f"{code}_{name}.csv")
    df_save.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已保存: {filename}")

    return True


def batch_download(stock_pool, start_date, end_date, output_dir='stock_data'):
    """
    批量下载股票数据

    参数:
        stock_pool: 股票池 {code: name}
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
    """

    print("="*70)
    print("股票数据批量下载工具 (baostock数据源)")
    print("="*70)
    print(f"数据源: baostock.com - 免费稳定的A股数据")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"股票数量: {len(stock_pool)}")
    print(f"保存目录: {output_dir}")
    print(f"复权方式: 前复权")
    print("="*70)

    # 登录baostock系统
    print("\n正在登录baostock系统...")
    lg = bs.login()

    if lg.error_code != '0':
        print(f"✗ 登录失败: {lg.error_msg}")
        return

    print(f"✓ 登录成功")

    # 批量下载
    success_count = 0
    fail_count = 0
    fail_stocks = []

    try:
        for i, (code, name) in enumerate(stock_pool.items(), 1):
            print(f"\n进度: [{i}/{len(stock_pool)}]")

            if download_and_save(code, name, start_date, end_date, output_dir):
                success_count += 1
            else:
                fail_count += 1
                fail_stocks.append(f"{name}({code})")

            # 避免请求过快
            if i < len(stock_pool):
                time.sleep(0.2)

    finally:
        # 登出系统
        print("\n正在登出系统...")
        bs.logout()
        print("✓ 已登出")

    # 统计结果
    print("\n" + "="*70)
    print("下载完成!")
    print("="*70)
    print(f"✓ 成功: {success_count} 只")
    print(f"✗ 失败: {fail_count} 只")

    if fail_stocks:
        print(f"\n失败列表:")
        for stock in fail_stocks:
            print(f"  - {stock}")

    print(f"\n数据保存位置: {os.path.abspath(output_dir)}")
    print("="*70)

    # 列出下载的文件
    if success_count > 0:
        print(f"\n已下载文件 (前10个):")
        try:
            csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
            for i, f in enumerate(csv_files[:10], 1):
                print(f"  {i}. {f}")
            if len(csv_files) > 10:
                print(f"  ... 共 {len(csv_files)} 个文件")
        except:
            pass


def test_single_stock():
    """测试单只股票下载"""
    print("="*70)
    print("测试单只股票下载")
    print("="*70)

    # 登录
    print("\n正在登录baostock...")
    lg = bs.login()

    if lg.error_code != '0':
        print(f"✗ 登录失败: {lg.error_msg}")
        return

    print("✓ 登录成功")

    try:
        code = "600036"
        name = "招商银行"
        start_date = "2023-01-01"
        end_date = "2024-10-18"

        success = download_and_save(code, name, start_date, end_date, output_dir='test_data')

        if success:
            print("\n✓ 测试成功！baostock数据源工作正常")
            print("可以开始批量下载")
        else:
            print("\n✗ 测试失败，请检查网络连接")

    finally:
        bs.logout()
        print("\n✓ 已登出")


def main():
    """主函数"""

    print("\n请选择运行模式:")
    print("1. 测试单只股票 (推荐首次使用)")
    print("2. 批量下载股票池")

    choice = input("\n请输入选项 (1/2，直接回车默认2): ").strip() or "2"

    if choice == "1":
        test_single_stock()
        return

    # 定义股票池
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


    # 下载参数
    start_date = "2022-01-01"
    end_date = "2025-10-18"
    output_dir = "stock_data"

    print(f"\n准备下载 {len(stock_pool)} 只股票")
    print(f"时间范围: {start_date} ~ {end_date}")

    confirm = input("\n是否继续? (y/n，直接回车默认y): ").strip().lower() or 'y'

    if confirm != 'y':
        print("已取消")
        return

    # 批量下载
    batch_download(stock_pool, start_date, end_date, output_dir)

    print("\n" + "="*70)
    print("使用说明:")
    print("="*70)
    print("1. 下载完成后，运行 multi_stock_bt_csv.py 进行回测")
    print("2. CSV文件可重复使用，无需每次回测都下载")
    print("3. 需要更新数据时，重新运行本脚本即可")
    print("4. baostock数据每日更新，建议定期下载最新数据")
    print("="*70)


if __name__ == "__main__":
    main()
