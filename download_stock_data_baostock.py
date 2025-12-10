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


def get_index_constituent_stocks(index_code='sh.000903', date='2025-11-21'):
    """
    获取指数成分股列表

    参数:
        index_code: 指数代码
            - 'sh.000903': 中证100
            - 'sh.000300': 沪深300
            - 'sh.000905': 中证500
            - 'sh.000852': 中证1000
        date: 查询日期 'YYYY-MM-DD'

    返回:
        dict: {股票代码: 股票名称}
        None: 如果获取失败
    """
    print(f"\n正在获取指数成分股列表...")
    print(f"指数代码: {index_code}")
    print(f"查询日期: {date}")

    try:
        # 查询指数成分股
        rs = bs.query_sz50_stocks(date=date) if index_code == 'sh.000016' else \
             bs.query_hs300_stocks(date=date) if index_code == 'sh.000300' else \
             bs.query_zz500_stocks(date=date) if index_code == 'sh.000905' else None

        # 中证100需要特殊处理（baostock没有直接接口，通过沪深300筛选）
        if index_code == 'sh.000903':
            print("提示: baostock没有中证100接口，使用备用方案...")
            # 方案：返回预定义的中证100股票池（2024年成分股）
            stock_pool = get_zz100_stocks_fallback()
            print(f"✓ 成功获取中证100成分股: {len(stock_pool)}只")
            return stock_pool

        if rs is None:
            print(f"✗ 不支持的指数代码: {index_code}")
            return None

        if rs.error_code != '0':
            print(f"✗ 查询失败: {rs.error_msg}")
            return None

        # 获取数据
        stock_list = []
        while (rs.error_code == '0') & rs.next():
            stock_list.append(rs.get_row_data())

        if not stock_list:
            print(f"✗ 未获取到成分股数据")
            return None

        # 转换为字典 {代码: 名称}
        stock_pool = {}
        for stock in stock_list:
            code = stock[1]  # code字段
            name = stock[2]  # code_name字段
            # 去掉前缀 sz./sh.
            code = code.split('.')[-1]
            stock_pool[code] = name

        print(f"✓ 成功获取成分股: {len(stock_pool)}只")
        return stock_pool

    except Exception as e:
        print(f"✗ 获取成分股异常: {str(e)}")
        return None


def get_zz100_stocks_fallback():
    """
    中证100成分股备用列表（2024年典型成分股）
    由于baostock没有直接接口，这里使用预定义列表
    """
    return {
        # 金融 (银行、保险、证券)
        '601398': '工商银行', '601939': '建设银行', '601288': '农业银行',
        '601988': '中国银行', '600036': '招商银行', '600000': '浦发银行',
        '601166': '兴业银行', '600016': '民生银行', '601328': '交通银行',
        '601818': '光大银行', '002142': '宁波银行', '601601': '中国太保',
        '601318': '中国平安', '601628': '中国人寿', '601336': '新华保险',
        '600030': '中信证券', '601688': '华泰证券', '600999': '招商证券',
        '601211': '国泰君安', '600837': '海通证券',

        # 消费 (食品饮料、家电)
        '600519': '贵州茅台', '000858': '五粮液', '000568': '泸州老窖',
        '600809': '山西汾酒', '000596': '古井贡酒', '603288': '海天味业',
        '600887': '伊利股份', '000333': '美的集团', '000651': '格力电器',
        '600690': '海尔智家',

        # 医药
        '600276': '恒瑞医药', '300760': '迈瑞医疗', '603259': '药明康德',
        '000661': '长春高新', '600196': '复星医药', '002007': '华兰生物',

        # 科技 (电子、通信、计算机)
        '002415': '海康威视', '000063': '中兴通讯', '002475': '立讯精密',
        '300059': '东方财富', '002230': '科大讯飞', '600588': '用友网络',
        '688111': '金山办公', '600745': '闻泰科技', '002049': '紫光国微',
        '688981': '中芯国际',

        # 新能源与新材料
        '300750': '宁德时代', '002594': '比亚迪', '601012': '隆基绿能',
        '300014': '亿纬锂能', '002460': '赣锋锂业', '688599': '天合光能',
        '600438': '通威股份',

        # 周期 (化工、建材、钢铁、煤炭)
        '600585': '海螺水泥', '600309': '万华化学', '601899': '紫金矿业',
        '601088': '中国神华', '600019': '宝钢股份', '000002': '万科A',
        '600048': '保利发展', '600346': '恒力石化', '601225': '陕西煤业',

        # 能源 (石油、电力)
        '601857': '中国石油', '600028': '中国石化', '600900': '长江电力',
        '601088': '中国神华', '601898': '中煤能源',

        # 工业 (机械、军工、交运)
        '600031': '三一重工', '000338': '潍柴动力', '601919': '中远海控',
        '601021': '春秋航空', '600760': '中航沈飞', '600893': '航发动力',
        '002352': '顺丰控股', '600009': '上海机场',

        # 传媒与互联网
        '002027': '分众传媒', '600588': '用友网络',

        # 其他
        '601888': '中国中免', '600406': '国电南瑞', '600845': '宝信软件',
        '000725': '京东方A', '600050': '中国联通', '000001': '平安银行',
        '601211': '国泰君安', '002736': '国信证券'
    }


def download_index_data(index_code, index_name, start_date, end_date, output_dir='stock_data'):
    """
    下载指数数据（如上证指数）

    参数:
        index_code: 指数代码 (如 'sh.000001' 上证指数)
        index_name: 指数名称
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录

    返回:
        bool: 是否成功
    """
    print(f"\n下载 {index_name}({index_code})...")

    try:
        # 查询指数日K线数据
        rs = bs.query_history_k_data_plus(
            index_code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )

        if rs.error_code != '0':
            print(f"    ✗ 查询失败: {rs.error_msg}")
            return False

        # 获取数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            print(f"    ✗ 数据为空")
            return False

        # 转换为DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)

        # 数据清洗
        df['date'] = pd.to_datetime(df['date'])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除空值
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        if df.empty:
            print(f"    ✗ 过滤后数据为空")
            return False

        print(f"  ✓ 成功获取 {len(df)} 条数据")
        print(f"  日期范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存CSV
        save_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df_save = df[save_cols].copy()

        filename = os.path.join(output_dir, f"{index_code}_{index_name}.csv")
        df_save.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"  ✓ 已保存: {filename}")

        return True

    except Exception as e:
        print(f"    ✗ 下载异常: {str(e)[:80]}")
        return False


def batch_download(stock_pool, start_date, end_date, output_dir='stock_data', download_index=True):
    """
    批量下载股票数据

    参数:
        stock_pool: 股票池 {code: name}
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
        download_index: 是否下载上证指数
    """

    print("="*70)
    print("股票数据批量下载工具 (baostock数据源)")
    print("="*70)
    print(f"数据源: baostock.com - 免费稳定的A股数据")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"股票数量: {len(stock_pool)}")
    print(f"保存目录: {output_dir}")
    print(f"复权方式: 前复权")
    print(f"下载指数: {'是' if download_index else '否'}")
    print("="*70)

    # 批量下载
    success_count = 0
    fail_count = 0
    fail_stocks = []

    # 注意：batch_download假设已经登录baostock
    try:
        # 首先下载上证指数
        if download_index:
            print("\n" + "="*70)
            print("下载指数数据")
            print("="*70)
            if download_index_data('sh.000001', '上证指数', start_date, end_date, output_dir):
                print("✓ 上证指数下载成功")
            else:
                print("✗ 上证指数下载失败")
            time.sleep(0.5)

        # 下载股票数据
        print("\n" + "="*70)
        print("下载股票数据")
        print("="*70)

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
        pass  # 登出由main函数处理

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
    """主函数 - 从config.py读取配置"""

    # 尝试导入配置文件
    try:
        import config
        print("\n✓ 已加载配置文件 config.py")

        # 验证配置
        if not config.validate_config():
            print("\n请先修改 config.py 文件，确保配置正确")
            return

        # 从配置文件获取参数
        stock_pool = config.get_stock_pool()
        start_date = config.DOWNLOAD_START_DATE
        end_date = config.DOWNLOAD_END_DATE
        output_dir = config.DATA_DIR
        download_index = config.DOWNLOAD_INDEX

        print(f"\n✓ 使用配置:")
        print(f"  股票池: {config.STOCK_POOL_TYPE} ({len(stock_pool)}只)")
        print(f"  时间范围: {start_date} ~ {end_date}")
        print(f"  保存目录: {output_dir}")

    except ImportError:
        print("\n⚠️  未找到 config.py，使用交互模式")
        print("\n请选择运行模式:")
        print("1. 测试单只股票 (推荐首次使用)")
        print("2. 下载指数成分股 (推荐)")
        print("3. 使用自定义股票池")

        choice = input("\n请输入选项 (1/2/3，直接回车默认2): ").strip() or "2"

        if choice == "1":
            test_single_stock()
            return

        # 登录baostock
        print("\n正在登录baostock系统...")
        lg = bs.login()
        if lg.error_code != '0':
            print(f"✗ 登录失败: {lg.error_msg}")
            return
        print(f"✓ 登录成功")

        try:
            if choice == "2":
                # 下载指数成分股
                print("\n请选择指数:")
                print("1. 中证100 (100只大盘蓝筹)")
                print("2. 沪深300 (300只大中盘)")
                print("3. 中证500 (500只中小盘)")

                index_choice = input("\n请输入选项 (1/2/3，直接回车默认1): ").strip() or "1"

                if index_choice == "1":
                    index_code = 'sh.000903'
                    index_name = '中证100'
                elif index_choice == "2":
                    index_code = 'sh.000300'
                    index_name = '沪深300'
                elif index_choice == "3":
                    index_code = 'sh.000905'
                    index_name = '中证500'
                else:
                    print("无效选择，使用默认中证100")
                    index_code = 'sh.000903'
                    index_name = '中证100'

                # 获取成分股
                stock_pool = get_index_constituent_stocks(index_code=index_code)

                if stock_pool is None or len(stock_pool) == 0:
                    print("\n✗ 获取成分股失败")
                    return

                print(f"\n✓ 成功获取{index_name}成分股: {len(stock_pool)}只")

            else:
                # 使用自定义股票池
                print("\n使用自定义股票池...")
                stock_pool = get_custom_stock_pool()

        except Exception as e:
            print(f"\n✗ 初始化失败: {e}")
            bs.logout()
            return

        # 定义股票池（仅在choice=3时使用）
        def get_custom_stock_pool():
            return {
            '000001': '平安银行',
            '600036': '招商银行',
            '600519': '贵州茅台',
            '000858': '五粮液',
            '601318': '中国平安',
            }

        # 下载参数
        start_date = "2022-01-01"
        end_date = "2025-11-21"
        output_dir = "stock_data"
        download_index = True

    # 登录baostock
    print("\n正在登录baostock系统...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"✗ 登录失败: {lg.error_msg}")
        return
    print(f"✓ 登录成功")

    print(f"\n准备下载 {len(stock_pool)} 只股票")
    print(f"时间范围: {start_date} ~ {end_date}")

    confirm = input("\n是否继续? (y/n，直接回车默认y): ").strip().lower() or 'y'

    if confirm != 'y':
        print("已取消")
        bs.logout()
        return

    # 批量下载
    try:
        batch_download(stock_pool, start_date, end_date, output_dir, download_index=download_index)

        # 如果是中证100，额外下载中证100指数数据
        try:
            if 'config' in dir() and config.STOCK_POOL_TYPE == 'zz100':
                print("\n下载中证100指数数据...")
                download_index_data('sh.000903', '中证100', start_date, end_date, output_dir)
        except:
            pass

    finally:
        bs.logout()
        print("\n✓ 已登出baostock")

    print("\n" + "="*70)
    print("使用说明:")
    print("="*70)
    print("1. 下载完成后，运行 final_strategy.py 进行回测")
    print("2. CSV文件可重复使用，无需每次回测都下载")
    print("3. 需要更新数据时，重新运行本脚本即可")
    print("4. baostock数据每日更新，建议定期下载最新数据")
    print("="*70)


if __name__ == "__main__":
    main()
