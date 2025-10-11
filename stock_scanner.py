# A股量化选股扫描器
# 基于RSI+均线+成交量综合评分系统，扫描市场找出最佳买入机会

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

class StockScanner:
    """
    股票扫描器：使用综合评分系统筛选优质股票
    """

    def __init__(self, rsi_period=14, rsi_lower=30, rsi_upper=70,
                 ma_short=5, ma_medium=20, ma_long=60,
                 volume_ma_period=20, volume_ratio_threshold=1.2):
        """
        初始化扫描器参数（与1011.py策略保持一致）
        """
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.volume_ma_period = volume_ma_period
        self.volume_ratio_threshold = volume_ratio_threshold

    def get_stock_list(self, market='A股'):
        """
        获取股票列表
        market: 'A股', '沪深300', '中证500', '创业板'
        """
        try:
            if market == 'A股':
                # 获取所有A股列表
                df = ak.stock_zh_a_spot_em()
                stocks = df[['代码', '名称']].copy()
                stocks.columns = ['code', 'name']
                print(f"成功获取 {len(stocks)} 只A股股票")

            elif market == '沪深300':
                df = ak.index_stock_cons_csindex(symbol="000300")
                stocks = df[['成分券代码', '成分券名称']].copy()
                stocks.columns = ['code', 'name']
                print(f"成功获取沪深300成分股 {len(stocks)} 只")

            elif market == '中证500':
                df = ak.index_stock_cons_csindex(symbol="000905")
                stocks = df[['成分券代码', '成分券名称']].copy()
                stocks.columns = ['code', 'name']
                print(f"成功获取中证500成分股 {len(stocks)} 只")

            elif market == '创业板':
                df = ak.stock_zh_a_spot_em()
                stocks = df[df['代码'].str.startswith('300')][['代码', '名称']].copy()
                stocks.columns = ['code', 'name']
                print(f"成功获取创业板股票 {len(stocks)} 只")

            return stocks

        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame(columns=['code', 'name'])

    def get_stock_data(self, code, days=120):
        """
        获取单只股票的历史数据
        """
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df is None or len(df) < self.ma_long + 10:
                return None

            # 重命名列
            df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            }, inplace=True)

            df['date'] = pd.to_datetime(df['date'])

            return df

        except Exception as e:
            return None

    def calculate_rsi(self, prices, period=14):
        """
        计算RSI指标
        """
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        if down == 0:
            return 100

        rs = up / down
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_buy_score(self, df):
        """
        计算买入信号评分（满分100分）
        返回：评分、详细信息字典
        """
        if df is None or len(df) < self.ma_long:
            return 0, {}

        # 获取最新数据
        latest = df.iloc[-1]
        close_price = latest['close']
        volume = latest['volume']

        # 计算技术指标
        close_prices = df['close'].values

        # RSI
        rsi = self.calculate_rsi(close_prices, self.rsi_period)

        # 均线
        ma5 = df['close'].rolling(window=self.ma_short).mean().iloc[-1]
        ma20 = df['close'].rolling(window=self.ma_medium).mean().iloc[-1]
        ma60 = df['close'].rolling(window=self.ma_long).mean().iloc[-1]
        ma60_prev = df['close'].rolling(window=self.ma_long).mean().iloc[-2]

        # 成交量均线
        volume_ma = df['volume'].rolling(window=self.volume_ma_period).mean().iloc[-1]
        volume_ratio = volume / volume_ma if volume_ma > 0 else 0

        # === 评分系统 ===
        score = 0
        details = {}

        # 1. RSI超卖（权重40分）
        rsi_score = 0
        if rsi < self.rsi_lower:
            rsi_score = 40 * (self.rsi_lower - rsi) / self.rsi_lower
            score += rsi_score
        details['RSI'] = rsi
        details['RSI得分'] = rsi_score

        # 2. 均线趋势（权重35分）
        ma_score = 0
        if close_price > ma5:
            ma_score += 12
        if ma5 > ma20:
            ma_score += 12
        if ma20 > ma60 or ma60 >= ma60_prev:
            ma_score += 11

        score += ma_score
        details['MA5'] = ma5
        details['MA20'] = ma20
        details['MA60'] = ma60
        details['均线得分'] = ma_score

        # 3. 成交量放量（权重25分）
        volume_score = 0
        if volume_ratio >= 1.5:
            volume_score = 25
        elif volume_ratio >= 1.2:
            volume_score = 18
        elif volume_ratio >= 1.0:
            volume_score = 10

        score += volume_score
        details['量比'] = volume_ratio
        details['成交量得分'] = volume_score
        details['总分'] = score

        return score, details

    def scan_single_stock(self, code, name):
        """
        扫描单只股票
        """
        try:
            df = self.get_stock_data(code)
            if df is None:
                return None

            score, details = self.calculate_buy_score(df)

            if score > 0:  # 只返回有得分的股票
                result = {
                    '股票代码': code,
                    '股票名称': name,
                    '综合评分': round(score, 2),
                    'RSI': round(details.get('RSI', 0), 2),
                    'RSI得分': round(details.get('RSI得分', 0), 2),
                    'MA5': round(details.get('MA5', 0), 2),
                    'MA20': round(details.get('MA20', 0), 2),
                    'MA60': round(details.get('MA60', 0), 2),
                    '均线得分': round(details.get('均线得分', 0), 2),
                    '量比': round(details.get('量比', 0), 2),
                    '成交量得分': round(details.get('成交量得分', 0), 2),
                    '当前价格': round(df.iloc[-1]['close'], 2),
                    '扫描时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return result

        except Exception as e:
            pass

        return None

    def scan_market(self, market='沪深300', top_n=20, max_workers=10):
        """
        扫描市场并返回评分最高的股票

        Parameters:
        -----------
        market: str
            市场范围 ('A股', '沪深300', '中证500', '创业板')
        top_n: int
            返回前N只股票
        max_workers: int
            并发线程数

        Returns:
        --------
        DataFrame: 按评分排序的股票列表
        """
        print(f"\n{'='*60}")
        print(f"开始扫描 {market} 市场...")
        print(f"{'='*60}\n")

        # 获取股票列表
        stocks = self.get_stock_list(market)
        if stocks.empty:
            print("无法获取股票列表")
            return pd.DataFrame()

        # 多线程扫描
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(self.scan_single_stock, row['code'], row['name']): row
                for _, row in stocks.iterrows()
            }

            # 使用tqdm显示进度
            with tqdm(total=len(future_to_stock), desc="扫描进度") as pbar:
                for future in as_completed(future_to_stock):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

        # 转换为DataFrame并排序
        if not results:
            print("\n未找到符合条件的股票")
            return pd.DataFrame()

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('综合评分', ascending=False).reset_index(drop=True)

        # 只返回前N只
        top_stocks = df_results.head(top_n)

        print(f"\n{'='*60}")
        print(f"扫描完成！共扫描 {len(stocks)} 只股票，找到 {len(results)} 只有效候选")
        print(f"{'='*60}\n")

        return top_stocks

    def display_results(self, df_results):
        """
        美化显示扫描结果
        """
        if df_results.empty:
            print("没有结果可显示")
            return

        print(f"\n{'='*80}")
        print(f"{'股票选股结果 - 按综合评分排序':^70}")
        print(f"{'='*80}\n")

        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # 显示表格
        print(df_results.to_string(index=True))

        print(f"\n{'='*80}")
        print(f"推荐关注前3只股票：")
        print(f"{'='*80}")

        for idx, row in df_results.head(3).iterrows():
            print(f"\n🏆 第{idx+1}名: {row['股票名称']} ({row['股票代码']})")
            print(f"   综合评分: {row['综合评分']:.2f}分")
            print(f"   RSI: {row['RSI']:.2f} (得分:{row['RSI得分']:.2f})")
            print(f"   均线: MA5={row['MA5']:.2f}, MA20={row['MA20']:.2f}, MA60={row['MA60']:.2f} (得分:{row['均线得分']:.2f})")
            print(f"   量比: {row['量比']:.2f}x (得分:{row['成交量得分']:.2f})")
            print(f"   当前价格: ¥{row['当前价格']:.2f}")

    def save_results(self, df_results, filename=None):
        """
        保存扫描结果到CSV文件
        """
        if df_results.empty:
            print("没有结果可保存")
            return

        if filename is None:
            filename = f"stock_scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {filename}")


def main():
    """
    主函数 - 运行股票扫描
    """
    print("=== A股量化选股扫描系统 ===")
    print("基于RSI+均线+成交量综合评分\n")

    # 创建扫描器实例
    scanner = StockScanner(
        rsi_period=14,
        rsi_lower=30,
        rsi_upper=70,
        ma_short=5,
        ma_medium=20,
        ma_long=60,
        volume_ma_period=20
    )

    # 选择扫描范围
    print("请选择扫描范围：")
    print("1. 沪深300 (推荐，速度快)")
    print("2. 中证500")
    print("3. 创业板")
    print("4. 全部A股 (约5000只，耗时较长)")

    choice = input("\n请输入选项 (1-4，默认1): ").strip() or "1"

    market_map = {
        '1': '沪深300',
        '2': '中证500',
        '3': '创业板',
        '4': 'A股'
    }

    market = market_map.get(choice, '沪深300')

    # 设置返回数量
    top_n = int(input(f"\n返回前几只股票 (默认20): ").strip() or "20")

    # 开始扫描
    start_time = time.time()
    results = scanner.scan_market(market=market, top_n=top_n, max_workers=10)
    elapsed_time = time.time() - start_time

    # 显示结果
    if not results.empty:
        scanner.display_results(results)

        # 询问是否保存
        save = input(f"\n是否保存结果到CSV文件？(y/n，默认y): ").strip().lower() or 'y'
        if save == 'y':
            scanner.save_results(results)

    print(f"\n总耗时: {elapsed_time:.2f}秒")
    print("\n扫描完成！")


if __name__ == "__main__":
    main()
