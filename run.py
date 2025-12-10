"""
========================================
量化交易系统 - 一键启动脚本
========================================

📌 这是最简单的启动方式！
    - 新手推荐：直接运行此文件即可
    - 系统会引导你完成所有操作
    - 无需分别运行多个脚本

📌 使用方法：
    python run.py

📌 功能菜单：
    1. 下载股票数据
    2. 训练模型并回测
    3. 每日预测
    4. 全流程运行（下载→训练→预测）
    5. 检查配置
    6. 退出
"""

import os
import sys
from datetime import datetime


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*70)
    print("      📊 量化交易系统 - 基于机器学习的股票预测      ")
    print("="*70)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_menu():
    """打印功能菜单"""
    print("\n" + "="*70)
    print("请选择操作：")
    print("="*70)
    print("  1. 📥 下载股票数据（首次使用必须执行）")
    print("  2. 🤖 训练模型并回测（验证策略效果）")
    print("  3. 📈 每日预测（生成今日推荐股票）")
    print("  4. 🚀 全流程运行（自动执行1→2→3）")
    print("  5. ⚙️  检查配置（查看当前参数设置）")
    print("  6. 🔧 修改配置（打开配置文件）")
    print("  7. ❓ 帮助文档")
    print("  8. ❌ 退出")
    print("="*70)


def check_config():
    """检查配置文件"""
    try:
        import config
        print("\n正在检查配置...")
        if config.validate_config():
            config.print_config_summary()
            return True
        return False
    except ImportError:
        print("\n❌ 找不到 config.py 配置文件！")
        print("请确保 config.py 文件在当前目录下")
        return False
    except Exception as e:
        print(f"\n❌ 配置检查失败: {e}")
        return False


def run_download():
    """运行数据下载脚本"""
    print("\n" + "="*70)
    print("📥 开始下载股票数据")
    print("="*70)
    print("提示: 首次下载可能需要10-30分钟，请耐心等待...")
    print("="*70 + "\n")

    confirm = input("是否继续? (y/n，默认y): ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消下载")
        return

    try:
        import download_stock_data_baostock
        # 这里不直接调用main()，而是提示用户手动运行
        # 因为download脚本有交互式输入
        print("\n即将启动下载脚本（请在新窗口中完成交互）...\n")
        os.system("python download_stock_data_baostock.py")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("请手动运行: python download_stock_data_baostock.py")


def run_backtest():
    """运行回测脚本"""
    print("\n" + "="*70)
    print("🤖 开始训练模型并回测")
    print("="*70)
    print("提示: 训练和回测可能需要5-15分钟...")
    print("="*70 + "\n")

    # 检查是否有数据
    if not os.path.exists('stock_data'):
        print("❌ 找不到 stock_data 目录！")
        print("请先运行选项1下载数据")
        return

    confirm = input("是否继续? (y/n，默认y): ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消回测")
        return

    try:
        import final_strategy
        print("\n正在运行回测...\n")
        final_strategy.main()
        print("\n✅ 回测完成！")
        print(f"   - 模型已保存到: rf_model.pkl")
        print(f"   - 可视化图表: backtest_visualization.png")
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()


def run_prediction():
    """运行每日预测脚本"""
    print("\n" + "="*70)
    print("📈 开始每日预测")
    print("="*70)
    print("提示: 预测通常需要2-5分钟...")
    print("="*70 + "\n")

    # 检查是否有模型
    if not os.path.exists('rf_model.pkl'):
        print("❌ 找不到模型文件 rf_model.pkl！")
        print("请先运行选项2训练模型")
        return

    confirm = input("是否继续? (y/n，默认y): ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消预测")
        return

    try:
        import predict_today
        print("\n正在运行预测...\n")
        predict_today.predict_today()
        print("\n✅ 预测完成！")
        print(f"   - Excel文件已生成，请查看当前目录")
    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()


def run_full_pipeline():
    """运行完整流程"""
    print("\n" + "="*70)
    print("🚀 全流程运行")
    print("="*70)
    print("将依次执行:")
    print("  Step 1: 下载股票数据")
    print("  Step 2: 训练模型并回测")
    print("  Step 3: 每日预测")
    print("\n⚠️  注意: 全流程可能需要30-60分钟，请确保网络稳定")
    print("="*70 + "\n")

    confirm = input("是否继续? (y/n，默认n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # Step 1: 下载数据
    print("\n" + "="*70)
    print("Step 1/3: 下载股票数据")
    print("="*70)
    run_download()

    # Step 2: 训练和回测
    print("\n" + "="*70)
    print("Step 2/3: 训练模型并回测")
    print("="*70)
    input("\n按回车键继续...")
    run_backtest()

    # Step 3: 预测
    print("\n" + "="*70)
    print("Step 3/3: 每日预测")
    print("="*70)
    input("\n按回车键继续...")
    run_prediction()

    print("\n" + "="*70)
    print("🎉 全流程完成！")
    print("="*70)


def open_config():
    """打开配置文件"""
    print("\n正在打开配置文件 config.py...")

    if sys.platform == 'win32':
        os.system('notepad config.py')
    elif sys.platform == 'darwin':  # macOS
        os.system('open -a TextEdit config.py')
    else:  # Linux
        os.system('gedit config.py || nano config.py')


def show_help():
    """显示帮助文档"""
    help_text = """
    ========================================
    量化交易系统 - 帮助文档
    ========================================

    📌 系统介绍：
        - 基于随机森林机器学习算法预测股票上涨概率
        - 支持沪深300、中证100等股票池
        - 自动回测验证策略效果
        - 每日生成推荐股票列表

    📌 文件说明：
        config.py                      - 配置文件（重要！所有参数在这里修改）
        run.py                         - 一键启动脚本（本文件）
        download_stock_data_baostock.py - 数据下载脚本
        final_strategy.py              - 模型训练和回测脚本
        predict_today.py               - 每日预测脚本
        hs300_stocks.py                - 沪深300股票池定义

    📌 使用流程：
        首次使用:
            1. 修改 config.py 设置参数（可选，默认参数已优化）
            2. 运行 run.py 选择"全流程运行"
            3. 等待完成，查看回测结果和预测Excel

        日常使用:
            - 每天运行 run.py 选择"每日预测"
            - 查看生成的Excel文件
            - 根据概率高低选择交易标的

    📌 配置说明：
        重要参数（在config.py中修改）：
            - STOCK_POOL_TYPE: 股票池类型（'hs300'/'zz100'/'custom'）
            - BUY_THRESHOLD: 买入阈值，推荐0.60（60%概率）
            - SELL_THRESHOLD: 卖出阈值，推荐0.50（50%概率）
            - INITIAL_CASH: 初始资金，默认100,000元

    📌 常见问题：
        Q1: 下载数据很慢怎么办？
        A1: 正常现象，首次下载300只股票需要20-30分钟

        Q2: 如何提高预测准确率？
        A2: 增加训练数据量（延长TRAIN_START_DATE）
            提高买入阈值（BUY_THRESHOLD）

        Q3: 回测收益率低怎么办？
        A3: 调整买入/卖出阈值，或更换股票池

        Q4: 如何添加自己的股票？
        A4: 在config.py中设置STOCK_POOL_TYPE='custom'
            然后填写CUSTOM_STOCK_POOL字典

        Q5: 可以用于实盘交易吗？
        A5: 本系统仅供学习研究，实盘需谨慎
            建议先小资金测试，观察一段时间

    📌 技术支持：
        - 遇到问题请检查config.py配置
        - 查看错误提示信息
        - 确保网络连接正常

    📌 风险提示：
        ⚠️  股市有风险，投资需谨慎
        ⚠️  历史回测不代表未来收益
        ⚠️  模型预测仅供参考，不构成投资建议

    ========================================
    """
    print(help_text)
    input("\n按回车键返回菜单...")


def main():
    """主函数"""
    print_banner()

    # 首次运行检查
    if not os.path.exists('config.py'):
        print("❌ 找不到配置文件 config.py！")
        print("请确保所有文件都在同一目录下")
        return

    print("欢迎使用量化交易系统！")
    print("\n💡 提示: 首次使用请先运行选项5检查配置，然后运行选项4全流程")

    while True:
        print_menu()

        choice = input("\n请输入选项编号 (1-8): ").strip()

        if choice == '1':
            run_download()
        elif choice == '2':
            run_backtest()
        elif choice == '3':
            run_prediction()
        elif choice == '4':
            run_full_pipeline()
        elif choice == '5':
            check_config()
        elif choice == '6':
            open_config()
        elif choice == '7':
            show_help()
        elif choice == '8':
            print("\n👋 感谢使用，再见！\n")
            break
        else:
            print("\n❌ 无效选项，请输入1-8")

        input("\n按回车键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！\n")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
