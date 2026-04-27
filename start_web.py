import sys
import subprocess


def ignite_server():
    print("🚀 正在通过底层 sys.executable 强行注入环境...")

    # 1. 强制使用当前完美的 Python 环境安装 Streamlit
    print("⏳ 正在安装 Streamlit (清华源加速)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "streamlit", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

    # 2. 强制使用当前环境启动 app.py 网页服务器
    print("🔥 点火！启动 Web 服务器...")
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    ignite_server()