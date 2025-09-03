#!/bin/bash

#==================================================
# 脚本功能：全自动安装 Miniconda，配置国内镜像，创建环境，安装 faiss-gpu 并验证 GPU 支持
# 支持系统：Linux / WSL
# 支持 Shell：bash / fish
# 特性：自动处理网络问题、ToS、conda 激活、镜像源
#==================================================

set -euo pipefail

# -------------------------------
# 配置变量
# -------------------------------
export MINICONDA_DIR="$HOME/miniconda3"
export CONDA_EXE="$MINICONDA_DIR/bin/conda"
export MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
export MINICONDA_SH="/tmp/miniconda.sh"
export ENV_NAME="venv"
export PYTHON_VERSION="3.11"

# 只保留成功的镜像源
export MIRROR_URL="https://mirrors.ustc.edu.cn"

# -------------------------------
# 1. 检测当前 shell 类型
# -------------------------------
detect_shell() {
    case "$SHELL" in
        */fish) echo "fish" ;;
        */zsh)  echo "zsh"  ;;
        *)      echo "bash" ;;
    esac
}

SHELL_TYPE=$(detect_shell)
echo "🔍 检测到当前 shell 类型: $SHELL_TYPE"

# -------------------------------
# 2. 安装 Miniconda（如未安装）
# -------------------------------
if [ ! -d "$MINICONDA_DIR" ]; then
    echo "📥 Miniconda 未安装，开始下载..."

    if command -v wget > /dev/null; then
        wget -qO "$MINICONDA_SH" "$MINICONDA_URL"
    elif command -v curl > /dev/null; then
        curl -o "$MINICONDA_SH" -L "$MINICONDA_URL"
    else
        echo "❌ 错误：系统中未找到 wget 或 curl" >&2
        exit 1
    fi

    if [ ! -s "$MINICONDA_SH" ]; then
        echo "❌ 错误：Miniconda 安装包为空或下载失败" >&2
        exit 1
    fi

    echo "📦 正在静默安装 Miniconda 到 $MINICONDA_DIR..."
    bash "$MINICONDA_SH" -b -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_SH"

    echo "✅ Miniconda 安装完成。"
else
    echo "✅ Miniconda 已存在：$MINICONDA_DIR"
fi

# -------------------------------
# 3. 确保 conda 命令可用
# -------------------------------
if ! command -v conda &> /dev/null; then
    echo "🔄 尝试手动加载 conda..."
    if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
        source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    else
        echo "❌ 错误：conda 可执行文件不存在：$CONDA_EXE" >&2
        exit 1
    fi
fi

# -------------------------------
# 4. 初始化当前 shell（支持 fish）
# -------------------------------
if [ "$SHELL_TYPE" = "fish" ]; then
    if ! grep -q "conda init fish" ~/.config/fish/config.fish 2>/dev/null; then
        echo "⚙️ 为 fish shell 初始化 conda..."
        conda init fish
        echo "💡 请重启终端以确保 conda 正常工作。"
    fi
fi

# -------------------------------
# 5. 接受 Anaconda 服务条款 (ToS)
# -------------------------------
echo "📝 正在接受 Anaconda 频道服务条款 (ToS)..."
for channel_url in "https://repo.anaconda.com/pkgs/main" "https://repo.anaconda.com/pkgs/r"; do
    echo "✅ 接受频道: $channel_url"
    conda tos accept --override-channels --channel "$channel_url" || true
done

# -------------------------------
# 6. 配置国内镜像源
# -------------------------------
echo "🌐 配置国内镜像源: $MIRROR_URL ..."
conda config --add channels "$MIRROR_URL/anaconda/cloud/pytorch/"
conda config --set show_channel_urls yes
conda config --set channel_priority strict

# 清理缓存以强制使用新源
echo "🧹 清理 conda 缓存..."
conda clean -i -y

# -------------------------------
# 7. 创建虚拟环境（如不存在）
# -------------------------------
if ! conda env list | grep -q "^$ENV_NAME\s"; then
    echo "📦 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
else
    echo "✅ 环境已存在: $ENV_NAME"
fi

# -------------------------------
# 8. 激活环境
# -------------------------------
echo "🔁 激活环境 $ENV_NAME..."
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || {
    echo "❌ 激活环境失败。"
    echo "💡 请确保 conda 已初始化。建议重启终端后重试。"
    exit 1
}

# -------------------------------
# 9. 安装 faiss-gpu
# -------------------------------
echo "📥 正在安装 faiss-gpu (来自 pytorch 频道)..."
conda install -y faiss-gpu || {
    echo "❌ faiss-gpu 安装失败！"
    echo "💡 这通常是网络问题或镜像源不稳定导致的。"
    exit 1
}

# -------------------------------
# 10. 验证安装与 GPU 支持
# -------------------------------
echo "🧪 正在验证 faiss 安装..."
python -c "
import faiss
print('✅ faiss 版本:', faiss.__version__)
print('✅ GPU 支持:', hasattr(faiss, 'StandardGpuResources'))
" || {
    echo "❌ faiss 安装可能不完整或 GPU 驱动未就绪"
    exit 1
}

# -------------------------------
# 11. 完成提示
# -------------------------------
echo ""
echo "🎉 所有步骤完成！"
echo ""
echo "💡 激活环境命令："
echo "   conda activate $ENV_NAME"
echo ""
echo "📌 如果你使用的是 fish shell，请确保在安装前已初始化过 conda 或重启终端。"