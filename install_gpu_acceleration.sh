#!/bin/bash
# GPU加速環境安裝腳本

echo "🚀 安裝GPU加速機器學習環境"

# 檢查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 檢測到NVIDIA GPU"
    nvidia-smi
    
    # 安裝cuML (RAPIDS)
    echo "📦 安裝cuML..."
    conda install -c rapidsai -c nvidia -c conda-forge cuml=24.02 python=3.10 cudatoolkit=11.8 -y
    
    # 安裝cuDF
    echo "📦 安裝cuDF..."
    conda install -c rapidsai -c nvidia -c conda-forge cudf=24.02 python=3.10 cudatoolkit=11.8 -y
    
else
    echo "⚠️  未檢測到NVIDIA GPU，安裝CPU優化版本"
    
    # 安裝Intel優化版本
    pip install scikit-learn-intelex
    
    # 安裝多核優化庫
    pip install joblib
    pip install threadpoolctl
fi

# 安裝其他加速庫
echo "📦 安裝其他優化庫..."
pip install numba
pip install fastcluster  # 快速層次聚類
pip install faiss-cpu    # 快速相似性搜索

echo "✅ 安裝完成！"

# 測試安裝
python -c "
try:
    import cuml
    print('✓ cuML 可用')
except:
    print('✗ cuML 不可用')

try:
    import numba
    print('✓ Numba 可用')
except:
    print('✗ Numba 不可用')

try:
    import faiss
    print('✓ Faiss 可用')
except:
    print('✗ Faiss 不可用')
" 