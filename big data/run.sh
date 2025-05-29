#!/bin/bash

echo "===== 開始執行聚類分析 ====="
python main.py

echo "===== 評估結果 ====="
python eval.py

echo "===== 生成報告 ====="
python convert_to_pdf.py

echo "===== 完成 ====="
echo "生成的文件:"
echo "- public_submission.csv"
echo "- private_submission.csv"
echo "- r119020XX_report.pdf"
echo "- pca_visualization.png"
echo "- best_clustering.png" 