# 聚類分析專案使用指南

## 環境設置

請確保您的系統已安裝以下軟件：

1. Python 3.10 或更高版本
2. 必要的Python庫：
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. 如果要生成PDF報告，還需要安裝Pandoc和LaTeX：
   - Pandoc: https://pandoc.org/installing.html
   - LaTeX (如XeTeX): https://www.latex-project.org/get/

## 文件說明

- `main.py`: 主要的聚類分析程式
- `eval.py`: 評估聚類結果的程式
- `convert_to_pdf.py`: 將Markdown報告轉換為PDF的程式
- `run.sh`: 一鍵運行整個流程的腳本
- `r119020XX_report.md`: 聚類分析報告（Markdown格式）

## 運行方式

### 方法一：使用腳本一鍵運行

```bash
chmod +x run.sh  # 給予執行權限
./run.sh
```

### 方法二：逐步運行

1. 運行聚類分析：
   ```
   python main.py
   ```

2. 評估結果：
   ```
   python eval.py
   ```

3. 生成PDF報告（可選）：
   ```
   python convert_to_pdf.py
   ```

## 輸出文件

成功運行後，將生成以下文件：

- `public_submission.csv`: 公共數據集的聚類結果
- `private_submission.csv`: 私有數據集的聚類結果
- `r119020XX_report.pdf`: 聚類分析報告（PDF格式）
- `pca_visualization.png`: 原始數據的PCA可視化
- `best_clustering.png`: 最佳聚類結果的可視化

## 疑難排解

1. 如果遇到缺少庫的錯誤，請使用pip安裝對應的庫。
2. 如果PDF轉換失敗，請確保已正確安裝Pandoc和LaTeX，或者直接查看Markdown格式的報告。
3. 如果運行時間過長，可以在`main.py`中減少嘗試的參數範圍。 