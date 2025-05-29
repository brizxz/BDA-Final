# Final-Project-Big-Data

## CSV Format
**public_submission.csv**
```
id,label
1,0
2,0
...
```


## How to grade?
1. Modify `public_submission.csv`.
2. Push your file to this repository (`git push`).
3. Wait for the outcome of git action.


## 環境設置

1.  確保已安裝 Python 3.7 或更高版本。
2.  創建並激活虛擬環境（推薦）：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  安裝所需的 Python 庫：
    ```bash
    pip install -r requirements.txt
    ```

## 如何運行

1.  將提供的 `public_data.csv` 文件放置在 `data/` 目錄下。
2.  如果有名為 `private_data.csv` 的私人數據集，也請放置在 `data/` 目錄下，並相應修改 `src/data_loader.py` 中的加載邏輯以及 `src/main.py` 中的調用參數。目前，私人數據集部分在代碼中是概念性的，會生成隨機數據進行流程演示。
3.  從專案根目錄運行主腳本：
    ```bash
    python src/main.py
    ```
    腳本將執行以下操作：
    *   加載並預處理公共數據集（4維）。
    *   對其進行聚類分析（目標簇數為 $4 \times 4 - 1 = 15$），並輸出評估指標。
    *   使用聚類結果作為偽標籤，進行分類模型訓練和評估。
    *   概念性地處理私人數據集（6維，目標簇數為 $4 \times 6 - 1 = 23$），包括可能的 PCA 降維。

## 注意事項

*   **DBSCAN 簇數：** DBSCAN 算法會自動確定簇的數量。腳本中的實現會嘗試根據數據維度設置 `min_samples`，但 `eps` 參數通常需要針對特定數據集進行仔細調優才能獲得期望數量的簇。精確控制 DBSCAN 產生 `4n-1` 個簇是具有挑戰性的。
*   **NMI 評估：** 在沒有真實標籤的情況下，NMI 主要用於比較不同聚類算法在同一數據集上的結果相似性。腳本中計算了 K-Means 和層次聚類結果之間的 NMI。對於單個聚類算法的內部評估，使用了輪廓係數和 Davies-Bouldin 指數。
*   **私人數據集：** `main.py` 中對私人數據集的處理目前是概念性的。若要使用真實的私人數據，需要修改 `src/data_loader.py` 中的 `load_private_data_placeholder` 函數，並確保 `main.py` 中調用 `run_analysis` 時傳遞正確的文件路徑和特徵列名。為了演示，當前 `main.py` 中的私人數據集流程會生成隨機6維數據。
*   **模型參數：** 各算法的參數（如DBSCAN的 `eps`，GMM的 `covariance_type` 等）在腳本中使用了常見的默認值或示例值。在實際應用中，這些參數可能需要根據數據特性進行調優以獲得最佳性能。