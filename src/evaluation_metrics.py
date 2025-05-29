# src/evaluation_metrics.py
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_clustering_nmi(labels_true, labels_pred):
    """
    使用標準化互信息 (NMI) 評估聚類結果。
    注意：labels_true 通常是真實標籤，如果沒有，可以用於比較不同聚類算法的結果。
    """
    if len(set(labels_true)) < 2 or len(set(labels_pred)) < 2 :
        print("警告: NMI 計算至少需要兩個簇。一個或多個標籤集只有一個簇。")
        return np.nan # 或者返回一個特定的值，如0或-1
    nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
    print(f"標準化互信息 (NMI): {nmi:.4f}")
    return nmi

def evaluate_clustering_internal(data, labels):
    """
    使用內部指標評估聚類結果：輪廓係數和 Davies-Bouldin 指數。
    """
    if len(set(labels)) < 2 : # Silhouette 和 Davies-Bouldin 需要至少2個簇
        print("警告: 輪廓係數和 Davies-Bouldin 指數計算至少需要兩個簇。")
        s_score = np.nan
        db_score = np.nan
    else:
        s_score = silhouette_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
    
    print(f"輪廓係數 (Silhouette Score): {s_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    return s_score, db_score

def evaluate_classification(y_true, y_pred, y_proba=None, average='weighted', n_classes=None):
    """
    評估分類模型性能。
    """
    accuracy = accuracy_score(y_true, y_pred)
    # 如果只有一個類別被預測或真實存在，precision/recall/f1 可能會報錯或無意義
    # 設置 zero_division=0 來處理這種情況
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    print(f"\n分類評估 ({average} average):")
    print(f"  準確率 (Accuracy): {accuracy:.4f}")
    print(f"  精確率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1 分數 (F1-Score): {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("  混淆矩陣 (Confusion Matrix):\n", cm)

    auc_score_val = None
    if y_proba is not None and n_classes is not None:
        if n_classes == 2: # 二分類
            try:
                auc_score_val = roc_auc_score(y_true, y_proba[:, 1])
                print(f"  AUC: {auc_score_val:.4f}")
            except ValueError as e:
                print(f"  計算 AUC 時出錯: {e} (可能只有一個類別出現在 y_true 中)")
        elif n_classes > 2: # 多分類
            try:
                auc_score_val = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
                print(f"  AUC (One-vs-Rest, {average} average): {auc_score_val:.4f}")
            except ValueError as e:
                print(f"  計算 AUC 時出錯: {e} (可能只有一個類別出現在 y_true 中)")
                
    return accuracy, precision, recall, f1, cm, auc_score_val

if __name__ == '__main__':
    # 示例聚類評估
    labels_true_sample = np.array()
    labels_pred_sample_1 = np.array() # 模擬 K-Means 結果
    labels_pred_sample_2 = np.array() # 模擬 Agglomerative 結果
    sample_cluster_data = np.random.rand(7, 3)

    print("聚類評估 (NMI - 比較兩種算法結果):")
    evaluate_clustering_nmi(labels_pred_sample_1, labels_pred_sample_2)
    
    print("\n聚類評估 (內部指標 - K-Means):")
    evaluate_clustering_internal(sample_cluster_data, labels_pred_sample_1)

    # 示例分類評估
    y_true_clf = np.array()
    y_pred_clf = np.array()
    # 假設的概率輸出 (對於多分類, shape 是 n_samples, n_classes)
    # 確保概率總和為1 (這裡只是示例，實際模型會生成)
    y_proba_clf = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2], 
                            [0.9, 0.05, 0.05], [0.1, 0.3, 0.6], [0.05, 0.15, 0.8],
                            [0.7, 0.2, 0.1], [0.4, 0.5, 0.1], [0.1, 0.7, 0.2]])


    evaluate_classification(y_true_clf, y_pred_clf, y_proba=y_proba_clf, n_classes=3)