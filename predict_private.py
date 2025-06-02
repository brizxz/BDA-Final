#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大數據分析期末專案 - Private Dataset 預測
使用在公開數據集上訓練的最佳模型對私有數據集進行預測
"""

import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# 基礎聚類算法
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

# 進階聚類算法
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("警告: HDBSCAN未安裝，將跳過HDBSCAN算法")

# 數據預處理
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class PrivateDatasetPredictor:
    """Private Dataset 預測器"""
    
    def __init__(self):
        self.public_data = None
        self.private_data = None
        self.features_public = None
        self.features_private = None
        self.feature_columns = None
        self.scaler = None
        self.trained_models = {}
        self.best_model_info = None
        
    def load_public_data(self):
        """載入公開數據集"""
        try:
            self.public_data = pd.read_csv("data/public_data.csv")
            print(f"成功載入公開數據: {self.public_data.shape}")
            return True
        except FileNotFoundError:
            print("錯誤: 找不到public_data.csv檔案")
            return False
    
    def load_private_data(self):
        """載入私有數據集"""
        try:
            self.private_data = pd.read_csv("data/private_data.csv")
            print(f"成功載入私有數據: {self.private_data.shape}")
            return True
        except FileNotFoundError:
            print("錯誤: 找不到private_data.csv檔案")
            return False
    
    def preprocess_data(self, scaling_method='standard'):
        """數據預處理 - 在公開數據上訓練scaler，然後應用到私有數據"""
        print(f"開始數據預處理 (縮放方法: {scaling_method})...")
        
        # 提取特徵（除了id列）
        self.feature_columns = [col for col in self.public_data.columns if col != 'id']
        
        # 公開數據特徵
        features_public_raw = self.public_data[self.feature_columns].values
        
        # 私有數據特徵
        features_private_raw = self.private_data[self.feature_columns].values
        
        # 選擇縮放方法 - 在公開數據上fit
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("縮放方法必須是 'standard' 或 'minmax'")
        
        # fit在公開數據上，transform兩個數據集
        self.features_public = self.scaler.fit_transform(features_public_raw)
        self.features_private = self.scaler.transform(features_private_raw)
        
        print(f"公開數據特徵維度: {self.features_public.shape}")
        print(f"私有數據特徵維度: {self.features_private.shape}")
        print(f"特徵列: {self.feature_columns}")
        
        return True
    
    def train_all_models(self):
        """在公開數據上訓練所有聚類模型"""
        print("\n在公開數據上訓練所有聚類模型...")
        self.trained_models = {}
        
        # 1. K-Means
        print("訓練 K-Means...")
        best_k = self._find_optimal_k_kmeans()
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(self.features_public)
        
        self.trained_models['kmeans'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'params': {'n_clusters': best_k},
            'metrics': self._evaluate_clustering(kmeans_labels, self.features_public)
        }
        
        # 2. DBSCAN
        print("訓練 DBSCAN...")
        best_eps, best_min_samples = self._find_optimal_dbscan_params()
        dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        dbscan_labels = dbscan.fit_predict(self.features_public)
        
        self.trained_models['dbscan'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            'params': {'eps': best_eps, 'min_samples': best_min_samples},
            'metrics': self._evaluate_clustering(dbscan_labels, self.features_public)
        }
        
        # 3. 層次聚類
        print("訓練 層次聚類...")
        best_k_agg = self._find_optimal_k_agglomerative()
        agglomerative = AgglomerativeClustering(n_clusters=best_k_agg)
        agg_labels = agglomerative.fit_predict(self.features_public)
        
        self.trained_models['agglomerative'] = {
            'model': agglomerative,
            'labels': agg_labels,
            'params': {'n_clusters': best_k_agg},
            'metrics': self._evaluate_clustering(agg_labels, self.features_public)
        }
        
        # 4. GMM
        print("訓練 GMM...")
        best_k_gmm = self._find_optimal_k_gmm()
        gmm = GaussianMixture(n_components=best_k_gmm, random_state=42)
        gmm_labels = gmm.fit_predict(self.features_public)
        
        self.trained_models['gmm'] = {
            'model': gmm,
            'labels': gmm_labels,
            'params': {'n_components': best_k_gmm},
            'metrics': self._evaluate_clustering(gmm_labels, self.features_public)
        }
        
        # 5. 譜聚類
        print("訓練 譜聚類...")
        best_k_spectral = self._find_optimal_k_spectral()
        spectral = SpectralClustering(n_clusters=best_k_spectral, random_state=42)
        spectral_labels = spectral.fit_predict(self.features_public)
        
        self.trained_models['spectral'] = {
            'model': spectral,
            'labels': spectral_labels,
            'params': {'n_clusters': best_k_spectral},
            'metrics': self._evaluate_clustering(spectral_labels, self.features_public)
        }
        
        # 6. HDBSCAN (如果可用)
        if HDBSCAN_AVAILABLE:
            print("訓練 HDBSCAN...")
            best_min_cluster_size = self._find_optimal_hdbscan_params()
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=best_min_cluster_size)
            hdbscan_labels = hdbscan_model.fit_predict(self.features_public)
            
            self.trained_models['hdbscan'] = {
                'model': hdbscan_model,
                'labels': hdbscan_labels,
                'params': {'min_cluster_size': best_min_cluster_size},
                'metrics': self._evaluate_clustering(hdbscan_labels, self.features_public)
            }
        
        print("所有模型訓練完成！")
        
    def _find_optimal_k_kmeans(self, max_k=15):
        """找到K-Means的最佳K值"""
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_public)
            
            try:
                score = silhouette_score(self.features_public, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
                
        print(f"K-Means 最佳K值: {best_k} (Silhouette: {best_score:.4f})")
        return best_k
    
    def _find_optimal_dbscan_params(self):
        """找到DBSCAN的最佳參數"""
        eps_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        min_samples_values = [3, 5, 10, 15]
        
        best_eps = 0.5
        best_min_samples = 5
        best_score = -1
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.features_public)
                
                if len(set(labels)) > 1 and -1 not in labels:
                    try:
                        score = silhouette_score(self.features_public, labels)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                    except:
                        continue
        
        print(f"DBSCAN 最佳參數: eps={best_eps}, min_samples={best_min_samples} (Silhouette: {best_score:.4f})")
        return best_eps, best_min_samples
    
    def _find_optimal_k_agglomerative(self, max_k=15):
        """找到層次聚類的最佳K值"""
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(self.features_public)
            
            try:
                score = silhouette_score(self.features_public, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
                
        print(f"層次聚類 最佳K值: {best_k} (Silhouette: {best_score:.4f})")
        return best_k
    
    def _find_optimal_k_gmm(self, max_k=15):
        """找到GMM的最佳K值"""
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(self.features_public)
            
            try:
                score = silhouette_score(self.features_public, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
                
        print(f"GMM 最佳K值: {best_k} (Silhouette: {best_score:.4f})")
        return best_k
    
    def _find_optimal_k_spectral(self, max_k=10):
        """找到譜聚類的最佳K值"""
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            spectral = SpectralClustering(n_clusters=k, random_state=42)
            labels = spectral.fit_predict(self.features_public)
            
            try:
                score = silhouette_score(self.features_public, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
                
        print(f"譜聚類 最佳K值: {best_k} (Silhouette: {best_score:.4f})")
        return best_k
    
    def _find_optimal_hdbscan_params(self):
        """找到HDBSCAN的最佳參數"""
        min_cluster_sizes = [5, 10, 15, 20, 30]
        
        best_min_cluster_size = 10
        best_score = -1
        
        for min_cluster_size in min_cluster_sizes:
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = hdbscan_model.fit_predict(self.features_public)
            
            if len(set(labels)) > 1:
                valid_mask = labels != -1
                if sum(valid_mask) > 0:
                    try:
                        score = silhouette_score(self.features_public[valid_mask], labels[valid_mask])
                        if score > best_score:
                            best_score = score
                            best_min_cluster_size = min_cluster_size
                    except:
                        continue
        
        print(f"HDBSCAN 最佳參數: min_cluster_size={best_min_cluster_size} (Silhouette: {best_score:.4f})")
        return best_min_cluster_size
    
    def _evaluate_clustering(self, labels, features):
        """評估聚類結果"""
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_features = features[valid_mask]
        
        if len(set(valid_labels)) < 2:
            return {
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': len(set(valid_labels)),
                'n_noise': sum(labels == -1)
            }
        
        try:
            return {
                'silhouette_score': silhouette_score(valid_features, valid_labels),
                'calinski_harabasz_score': calinski_harabasz_score(valid_features, valid_labels),
                'davies_bouldin_score': davies_bouldin_score(valid_features, valid_labels),
                'n_clusters': len(set(valid_labels)),
                'n_noise': sum(labels == -1)
            }
        except:
            return {
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': len(set(valid_labels)),
                'n_noise': sum(labels == -1)
            }
    
    def select_best_model(self):
        """選擇最佳模型"""
        print("\n選擇最佳模型...")
        
        best_method = None
        best_score = -1
        
        for method, model_info in self.trained_models.items():
            metrics = model_info['metrics']
            score = metrics['silhouette_score']
            
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_method = method
        
        if best_method:
            self.best_model_info = {
                'method': best_method,
                'model': self.trained_models[best_method]['model'],
                'params': self.trained_models[best_method]['params'],
                'metrics': self.trained_models[best_method]['metrics']
            }
            
            print(f"最佳模型: {best_method.upper()}")
            print(f"Silhouette Score: {best_score:.4f}")
            print(f"聚類數: {self.trained_models[best_method]['metrics']['n_clusters']}")
            
            return best_method
        else:
            print("無法找到有效的最佳模型")
            return None
    
    def predict_private_data(self):
        """對私有數據進行預測"""
        if self.best_model_info is None:
            print("錯誤: 尚未選擇最佳模型")
            return None
        
        print(f"\n使用 {self.best_model_info['method'].upper()} 對私有數據進行預測...")
        
        method = self.best_model_info['method']
        model = self.best_model_info['model']
        
        try:
            if method == 'kmeans':
                private_labels = model.predict(self.features_private)
            elif method == 'gmm':
                private_labels = model.predict(self.features_private)
            elif method == 'dbscan':
                # DBSCAN需要重新fit整個數據集
                combined_features = np.vstack([self.features_public, self.features_private])
                combined_labels = model.fit_predict(combined_features)
                private_labels = combined_labels[len(self.features_public):]
            elif method == 'agglomerative':
                # 層次聚類需要重新fit整個數據集
                combined_features = np.vstack([self.features_public, self.features_private])
                combined_labels = model.fit_predict(combined_features)
                private_labels = combined_labels[len(self.features_public):]
            elif method == 'spectral':
                # 譜聚類需要重新fit整個數據集
                combined_features = np.vstack([self.features_public, self.features_private])
                combined_labels = model.fit_predict(combined_features)
                private_labels = combined_labels[len(self.features_public):]
            elif method == 'hdbscan':
                # HDBSCAN需要重新fit整個數據集
                combined_features = np.vstack([self.features_public, self.features_private])
                combined_labels = model.fit_predict(combined_features)
                private_labels = combined_labels[len(self.features_public):]
            else:
                print(f"不支援的模型類型: {method}")
                return None
            
            print(f"私有數據預測完成！")
            print(f"私有數據聚類分佈:")
            unique_labels, counts = np.unique(private_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                percentage = count / len(private_labels) * 100
                print(f"聚類 {label}: {count} 樣本 ({percentage:.1f}%)")
            
            return private_labels
            
        except Exception as e:
            print(f"預測過程中發生錯誤: {e}")
            return None
    
    def create_private_submission(self, private_labels, filename="private_submission.csv"):
        """創建私有數據集的提交檔案"""
        if private_labels is None:
            print("錯誤: 沒有私有數據預測結果")
            return None
        
        print(f"\n創建私有數據提交檔案: {filename}")
        
        submission = pd.DataFrame({
            'id': self.private_data['id'],
            'label': private_labels
        })
        
        # 確保標籤從0開始且連續
        unique_labels = sorted(set(private_labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        submission['label'] = submission['label'].map(label_mapping)
        
        submission.to_csv(filename, index=False)
        
        print(f"私有數據提交檔案已保存: {filename}")
        print(f"總計 {len(submission)} 筆記錄")
        print(f"聚類標籤分佈:")
        print(submission['label'].value_counts().sort_index())
        
        return submission
    
    def generate_comparison_report(self):
        """生成模型比較報告"""
        print("\n生成模型比較報告...")
        
        report_filename = "private_prediction_report.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("大數據分析期末專案 - Private Dataset 預測報告\n")
            f.write("="*70 + "\n\n")
            
            # 數據概述
            f.write("1. 數據概述\n")
            f.write("-" * 20 + "\n")
            f.write(f"公開數據集大小: {self.public_data.shape[0]} 樣本, {len(self.feature_columns)} 特徵\n")
            f.write(f"私有數據集大小: {self.private_data.shape[0]} 樣本, {len(self.feature_columns)} 特徵\n")
            f.write(f"特徵列: {', '.join(self.feature_columns)}\n")
            f.write(f"縮放方法: {type(self.scaler).__name__}\n\n")
            
            # 模型比較
            f.write("2. 在公開數據上的模型性能比較\n")
            f.write("-" * 40 + "\n")
            for method, model_info in self.trained_models.items():
                metrics = model_info['metrics']
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  - 聚類數: {metrics['n_clusters']}\n")
                f.write(f"  - 噪聲點: {metrics.get('n_noise', 0)}\n")
                f.write(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}\n")
                f.write(f"  - Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 0):.2f}\n")
                f.write(f"  - Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.4f}\n")
            
            # 最佳模型
            if self.best_model_info:
                f.write(f"\n3. 選定的最佳模型\n")
                f.write("-" * 20 + "\n")
                f.write(f"模型: {self.best_model_info['method'].upper()}\n")
                f.write(f"參數: {self.best_model_info['params']}\n")
                f.write(f"Silhouette Score: {self.best_model_info['metrics']['silhouette_score']:.4f}\n")
                f.write(f"聚類數: {self.best_model_info['metrics']['n_clusters']}\n")
        
        print(f"比較報告已保存: {report_filename}")

def main():
    """主函數"""
    print("="*70)
    print("大數據分析期末專案 - Private Dataset 預測")
    print("="*70)
    
    # 1. 初始化預測器
    predictor = PrivateDatasetPredictor()
    
    # 2. 載入數據
    print("\n載入數據...")
    if not predictor.load_public_data():
        return
    if not predictor.load_private_data():
        return
    
    # 3. 數據預處理
    print("\n數據預處理...")
    if not predictor.preprocess_data():
        return
    
    # 4. 在公開數據上訓練所有模型
    predictor.train_all_models()
    
    # 5. 選擇最佳模型
    best_method = predictor.select_best_model()
    
    if best_method is None:
        print("無法找到有效的模型，程式結束")
        return
    
    # 6. 對私有數據進行預測
    private_labels = predictor.predict_private_data()
    
    if private_labels is None:
        print("私有數據預測失敗，程式結束")
        return
    
    # 7. 創建提交檔案
    submission = predictor.create_private_submission(private_labels)
    
    # 8. 生成報告
    predictor.generate_comparison_report()
    
    # 9. 輸出總結
    print("\n" + "="*70)
    print("Private Dataset 預測完成！")
    print("="*70)
    print(f"使用模型: {best_method.upper()}")
    print(f"私有數據樣本數: {len(private_labels)}")
    print(f"聚類數量: {len(set(private_labels))}")
    print("\n生成的檔案:")
    print("- private_submission.csv (私有數據預測結果)")
    print("- private_prediction_report.txt (預測報告)")
    
    return submission

if __name__ == "__main__":
    submission = main() 