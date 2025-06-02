#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大數據分析期末專案 - 進階聚類分析
支援多種聚類演算法並使用多重評估指標
"""

import pandas as pd
import numpy as np
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

# 數據預處理與評估
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# 可視化
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("警告: matplotlib/seaborn未安裝，將跳過可視化")

class AdvancedClusteringAnalysis:
    """進階聚類分析類"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.feature_columns = None
        self.scaler = None
        self.results = {}
        
    def load_data(self):
        """載入數據"""
        try:
            # 嘗試從data目錄載入
            self.data = pd.read_csv("data/public_data.csv")
            print(f"成功載入公共數據: {self.data.shape}")
            return True
        except FileNotFoundError:
            try:
                # 嘗試從big data目錄載入
                self.data = pd.read_csv("big data/public_data.csv")
                print(f"成功載入公共數據: {self.data.shape}")
                return True
            except FileNotFoundError:
                print("錯誤: 找不到public_data.csv檔案")
                return False
    
    def preprocess_data(self, scaling_method='standard'):
        """數據預處理"""
        print(f"開始數據預處理 (縮放方法: {scaling_method})...")
        
        # 提取特徵（除了id列）
        self.feature_columns = [col for col in self.data.columns if col != 'id']
        features_raw = self.data[self.feature_columns].values
        
        # 選擇縮放方法
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("縮放方法必須是 'standard' 或 'minmax'")
        
        self.features = self.scaler.fit_transform(features_raw)
        
        print(f"特徵維度: {self.features.shape}")
        print(f"特徵列: {self.feature_columns}")
        
        return True
    
    def evaluate_clustering(self, labels, features, algorithm_name):
        """全面評估聚類結果"""
        # 過濾掉噪聲點（標籤為-1的點）
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_features = features[valid_mask]
        
        if len(set(valid_labels)) < 2:
            return {
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': len(set(valid_labels)),
                'n_noise': sum(labels == -1),
                'execution_time': 0
            }
        
        metrics = {
            'silhouette_score': silhouette_score(valid_features, valid_labels),
            'calinski_harabasz_score': calinski_harabasz_score(valid_features, valid_labels),
            'davies_bouldin_score': davies_bouldin_score(valid_features, valid_labels),
            'n_clusters': len(set(valid_labels)),
            'n_noise': sum(labels == -1),
        }
        
        return metrics
    
    def apply_kmeans_variants(self, max_k=15):
        """應用K-means及其變體"""
        print("\n=== K-Means 聚類分析 ===")
        
        best_k = 2
        best_score = -1
        k_scores = []
        
        # 找到最佳K值
        for k in range(2, max_k + 1):
            start_time = time.time()
            
            # 標準K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm='lloyd')
            labels = kmeans.fit_predict(self.features)
            
            execution_time = time.time() - start_time
            metrics = self.evaluate_clustering(labels, self.features, f"K-Means(k={k})")
            metrics['execution_time'] = execution_time
            
            silhouette = metrics['silhouette_score']
            k_scores.append((k, silhouette))
            
            if silhouette > best_score:
                best_score = silhouette
                best_k = k
            
            print(f"K={k:2d}: Silhouette={silhouette:.4f}, 執行時間={execution_time:.2f}s")
        
        # 使用最佳K值重新訓練
        start_time = time.time()
        best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        best_labels = best_kmeans.fit_predict(self.features)
        execution_time = time.time() - start_time
        
        metrics = self.evaluate_clustering(best_labels, self.features, "K-Means")
        metrics['execution_time'] = execution_time
        metrics['optimal_k'] = best_k
        
        self.results['kmeans'] = {
            'labels': best_labels,
            'model': best_kmeans,
            'metrics': metrics,
            'k_scores': k_scores
        }
        
        print(f"最佳K-Means: K={best_k}, Silhouette={best_score:.4f}")
        return best_labels, metrics
    
    def apply_dbscan_variants(self):
        """應用DBSCAN及其參數調優版本"""
        print("\n=== DBSCAN 聚類分析 ===")
        
        # 參數網格搜索
        eps_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        min_samples_values = [3, 5, 10, 15]
        
        best_score = -1
        best_params = None
        best_labels = None
        best_model = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                start_time = time.time()
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.features)
                
                execution_time = time.time() - start_time
                metrics = self.evaluate_clustering(labels, self.features, 
                                                 f"DBSCAN(eps={eps},min_samples={min_samples})")
                
                if not np.isnan(metrics['silhouette_score']) and metrics['n_clusters'] >= 2:
                    score = metrics['silhouette_score']
                    print(f"eps={eps:4.1f}, min_samples={min_samples:2d}: "
                          f"聚類數={metrics['n_clusters']:2d}, "
                          f"噪聲點={metrics['n_noise']:4d}, "
                          f"Silhouette={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                        best_labels = labels
                        best_model = dbscan
                        metrics['execution_time'] = execution_time
                        best_metrics = metrics
        
        if best_labels is not None:
            self.results['dbscan'] = {
                'labels': best_labels,
                'model': best_model,
                'metrics': best_metrics,
                'best_params': best_params
            }
            print(f"最佳DBSCAN: eps={best_params[0]}, min_samples={best_params[1]}, "
                  f"Silhouette={best_score:.4f}")
        else:
            print("DBSCAN: 未找到合適的參數組合")
            
        return best_labels, best_metrics if best_labels is not None else None
    
    def apply_hdbscan(self):
        """應用HDBSCAN算法"""
        if not HDBSCAN_AVAILABLE:
            print("\n=== HDBSCAN 未安裝，跳過 ===")
            return None, None
            
        print("\n=== HDBSCAN 聚類分析 ===")
        
        # 參數搜索
        min_cluster_sizes = [5, 10, 15, 20, 30]
        min_samples_values = [3, 5, 10]
        
        best_score = -1
        best_params = None
        best_labels = None
        best_model = None
        
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_values:
                start_time = time.time()
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.0
                )
                labels = clusterer.fit_predict(self.features)
                
                execution_time = time.time() - start_time
                metrics = self.evaluate_clustering(labels, self.features, 
                                                 f"HDBSCAN(min_cluster_size={min_cluster_size})")
                
                if not np.isnan(metrics['silhouette_score']) and metrics['n_clusters'] >= 2:
                    score = metrics['silhouette_score']
                    print(f"min_cluster_size={min_cluster_size:2d}, min_samples={min_samples:2d}: "
                          f"聚類數={metrics['n_clusters']:2d}, "
                          f"噪聲點={metrics['n_noise']:4d}, "
                          f"Silhouette={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = (min_cluster_size, min_samples)
                        best_labels = labels
                        best_model = clusterer
                        metrics['execution_time'] = execution_time
                        best_metrics = metrics
        
        if best_labels is not None:
            self.results['hdbscan'] = {
                'labels': best_labels,
                'model': best_model,
                'metrics': best_metrics,
                'best_params': best_params
            }
            print(f"最佳HDBSCAN: min_cluster_size={best_params[0]}, "
                  f"Silhouette={best_score:.4f}")
        else:
            print("HDBSCAN: 未找到合適的參數組合")
            
        return best_labels, best_metrics if best_labels is not None else None
    
    def apply_spectral_clustering(self, max_k=10):
        """應用譜聚類（針對大數據集優化）"""
        print("\n=== 譜聚類分析 ===")
        
        # 對於大數據集，限制樣本數量以避免記憶體問題
        if len(self.features) > 5000:
            print(f"數據集過大 ({len(self.features)} 樣本)，使用前5000個樣本進行譜聚類")
            sample_indices = np.random.choice(len(self.features), 5000, replace=False)
            sample_features = self.features[sample_indices]
            print("注意：譜聚類結果基於樣本數據，可能不完全代表整體數據特性")
        else:
            sample_features = self.features
            sample_indices = None
        
        best_k = 2
        best_score = -1
        best_labels = None
        best_model = None
        
        # 限制最大聚類數以避免計算問題
        max_k = min(max_k, 8, len(sample_features) // 100)
        
        for k in range(2, max_k + 1):
            start_time = time.time()
            
            try:
                # 使用更穩健的參數設置
                spectral = SpectralClustering(
                    n_clusters=k,
                    eigen_solver='arpack',
                    affinity='nearest_neighbors',  # 改用nearest_neighbors，對大數據集更穩定
                    n_neighbors=min(10, len(sample_features) // 10),
                    gamma=1.0,
                    random_state=42,
                    n_jobs=1  # 限制單線程執行
                )
                
                # 對樣本數據進行聚類
                sample_labels = spectral.fit_predict(sample_features)
                
                execution_time = time.time() - start_time
                
                # 如果使用了樣本，需要為全數據集預測標籤
                if sample_indices is not None:
                    # 使用KMeans對樣本聚類結果進行擴展到全數據集
                    from sklearn.cluster import KMeans
                    kmeans_extend = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_extend.fit(sample_features)
                    full_labels = kmeans_extend.predict(self.features)
                else:
                    full_labels = sample_labels
                
                metrics = self.evaluate_clustering(full_labels, self.features, f"Spectral(k={k})")
                metrics['execution_time'] = execution_time
                
                silhouette = metrics['silhouette_score']
                print(f"K={k:2d}: Silhouette={silhouette:.4f}, 執行時間={execution_time:.2f}s")
                
                if not np.isnan(silhouette) and silhouette > best_score:
                    best_score = silhouette
                    best_k = k
                    best_labels = full_labels
                    best_model = spectral
                    best_metrics = metrics
                    
            except Exception as e:
                print(f"K={k:2d}: 計算失敗 - {str(e)}")
                continue
        
        if best_labels is not None:
            self.results['spectral'] = {
                'labels': best_labels,
                'model': best_model,
                'metrics': best_metrics,
                'optimal_k': best_k
            }
            print(f"最佳譜聚類: K={best_k}, Silhouette={best_score:.4f}")
        else:
            print("譜聚類: 所有參數組合都失敗")
        
        return best_labels, best_metrics if best_labels is not None else None
    
    def apply_gaussian_mixture(self, max_k=15):
        """應用高斯混合模型"""
        print("\n=== 高斯混合模型聚類分析 ===")
        
        best_k = 2
        best_score = -1
        best_labels = None
        best_model = None
        
        for k in range(2, max_k + 1):
            start_time = time.time()
            
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(self.features)
                labels = gmm.predict(self.features)
                
                execution_time = time.time() - start_time
                metrics = self.evaluate_clustering(labels, self.features, f"GMM(k={k})")
                metrics['execution_time'] = execution_time
                metrics['aic'] = gmm.aic(self.features)
                metrics['bic'] = gmm.bic(self.features)
                
                silhouette = metrics['silhouette_score']
                print(f"K={k:2d}: Silhouette={silhouette:.4f}, "
                      f"AIC={metrics['aic']:.1f}, BIC={metrics['bic']:.1f}, "
                      f"執行時間={execution_time:.2f}s")
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_k = k
                    best_labels = labels
                    best_model = gmm
                    best_metrics = metrics
                    
            except Exception as e:
                print(f"K={k:2d}: 計算失敗 - {str(e)}")
                continue
        
        if best_labels is not None:
            self.results['gmm'] = {
                'labels': best_labels,
                'model': best_model,
                'metrics': best_metrics,
                'optimal_k': best_k
            }
            print(f"最佳GMM: K={best_k}, Silhouette={best_score:.4f}")
        
        return best_labels, best_metrics if best_labels is not None else None
    
    def apply_agglomerative_clustering(self, max_k=15):
        """應用層次聚類 - 改進版本"""
        print("\n=== 層次聚類分析 ===")
        
        linkage_methods = ['ward', 'complete', 'average', 'single']
        best_score = -1
        best_params = None
        best_labels = None
        best_model = None
        
        # 對於大數據集，考慮使用樣本
        sample_size = min(10000, len(self.features))
        if len(self.features) > sample_size:
            print(f"數據集較大，使用 {sample_size} 個樣本進行參數調優")
            sample_indices = np.random.choice(len(self.features), sample_size, replace=False)
            sample_features = self.features[sample_indices]
        else:
            sample_features = self.features
            sample_indices = None
        
        for linkage in linkage_methods:
            print(f"\n{linkage.upper()} 連接方法:")
            for k in range(2, max_k + 1):
                start_time = time.time()
                
                try:
                    # 先用樣本數據找最佳參數
                    agg_sample = AgglomerativeClustering(
                        n_clusters=k,
                        linkage=linkage
                    )
                    sample_labels = agg_sample.fit_predict(sample_features)
                    
                    # 檢查樣本結果是否合理
                    unique_labels = np.unique(sample_labels)
                    if len(unique_labels) < 2:
                        print(f"  K={k:2d}: 樣本聚類失敗（只有一個聚類）")
                        continue
                    
                    # 檢查是否有極度不平衡的聚類
                    sample_counts = np.bincount(sample_labels)
                    max_ratio = sample_counts.max() / sample_counts.min()
                    if max_ratio > 1000:  # 如果比例超過1000，跳過
                        print(f"  K={k:2d}: 樣本結果不平衡（比例 {max_ratio:.1f}），跳過")
                        continue
                    
                    # 如果樣本結果合理，對全部數據進行聚類
                    agg = AgglomerativeClustering(
                        n_clusters=k,
                        linkage=linkage
                    )
                    labels = agg.fit_predict(self.features)
                    
                    execution_time = time.time() - start_time
                    
                    # 檢查全數據結果
                    unique_labels_full = np.unique(labels)
                    if len(unique_labels_full) < 2:
                        print(f"  K={k:2d}: 全數據聚類失敗（只有一個聚類）")
                        continue
                    
                    full_counts = np.bincount(labels)
                    full_max_ratio = full_counts.max() / full_counts.min()
                    if full_max_ratio > 500:  # 如果比例超過500，降低評分
                        print(f"  K={k:2d}: 結果不平衡（比例 {full_max_ratio:.1f}），降低評分")
                        continue
                    
                    metrics = self.evaluate_clustering(labels, self.features, 
                                                     f"Agglomerative-{linkage}(k={k})")
                    metrics['execution_time'] = execution_time
                    
                    silhouette = metrics['silhouette_score']
                    
                    # 加入平衡性懲罰
                    balance_penalty = min(1.0, 10.0 / full_max_ratio)  # 平衡性懲罰
                    adjusted_score = silhouette * balance_penalty
                    
                    print(f"  K={k:2d}: Silhouette={silhouette:.4f}, "
                          f"平衡比例={full_max_ratio:.1f}, "
                          f"調整分數={adjusted_score:.4f}, "
                          f"執行時間={execution_time:.2f}s")
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_params = (k, linkage)
                        best_labels = labels
                        best_model = agg
                        best_metrics = metrics
                        best_metrics['adjusted_score'] = adjusted_score
                        best_metrics['balance_ratio'] = full_max_ratio
                        
                except Exception as e:
                    print(f"  K={k:2d}: 計算失敗 - {str(e)}")
                    continue
        
        if best_labels is not None:
            self.results['agglomerative'] = {
                'labels': best_labels,
                'model': best_model,
                'metrics': best_metrics,
                'best_params': best_params
            }
            print(f"最佳層次聚類: K={best_params[0]}, linkage={best_params[1]}, "
                  f"Silhouette={best_metrics['silhouette_score']:.4f}, "
                  f"調整分數={best_metrics['adjusted_score']:.4f}")
        else:
            print("⚠️ 層次聚類無法找到合適的結果")
        
        return best_labels, best_metrics if best_labels is not None else None

    def compare_all_algorithms(self):
        """比較所有聚類算法的結果"""
        print("\n" + "="*70)
        print("聚類算法性能比較")
        print("="*70)
        
        if not self.results:
            print("沒有可比較的結果")
            return None
        
        # 創建比較表格
        comparison_data = []
        for method, result_info in self.results.items():
            metrics = result_info['metrics']
            comparison_data.append({
                '算法': method.upper(),
                '聚類數': metrics['n_clusters'],
                '噪聲點': metrics.get('n_noise', 0),
                'Silhouette Score': f"{metrics['silhouette_score']:.4f}",
                'Calinski-Harabasz': f"{metrics.get('calinski_harabasz_score', 0):.1f}",
                'Davies-Bouldin': f"{metrics.get('davies_bouldin_score', 0):.4f}",
                '執行時間(秒)': f"{metrics['execution_time']:.2f}"
            })
        
        # 顯示比較表格
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # 找到最佳方法
        best_method = max(self.results.keys(), 
                         key=lambda x: self.results[x]['metrics']['silhouette_score'])
        best_score = self.results[best_method]['metrics']['silhouette_score']
        
        print(f"\n推薦算法: {best_method.upper()}")
        print(f"最佳Silhouette Score: {best_score:.4f}")
        
        return best_method, self.results[best_method]
    
    def create_submission_file(self, labels, filename="public_submission.csv"):
        """創建提交檔案"""
        print(f"\n創建提交檔案: {filename}")
        
        submission = pd.DataFrame({
            'id': self.data['id'],
            'label': labels
        })
        
        # 確保標籤從0開始且連續
        unique_labels = sorted(set(labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        submission['label'] = submission['label'].map(label_mapping)
        
        submission.to_csv(filename, index=False)
        
        print(f"提交檔案已保存: {filename}")
        print(f"總計 {len(submission)} 筆記錄")
        print(f"聚類標籤分佈:")
        print(submission['label'].value_counts().sort_index())
        
        return submission
    
    def visualize_clustering_results(self, method_name, labels):
        """可視化聚類結果"""
        if not PLOT_AVAILABLE:
            print("無法載入matplotlib，跳過可視化")
            return
        
        print(f"\n生成 {method_name} 聚類結果可視化...")
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建子圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{method_name} 聚類分析結果', fontsize=16, fontweight='bold')
        
        # 1. PCA 2D可視化
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(self.features)
        
        scatter1 = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                     c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 0].set_title('PCA 2D 聚類結果')
        axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} 變異)')
        axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} 變異)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # 2. 聚類大小分佈
        unique_labels, counts = np.unique(labels, return_counts=True)
        bars = axes[0, 1].bar(range(len(unique_labels)), counts, 
                             color=plt.cm.tab10(np.linspace(0, 1, len(unique_labels))))
        axes[0, 1].set_title('聚類大小分佈')
        axes[0, 1].set_xlabel('聚類標籤')
        axes[0, 1].set_ylabel('樣本數量')
        axes[0, 1].set_xticks(range(len(unique_labels)))
        axes[0, 1].set_xticklabels(unique_labels)
        
        # 添加數值標籤
        for bar, count in zip(bars, counts):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                           str(count), ha='center', va='bottom', fontsize=9)
        
        # 3. 特徵相關性熱圖
        feature_df = pd.DataFrame(self.features, columns=self.feature_columns)
        corr_matrix = feature_df.corr()
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('特徵相關性矩陣')
        axes[1, 0].set_xticks(range(len(self.feature_columns)))
        axes[1, 0].set_yticks(range(len(self.feature_columns)))
        axes[1, 0].set_xticklabels(self.feature_columns)
        axes[1, 0].set_yticklabels(self.feature_columns)
        
        # 添加相關性數值
        for i in range(len(self.feature_columns)):
            for j in range(len(self.feature_columns)):
                axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. t-SNE可視化（如果樣本數不太大）
        if len(self.features) <= 10000:  # t-SNE對大數據集較慢
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.features)-1))
                features_tsne = tsne.fit_transform(self.features)
                
                scatter2 = axes[1, 1].scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                            c=labels, cmap='tab10', alpha=0.7, s=20)
                axes[1, 1].set_title('t-SNE 2D 聚類結果')
                axes[1, 1].set_xlabel('t-SNE 1')
                axes[1, 1].set_ylabel('t-SNE 2')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter2, ax=axes[1, 1])
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f't-SNE 計算失敗:\n{str(e)}', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('t-SNE 2D (計算失敗)')
        else:
            axes[1, 1].text(0.5, 0.5, 'data too large for t-SNE\n(>10,000 samples)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('t-SNE 2D (數據過大)')
        
        plt.tight_layout()
        
        # 保存圖片
        filename = f"{method_name.lower()}_clustering_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可視化結果已保存: {filename}")
    
    def generate_clustering_report(self, best_method, best_result):
        """生成詳細的聚類分析報告"""
        report_filename = "clustering_analysis_report.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("大數據分析期末專案 - 進階聚類分析報告\n")
            f.write("="*70 + "\n\n")
            
            # 數據概述
            f.write("1. 數據概述\n")
            f.write("-" * 20 + "\n")
            f.write(f"數據集大小: {self.data.shape[0]} 樣本, {len(self.feature_columns)} 特徵\n")
            f.write(f"特徵列: {', '.join(self.feature_columns)}\n")
            f.write(f"縮放方法: {type(self.scaler).__name__}\n\n")
            
            # 算法比較
            f.write("2. 算法性能比較\n")
            f.write("-" * 20 + "\n")
            for method, result_info in self.results.items():
                metrics = result_info['metrics']
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  - 聚類數: {metrics['n_clusters']}\n")
                f.write(f"  - 噪聲點: {metrics.get('n_noise', 0)}\n")
                f.write(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}\n")
                f.write(f"  - Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 0):.2f}\n")
                f.write(f"  - Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.4f}\n")
                f.write(f"  - 執行時間: {metrics['execution_time']:.2f} 秒\n")
            
            # 最佳結果
            f.write(f"\n3. 推薦結果\n")
            f.write("-" * 20 + "\n")
            f.write(f"推薦算法: {best_method.upper()}\n")
            best_metrics = best_result['metrics']
            f.write(f"聚類數: {best_metrics['n_clusters']}\n")
            f.write(f"Silhouette Score: {best_metrics['silhouette_score']:.4f}\n")
            f.write(f"執行時間: {best_metrics['execution_time']:.2f} 秒\n")
            
            # 聚類分佈
            f.write(f"\n4. 聚類分佈\n")
            f.write("-" * 20 + "\n")
            labels = best_result['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                percentage = count / len(labels) * 100
                f.write(f"聚類 {label}: {count} 樣本 ({percentage:.1f}%)\n")
        
        print(f"詳細報告已保存: {report_filename}")

def main():
    """主函數"""
    print("="*70)
    print("大數據分析期末專案 - 進階聚類分析")
    print("="*70)
    
    # 1. 初始化分析器
    analysis = AdvancedClusteringAnalysis()
    
    # 2. 載入數據
    if not analysis.load_data():
        return
    
    # 3. 數據預處理
    if not analysis.preprocess_data():
        return
    
    # 4. 應用各種聚類算法並為每個算法生成CSV檔案
    print("\n開始應用各種聚類算法...")
    
    try:
        print("\n正在執行 K-Means...")
        analysis.apply_kmeans_variants(max_k=12)
        if 'kmeans' in analysis.results:
            analysis.create_submission_file(
                analysis.results['kmeans']['labels'], 
                filename="kmeans_submission.csv"
            )
    except Exception as e:
        print(f"K-Means 分析失敗: {e}")
    
    try:
        print("\n正在執行 DBSCAN...")
        analysis.apply_dbscan_variants()
        if 'dbscan' in analysis.results:
            analysis.create_submission_file(
                analysis.results['dbscan']['labels'], 
                filename="dbscan_submission.csv"
            )
    except Exception as e:
        print(f"DBSCAN 分析失敗: {e}")
    
    try:
        print("\n正在執行 HDBSCAN...")
        analysis.apply_hdbscan()
        if 'hdbscan' in analysis.results:
            analysis.create_submission_file(
                analysis.results['hdbscan']['labels'], 
                filename="hdbscan_submission.csv"
            )
    except Exception as e:
        print(f"HDBSCAN 分析失敗: {e}")
    
    try:
        print("\n正在執行 譜聚類...")
        analysis.apply_spectral_clustering(max_k=8)
        if 'spectral' in analysis.results:
            analysis.create_submission_file(
                analysis.results['spectral']['labels'], 
                filename="spectral_submission.csv"
            )
    except Exception as e:
        print(f"譜聚類分析失敗: {e}")
    
    try:
        print("\n正在執行 GMM...")
        analysis.apply_gaussian_mixture(max_k=12)
        if 'gmm' in analysis.results:
            analysis.create_submission_file(
                analysis.results['gmm']['labels'], 
                filename="gmm_submission.csv"
            )
    except Exception as e:
        print(f"GMM 分析失敗: {e}")
    
    try:
        print("\n正在執行 層次聚類...")
        analysis.apply_agglomerative_clustering(max_k=12)
        if 'agglomerative' in analysis.results:
            analysis.create_submission_file(
                analysis.results['agglomerative']['labels'], 
                filename="agglomerative_submission.csv"
            )
    except Exception as e:
        print(f"層次聚類分析失敗: {e}")
    
    # 5. 比較所有算法結果
    if analysis.results:
        best_method, best_result = analysis.compare_all_algorithms()
        
        # 6. 創建最佳結果的提交檔案
        if best_method is not None:
            submission = analysis.create_submission_file(
                best_result['labels'], 
                filename="public_submission.csv"
            )
            
            # 7. 可視化結果
            analysis.visualize_clustering_results(best_method, best_result['labels'])
            
            # 8. 生成詳細報告
            analysis.generate_clustering_report(best_method, best_result)
            
            # 9. 輸出總結
            print("\n" + "="*70)
            print("分析完成！")
            print("="*70)
            print(f"推薦聚類算法: {best_method.upper()}")
            print(f"聚類數量: {best_result['metrics']['n_clusters']}")
            print(f"Silhouette Score: {best_result['metrics']['silhouette_score']:.4f}")
            print(f"執行時間: {best_result['metrics']['execution_time']:.2f} 秒")
            print("\n生成的檔案:")
            print("各算法預測結果:")
            for method in analysis.results.keys():
                print(f"- {method}_submission.csv")
            print("整合結果:")
            print("- public_submission.csv (最佳算法結果)")
            print("- clustering_analysis_report.txt (詳細報告)")
            print(f"- {best_method.lower()}_clustering_analysis.png (可視化結果)")
            
            return submission
        else:
            print("沒有最佳結果可以生成")
            return None
    else:
        print("沒有成功的聚類結果")
        return None

if __name__ == "__main__":
    submission = main() 