#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚類結果分析和比較腳本
"""

import pandas as pd
import numpy as np
import os

def analyze_clustering_results():
    """分析所有聚類結果"""
    
    # 算法列表
    algorithms = ['kmeans', 'dbscan', 'hdbscan', 'spectral', 'gmm', 'agglomerative']
    
    results_summary = []
    detailed_distributions = {}
    
    print("="*80)
    print("聚類算法結果分析")
    print("="*80)
    
    for alg in algorithms:
        filename = f'{alg}_submission.csv'
        if os.path.exists(filename):
            print(f"\n=== {alg.upper()} ===")
            
            # 讀取結果
            df = pd.read_csv(filename)
            label_counts = df['label'].value_counts().sort_index()
            
            # 基本統計
            n_clusters = len(label_counts)
            total_samples = len(df)
            max_cluster_size = label_counts.max()
            min_cluster_size = label_counts.min()
            
            # 平衡性指標
            std_cluster_size = label_counts.std()
            imbalance_ratio = max_cluster_size / min_cluster_size if min_cluster_size > 0 else float('inf')
            
            # 顯示詳細分佈
            print(f"聚類數量: {n_clusters}")
            print(f"總樣本數: {total_samples}")
            print(f"聚類分佈:")
            for label, count in label_counts.items():
                percentage = count / total_samples * 100
                print(f"  聚類 {label}: {count:5d} 樣本 ({percentage:5.2f}%)")
            
            print(f"最大聚類大小: {max_cluster_size}")
            print(f"最小聚類大小: {min_cluster_size}")
            print(f"不平衡比例: {imbalance_ratio:.2f}")
            print(f"聚類大小標準差: {std_cluster_size:.2f}")
            
            # 儲存到匯總
            results_summary.append({
                '算法': alg.upper(),
                '聚類數': n_clusters,
                '總樣本': total_samples,
                '最大聚類': max_cluster_size,
                '最小聚類': min_cluster_size,
                '不平衡比例': f"{imbalance_ratio:.2f}" if imbalance_ratio != float('inf') else "∞",
                '標準差': f"{std_cluster_size:.1f}",
                '平衡度': "優" if imbalance_ratio < 2 else "良" if imbalance_ratio < 5 else "差"
            })
            
            detailed_distributions[alg] = label_counts
        else:
            print(f"檔案不存在: {filename}")
    
    # 生成比較表格
    print("\n" + "="*80)
    print("聚類算法比較表格")
    print("="*80)
    
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        # 儲存表格到檔案
        summary_df.to_csv('clustering_comparison_table.csv', index=False, encoding='utf-8')
        print(f"\n比較表格已儲存至: clustering_comparison_table.csv")
        
        # 檢查異常情況
        print("\n" + "="*80)
        print("異常檢測")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            alg = row['算法']
            balance_ratio = row['不平衡比例']
            
            if balance_ratio == "∞":
                print(f"⚠️  {alg}: 存在空聚類（某些聚類大小為0）")
            elif balance_ratio != "∞" and float(balance_ratio) > 100:
                print(f"⚠️  {alg}: 嚴重不平衡（比例 {balance_ratio}）")
            elif balance_ratio != "∞" and float(balance_ratio) > 10:
                print(f"⚠️  {alg}: 中度不平衡（比例 {balance_ratio}）")
        
        # 特別分析agglomerative的問題
        if 'agglomerative' in detailed_distributions:
            agg_dist = detailed_distributions['agglomerative']
            print(f"\n🔍 層次聚類特別分析:")
            print(f"   標籤0: {agg_dist[0]} 樣本 ({agg_dist[0]/agg_dist.sum()*100:.4f}%)")
            print(f"   標籤1: {agg_dist[1]} 樣本 ({agg_dist[1]/agg_dist.sum()*100:.4f}%)")
            print(f"   這表明層次聚類將幾乎所有樣本分配到同一個聚類中")
            print(f"   可能的原因：")
            print(f"   1. 資料特性導致層次聚類傾向於形成一個大聚類")
            print(f"   2. 參數設定不當")
            print(f"   3. 資料預處理問題")
            print(f"   4. 演算法在此資料集上不適用")
    
    return results_summary, detailed_distributions

def check_data_characteristics():
    """檢查原始數據特性"""
    print("\n" + "="*80)
    print("原始數據特性分析")
    print("="*80)
    
    # 嘗試載入原始數據
    data_files = ['data/public_data.csv', 'big data/public_data.csv', 'public_data.csv']
    data = None
    
    for file_path in data_files:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            print(f"成功載入數據: {file_path}")
            break
    
    if data is None:
        print("無法找到原始數據檔案")
        return
    
    print(f"數據形狀: {data.shape}")
    print(f"特徵欄位: {list(data.columns)}")
    
    # 排除id欄位
    feature_cols = [col for col in data.columns if col != 'id']
    features = data[feature_cols]
    
    print(f"\n特徵統計:")
    print(features.describe())
    
    print(f"\n特徵相關性（絕對值平均）:")
    corr_matrix = features.corr()
    avg_corr = np.abs(corr_matrix).mean().mean()
    print(f"平均相關性: {avg_corr:.4f}")
    
    print(f"\n特徵範圍:")
    for col in feature_cols:
        min_val = features[col].min()
        max_val = features[col].max()
        range_val = max_val - min_val
        print(f"{col}: [{min_val:.4f}, {max_val:.4f}], 範圍={range_val:.4f}")

def main():
    """主函數"""
    # 檢查當前目錄
    print(f"當前工作目錄: {os.getcwd()}")
    
    # 分析聚類結果
    results_summary, detailed_distributions = analyze_clustering_results()
    
    # 檢查數據特性
    check_data_characteristics()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main() 