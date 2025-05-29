# src/classification_algorithms.py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_logistic_regression(X_train, y_train, random_state=42, solver='liblinear'):
    """訓練邏輯回歸模型。"""
    model = LogisticRegression(random_state=random_state, solver=solver, multi_class='auto', max_iter=1000)
    model.fit(X_train, y_train)
    print("邏輯回歸模型訓練完成。")
    return model

def train_svm(X_train, y_train, random_state=42, kernel='rbf', C=1.0, gamma='scale'):
    """訓練支持向量機模型。"""
    model = SVC(random_state=random_state, kernel=kernel, C=C, gamma=gamma, probability=True) # probability=True for ROC AUC if needed
    model.fit(X_train, y_train)
    print("SVM 模型訓練完成。")
    return model

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100, max_depth=None):
    """訓練隨機森林模型。"""
    model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    print("隨機森林模型訓練完成。")
    return model

if __name__ == '__main__':
    # 示例數據
    from sklearn.datasets import make_classification
    X_sample, y_sample = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=42)
    
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

    lr_model = train_logistic_regression(X_train_sample, y_train_sample)
    svm_model = train_svm(X_train_sample, y_train_sample)
    rf_model = train_random_forest(X_train_sample, y_train_sample)

    print("\n模型預測示例 (Logistic Regression):", lr_model.predict(X_test_sample[:5]))
    print("模型預測示例 (SVM):", svm_model.predict(X_test_sample[:5]))
    print("模型預測示例 (Random Forest):", rf_model.predict(X_test_sample[:5]))