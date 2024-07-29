import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def handle_nan(data):
    """
    NaN 값을 열의 평균값으로 대체하는 함수

    Parameters:
    data (pd.DataFrame): 데이터프레임

    Returns:
    pd.DataFrame: NaN 값이 처리된 데이터프레임
    """
    return data.fillna(data.mean())

# 이상치 비율 평가 (IQR 방법 사용)
def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum().sum()

def evaluate_preprocessing(original_data, preprocessed_data):
    """
    전처리 후 데이터의 상태를 평가하는 함수

    Parameters:
    original_data (pd.DataFrame): 전처리 전 데이터
    preprocessed_data (pd.DataFrame): 전처리 후 데이터

    Returns:
    dict: 평가 결과를 담은 딕셔너리
    """
    results = {}

    # NaN 값 처리
    original_data = handle_nan(original_data)
    preprocessed_data = handle_nan(preprocessed_data)

    # 결측치 비율 평가
    original_missing_ratio = original_data.isnull().sum().sum() / original_data.size
    preprocessed_missing_ratio = preprocessed_data.isnull().sum().sum() / preprocessed_data.size
    results['missing_ratio_improvement'] = original_missing_ratio - preprocessed_missing_ratio

    original_outliers = detect_outliers(original_data)
    preprocessed_outliers = detect_outliers(preprocessed_data)
    results['outlier_ratio_improvement'] = original_outliers - preprocessed_outliers

    # 데이터 분포 평가 (표준편차 사용)
    original_std = original_data.std().mean()
    preprocessed_std = preprocessed_data.std().mean()
    results['std_deviation_improvement'] = original_std - preprocessed_std

    # 회귀 모델 성능 평가 (예시로 MSE와 R^2 사용)
    if original_data.shape[1] == preprocessed_data.shape[1]:
        original_mse = mean_squared_error(original_data, preprocessed_data)
        original_r2 = r2_score(original_data, preprocessed_data)
        results['original_mse'] = original_mse
        results['original_r2'] = original_r2

    return results

# 예시 사용법
if __name__ == "__main__":
    # 예시 데이터 생성
    original_data = pd.DataFrame({
        'A': [1, 2, 3, np.nan, 5],
        'B': [10, 20, 30, 40, 50]
    })
    preprocessed_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })

    # 평가 함수 호출
    evaluation_results = evaluate_preprocessing(original_data, preprocessed_data)
    print(evaluation_results)