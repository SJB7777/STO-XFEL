import numpy as np
import pandas as pd

from src.preprocess.data_evaluation_tools import evaluate_preprocessing

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
