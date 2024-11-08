from feature_engine.transformation import YeoJohnsonTransformer, BoxCoxTransformer
import pandas as pd
# 예제 데이터셋 생성
data = {
    'feature_1': [1, 2, 3, 4, 5, 1000],
    'feature_2': [0.5, 1.5, 2.5, 3.5, 4.5, 100],
    'feature_3': [10, 20, 30, 40, 50, 60]
}
df = pd.DataFrame(data)

# 자동화된 변환기 설정
# Yeo-Johnson 변환: 음수와 양수 모두 처리 가능 (Box-Cox는 양수만)
transformer = YeoJohnsonTransformer(variables=None)  # None이면 모든 수치형 변수에 적용

# 변환 실행
df_transformed = transformer.fit_transform(df)

# 결과 확인
print("Original Data:")
print(df)
print("\nTransformed Data:")
print(df_transformed)

from ipywidgets import interact
import ipywidgets as widgets

@interact(scale_type=['Standard', 'MinMax'], feature=data.columns)
def plot_scaled(scale_type, feature):
    if scale_type == 'Standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(data[[feature]])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data[feature], kde=True)
    plt.title('Original Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(scaled_data, kde=True)
    plt.title(f'{scale_type} Scaled Distribution')
    
    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# 데이터셋 로드 (예: 펭귄 데이터셋)
df = sns.load_dataset("penguins")

# pairplot 생성
sns.pairplot(df, hue="species")
plt.show()