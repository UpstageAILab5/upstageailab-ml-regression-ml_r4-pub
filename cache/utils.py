# import platform
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import seaborn as sns
# import matplotlib as mpl
# import os

# def setup_matplotlib_korean(logger):
#     """matplotlib 한글 폰트 설정"""
#     import os
    
#     # 운영체제 확인
#     system = platform.system()
    
#     if system == 'Windows':
#         # 기존 설정 초기화
#         plt.rcParams.update(plt.rcParamsDefault)
        
#         # 폰트 직접 지정
#         plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
#         plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호
        
#         logger.info("Windows 폰트 설정 완료: Malgun Gothic")
    
#     elif system == 'Darwin':  # Mac
#         plt.rcParams['font.family'] = 'AppleGothic'
#     else:  # Linux
#         plt.rcParams['font.family'] = 'NanumGothic'
    
#     # 기본 스타일 설정
#     plt.style.use('seaborn')
#     sns.set_palette("husl")
    
#     # 폰트 관련 설정
#     plt.rcParams.update({
#         'font.size': 10,
#         'figure.figsize': (10, 6),
#         'axes.grid': True
#     })
    
#     # 설정 확인을 위한 로깅
#     log_items = {
#         'system': system,
#         'font': plt.rcParams["font.family"],
#         'font size': plt.rcParams["font.size"],
#         'font unicode_minus': plt.rcParams["axes.unicode_minus"]
#     }

#     logger.info('## 시각화 스타일 설정')
#     for key, value in log_items.items():
#         logger.info(f'{key}: {value}')

#     # 설정 테스트는 처음 호출할 때만 실행
#     if not hasattr(setup_matplotlib_korean, 'initialized'):
#         plt.figure(figsize=(3, 1))
#         plt.text(0.5, 0.5, '한글 테스트 2', ha='center', va='center')
#         plt.axis('off')
#         plt.show()
#         setup_matplotlib_korean.initialized = True
    
#     return None

# # . Windows: 나눔글꼴 설치 페이지에서 다운로드
# # 2. Linux: sudo apt-get install fonts-nanum
# # Mac: brew cask install font-nanum
