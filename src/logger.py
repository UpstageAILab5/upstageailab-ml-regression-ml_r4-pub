import logging
import sys
import os

class Logger:
    def __init__(self):
        # 로거 인스턴스 생성
        self.logger = logging.getLogger()
        self.logger = self.setup_logger()
        self.logger.info("Initialized Logger.")

    def setup_logger(self):
        """로거 설정: 핸들러 중복 방지 및 한글 인코딩 설정"""
        logger = self.logger
        
        # 이미 핸들러가 있다면 모두 제거
        if logger.handlers:
            logger.handlers.clear()
        
        logger.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 콘솔 핸들러 추가 (UTF-8 인코딩 설정)
        console_handler = logging.StreamHandler(sys.stdout)  # stdout으로 변경
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 추가 (선택사항, UTF-8 인코딩 설정)
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'preprocessing.log'), 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
