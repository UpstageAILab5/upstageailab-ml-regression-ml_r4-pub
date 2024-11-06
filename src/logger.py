import logging
import os

class Logger():
    def __init__(self):
        self.logger = logging.getLogger()
        logging.info(f"Initialized Logger.")

    def setup_logger(self):
        """로거 설정: 핸들러 중복 방지"""
        logger = self.logger
        
        # 이미 핸들러가 있다면 모두 제거
        if logger.handlers:
            logger.handlers.clear()
        
        logger.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
