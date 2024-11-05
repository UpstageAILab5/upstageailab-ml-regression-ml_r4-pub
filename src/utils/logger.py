import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

class Logger:
    """로깅 설정 및 관리 클래스"""
    
    def __init__(self, 
                 name: str = 'real_estate_ml',
                 log_dir: str = 'logs',
                 level: int = logging.INFO,
                 backup_count: int = 30):
        """
        Parameters
        ----------
        name : str
            로거 이름
        log_dir : str
            로그 파일 저장 디렉토리
        level : int
            로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        backup_count : int
            보관할 로그 파일 수
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        self.backup_count = backup_count
        self.logger = self._create_logger()
        
    def _create_logger(self) -> logging.Logger:
        """로거 생성 및 설정"""
        # 로거 생성
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # 이미 핸들러가 있다면 제거 (중복 방지)
        if logger.handlers:
            logger.handlers.clear()
            
        # 로그 포맷 설정
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 추가
        if self.log_dir:
            self._setup_file_handler(logger, formatter)
            
        return logger
        
    def _setup_file_handler(self, 
                           logger: logging.Logger, 
                           formatter: logging.Formatter) -> None:
        """파일 핸들러 설정"""
        # 로그 디렉토리 생성
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 날짜별 로그 파일 설정
        log_file = os.path.join(
            self.log_dir,
            f"{self.name}_{datetime.now():%Y%m%d}.log"
        )
        
        # 파일 핸들러 생성 (일별 로테이션)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    def get_logger(self) -> logging.Logger:
        """설정된 로거 반환"""
        return self.logger
        
    @staticmethod
    def setup_wandb_logging(project_name: str, 
                           run_name: Optional[str] = None,
                           config: Optional[dict] = None) -> None:
        """WandB 로깅 설정"""
        import wandb
        
        wandb.init(
            project=project_name,
            name=run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}",
            config=config,
            reinit=True
        )
