def get_unique_filename(filepath: str) -> str:
    """
    파일이 이미 존재할 경우 파일명_1, 파일명_2 등으로 변경
    
    Parameters:
    -----------
    filepath : str
        원본 파일 경로
    
    Returns:
    --------
    str : 유니크한 파일 경로
    """
    if not os.path.exists(filepath):
        return filepath
    
    # 파일 경로와 확장자 분리
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    # 새로운 파일명 생성
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1
