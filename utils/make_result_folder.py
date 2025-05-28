import os

def create_next_numbered_folder(root_path):
    # 주어진 경로에서 폴더들을 확인합니다.
    existing_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    # 폴더 이름이 숫자로 되어 있는 것들만 필터링합니다.
    numbered_folders = [int(folder) for folder in existing_folders if folder.isdigit()]
    
    # 가장 큰 숫자를 찾습니다. 만약 폴더가 없다면 0부터 시작합니다.
    next_number = max(numbered_folders) + 1 if numbered_folders else 0
    
    # 새 폴더 이름을 결정합니다.
    new_folder_name = str(next_number)
    new_folder_path = os.path.join(root_path, new_folder_name)
    
    # 새 폴더를 생성합니다.
    os.makedirs(new_folder_path, exist_ok=True)
    
    return new_folder_path