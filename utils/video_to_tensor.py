import torch
import cv2

def load_video_as_tensor(video_path, frame_nums, height, width):
    # 비디오를 로드하고 프레임을 지정된 크기로 변환
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened() and frame_count < frame_nums:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR에서 RGB로 변환 및 크기 조정
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frame = torch.tensor(frame, dtype=torch.float32) / 255.0  # [0, 1] 범위로 정규화
        frames.append(frame)
        frame_count += 1

    cap.release()

    # frame_nums에 맞춰 패딩 추가
    while len(frames) < frame_nums:
        frames.append(frames[-1] if frames else torch.zeros((height, width, 3), dtype=torch.float32))

    frames = torch.stack(frames).permute(3, 0, 1, 2)  # shape: (channels, frame_nums, height, width)
    return frames

def create_video_batch_tensor(video_paths, batch_size, frame_nums, height, width):
    batch_videos = []

    for video_path in video_paths[:batch_size]:  # batch_size 만큼의 비디오 로드
        video_tensor = load_video_as_tensor(video_path, frame_nums, height, width)
        batch_videos.append(video_tensor)

    batch_tensor = torch.stack(batch_videos)  # shape: (batch_size, channels, frame_nums, height, width)
    return batch_tensor