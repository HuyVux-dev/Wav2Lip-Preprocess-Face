import os
import sys
import time
import subprocess
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm
import platform
import audio
import face_detection
from models import Wav2Lip

def log_time(func):
    """Decorator để tính thời gian chạy của các hàm"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} hoàn thành trong {end_time - start_time:.4f} giây.")
        return result

    return wrapper

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

@log_time
def face_detect(images, pads, face_det_batch_size, nosmooth, img_size, device):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Hình ảnh quá lớn để chạy phát hiện khuôn mặt trên GPU. Vui lòng sử dụng tham số resize_factor.')
            batch_size //= 2
            print('Khôi phục từ lỗi OOM; Kích thước batch mới: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # Kiểm tra khung hình này nơi khuôn mặt không được phát hiện.
            raise ValueError('Khuôn mặt không được phát hiện! Đảm bảo video chứa khuôn mặt trong tất cả các khung hình.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    face_det_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
                        for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return face_det_results

@log_time
def save_face_det_results(face_det_results):
    os.makedirs('temp', exist_ok=True)
    with open('temp/face_det_results.pkl', 'wb') as f:
        pickle.dump(face_det_results, f)

@log_time
def load_face_det_results():
    with open('temp/face_det_results.pkl', 'rb') as f:
        face_det_results = pickle.load(f)
    return face_det_results

@log_time
def process_face_detection(face_path, pads, resize_factor, rotate, crop, box, static, nosmooth,
                           face_det_batch_size, img_size, device):
    # Đọc video hoặc hình ảnh
    if not os.path.isfile(face_path):
        raise ValueError('--face phải là đường dẫn hợp lệ đến file video/hình ảnh')

    elif face_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face_path)]
    else:
        video_stream = cv2.VideoCapture(face_path)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame,
                                   (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("Số lượng khung hình có sẵn để phát hiện khuôn mặt: " + str(len(full_frames)))

    # Thực hiện phát hiện khuôn mặt
    print('Đang chạy phát hiện khuôn mặt...')
    if box[0] == -1:
        if not static:
            face_det_results = face_detect(full_frames, pads, face_det_batch_size, nosmooth, img_size, device)
        else:
            face_det_results = face_detect([full_frames[0]], pads, face_det_batch_size, nosmooth, img_size, device)
            face_det_results = face_det_results * len(full_frames)
    else:
        print('Sử dụng bounding box được chỉ định thay vì phát hiện khuôn mặt...')
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

    # Lưu kết quả phát hiện khuôn mặt
    save_face_det_results(face_det_results)
    print('Hoàn thành phát hiện khuôn mặt và đã lưu kết quả.')

@log_time
def datagen(frames, mels, face_det_results, static, img_size, wav2lip_batch_size):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch,
                                   [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch,
                               [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16

def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

@log_time
def load_model(path, device):
    model = Wav2Lip()
    print("Đang tải checkpoint từ: {}".format(path))
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

@log_time
def main(checkpoint_path, face, audio_path=None, outfile='results/result_voice.mp4', static=False, fps=25.0,
         pads=(0, 10, 0, 0), face_det_batch_size=16, wav2lip_batch_size=128, resize_factor=1,
         crop=(0, -1, 0, -1), box=(-1, -1, -1, -1), rotate=False, nosmooth=False, mode='synthesize',
         img_size=96):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Sử dụng {} để suy luận.'.format(device))

    if mode == 'detect':
        process_face_detection(face, pads, resize_factor, rotate, crop, box, static, nosmooth,
                               face_det_batch_size, img_size, device)
        return

    # Chế độ 'synthesize'
    if not os.path.isfile(face):
        raise ValueError('--face phải là đường dẫn hợp lệ đến file video/hình ảnh')

    elif face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face)]
        fps = fps
    else:
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Đang đọc các khung hình video...')

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame,
                                   (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Số lượng khung hình có sẵn để suy luận: " + str(len(full_frames)))

    # Tải kết quả phát hiện khuôn mặt
    if not os.path.exists('temp/face_det_results.pkl'):
        raise FileNotFoundError('Không tìm thấy kết quả phát hiện khuôn mặt. Vui lòng chạy script với --mode detect trước.')
    else:
        print('Đang tải kết quả phát hiện khuôn mặt...')
        face_det_results = load_face_det_results()

    if not audio_path:
        raise ValueError('Vui lòng cung cấp đường dẫn đến file âm thanh bằng cách sử dụng tham số audio_path.')

    if not audio_path.endswith('.wav'):
        print('Đang trích xuất âm thanh...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'

    # Bắt đầu tính thời gian xử lý âm thanh
    start_time = time.time()
    wav_audio = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav_audio)
    print(mel.shape)
    end_time = time.time()
    print(f"Xử lý âm thanh mất {end_time - start_time:.4f} giây.")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel chứa nan! Sử dụng giọng nói TTS? Thêm một chút nhiễu epsilon vào file wav và thử lại')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Độ dài của mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]
    face_det_results = face_det_results[:len(mel_chunks)]

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, face_det_results, static, img_size, wav2lip_batch_size)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(checkpoint_path, device)
            print("Đã tải mô hình")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        # Đo thời gian suy luận của mô hình
        start_time = time.time()
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        end_time = time.time()
        print(f"Suy luận mô hình cho batch {i} mất {end_time - start_time:.4f} giây.")

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

# Nếu bạn muốn chạy trực tiếp từ file này
if __name__ == '__main__':
    # Khai báo các biến cần thiết
    checkpoint_path = 'checkpoints/wav2lip_gan.pth'
    face = 'examples/face/1.mp4'
    audio_path = 'examples/audio/1.wav'
    outfile = 'results/output_video_test.mp4'
    static = False
    fps = 25.0
    pads = (0, 10, 0, 0)
    face_det_batch_size = 16
    wav2lip_batch_size = 128
    resize_factor = 1
    crop = (0, -1, 0, -1)
    box = (-1, -1, -1, -1)
    rotate = False
    nosmooth = False
    mode = 'synthesize'
    img_size = 96

    if os.path.isfile(face) and face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        static = True

    main(checkpoint_path, face, audio_path, outfile, static, fps, pads, face_det_batch_size,
         wav2lip_batch_size, resize_factor, crop, box, rotate, nosmooth, mode, img_size)
