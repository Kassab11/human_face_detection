import cv2
import argparse
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from mmcv import VideoReader

def detect_and_highlight_faces(video_path, output_path, log_file, detection_model, mtcnn_weights=None):
    if detection_model == 'haarcascade':
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open selected video file")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    elif detection_model == 'mtcnn':
        video_reader = VideoReader(video_path)
        frame_width, frame_height = video_reader.width, video_reader.height
        fps = video_reader.fps

    if detection_model == 'haarcascade':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        frame_number = 0
        logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.15, minNeighbors=8, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    time_in_seconds = frame_number / fps
                    logging.info(f"Haarcascade: Face detected at second {time_in_seconds:.2f} in the video.")

            out.write(frame)

            frame_number += 1

        cap.release()
        out.release()

    elif detection_model == 'mtcnn':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if mtcnn_weights:
            mtcnn = MTCNN(keep_all=True, device=device, weights_path=mtcnn_weights)
        else:
            mtcnn = MTCNN(keep_all=True, device=device)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_tracked = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_number = 0
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(message)s')

        for frame in video_reader:
            frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None:
                frame_draw = Image.fromarray(frame.copy())
                draw = ImageDraw.Draw(frame_draw)
                for box in boxes:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                video_tracked.write(np.array(frame_draw))
                
                time_in_seconds = frame_number / fps
                logging.info(f"MTCNN: Face detected at second {time_in_seconds:.2f} in the video.")
            else:
                video_tracked.write(frame)

            frame_number += 1

        #video_reader.close()
        video_tracked.release()

def main():
    parser = argparse.ArgumentParser(description="Process a video file and detect and highlight human faces.")
    parser.add_argument("--video_path", help="Path to input video file.")
    parser.add_argument("--output_path", help="Path to save the output video with highlighted faces.")
    parser.add_argument("--log_file", help="Path to save the log file.")
    parser.add_argument("--detection_model", choices=['haarcascade', 'mtcnn'], default='mtcnn',
                        help="Choose the face detection model to use: 'haarcascade' or 'mtcnn'.")
    parser.add_argument("--mtcnn_weights", help="Path to fine-tuned MTCNN weights (optional).")
    args = parser.parse_args()

    detect_and_highlight_faces(args.video_path, args.output_path, args.log_file, args.detection_model, args.mtcnn_weights)

if __name__ == "__main__":
    main()
