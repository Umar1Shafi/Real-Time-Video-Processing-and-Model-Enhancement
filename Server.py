import os
import cv2
import socket
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import shutil
import torch
import uuid
import time
import random

# Constants
BASE_DIR = os.path.abspath(".")
NEW_DATA_DIR = os.path.join(BASE_DIR, "new_data")
FRAMES_DIR = os.path.join(NEW_DATA_DIR, "images", "train")
ANNOTATIONS_DIR = os.path.join(NEW_DATA_DIR, "labels", "train")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights")
NEW_TRAINING_PATH = os.path.join(BASE_DIR, "runs", "detect", "new_training")
NEW_TRAINING_WEIGHTS_PATH = os.path.join(NEW_TRAINING_PATH, "weights")
NEW_CONFIG_PATH = os.path.join(BASE_DIR, "new_config.yaml")
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(BASE_DIR, "data", "labels", "train")
BACKUP_DIR = os.path.join(BASE_DIR, "runs", "detect", "backups")
VIDEO_SAVE_PATH = os.path.join(BASE_DIR, "original_video.mp4")
PRE_TRAINING_VIDEO_SAVE_PATH = os.path.join(BASE_DIR, "pre_training_video.mp4")
POST_TRAINING_VIDEO_SAVE_PATH = os.path.join(BASE_DIR, "post_training_video.mp4")
COMBINED_VIDEO_SAVE_PATH = os.path.join(BASE_DIR, "combined_training_video.mp4")
TCP_IP = "192.168.1.3"
TCP_PORT = 5005
BUFFER_SIZE = 65507

def setup_directories():
    """Ensure all required directories exist."""
    for directory in [FRAMES_DIR, ANNOTATIONS_DIR, BACKUP_DIR]:
        os.makedirs(directory, exist_ok=True)

def cleanup_directories(directories):
    """Remove specified directories."""
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Cleaned up directory: {directory}")

def receive_frame(sock):
    """Receive a frame from the socket."""
    print("Receiving frame...")
    frame_size_bytes = sock.recv(4)
    if not frame_size_bytes:
        return None
    frame_size = int.from_bytes(frame_size_bytes, byteorder='big')
    frame_data = b''
    while len(frame_data) < frame_size:
        chunk = sock.recv(min(frame_size - len(frame_data), BUFFER_SIZE))
        if not chunk:
            return None
        frame_data += chunk
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)

def manage_backups():
    """Manage the backup directory by removing old backups."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    backups = sorted(os.listdir(BACKUP_DIR), reverse=True)
    if len(backups) > 3:
        for backup in backups[3:]:
            shutil.rmtree(os.path.join(BACKUP_DIR, backup))
        print("Old backups removed")

def annotate_and_save_frame(model, frame, frame_count, video_writer):
    """Annotate a frame with model predictions and save it to a video file."""
    results_list = model(frame, conf=0.02)
    for results in results_list:
        if results.boxes is not None:
            for box in results.boxes:
                conf = box.conf[0]
                if conf > 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{box.cls[0]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    video_writer.write(frame)
    print(f"Annotated and saved frame {frame_count}")

def save_and_annotate_frame(frame, frame_count, session_id, model, width, height):
    """Save a frame and its annotations."""
    frame_filename = os.path.join(FRAMES_DIR, f"{session_id}_frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved frame {frame_count}")

    results_list = model(frame, conf=0.02)
    annotations = []
    for results in results_list:
        if results.boxes is not None:
            for box in results.boxes:
                conf = box.conf[0]
                if conf > 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height
                    annotation = f"{int(box.cls[0])} {x_center} {y_center} {bbox_width} {bbox_height}"
                    annotations.append(annotation)
    if annotations:
        annotation_filename = os.path.join(ANNOTATIONS_DIR, f"{session_id}_frame_{frame_count}.txt")
        with open(annotation_filename, 'w') as f:
            f.write("\n".join(annotations))
        print(f"Saved annotations for frame {frame_count}")


def annotate_video(input_video_path, output_video_path, model):
    """Annotate a video with model predictions and save it."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video details - FPS: {fps}, Width: {width}, Height: {height}")

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or read error at frame {frame_count}")
            break

        # Ensure frame dimensions are consistent
        if frame.shape[0] != height or frame.shape[1] != width:
            print(f"Frame size mismatch at frame {frame_count}: {frame.shape}")
            continue

        # Get model predictions
        results_list = model(frame, conf=0.02)
        annotated_frame = frame.copy()
        for results in results_list:
            if results.boxes is not None:
                for box in results.boxes:
                    conf = box.conf[0]
                    if conf > 0.7:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{box.cls[0]}: {conf:.2f}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_video_path}")

def export_to_onnx(model):
    """Export the model to ONNX format."""
    print("Exporting model to ONNX format.")
    try:
        import onnx
        onnx_model_filename = "most_recent_model.onnx"
        onnx_model_path = os.path.join(BASE_DIR, onnx_model_filename)
        example_input = torch.randn(1, 3, 640, 640)
        torch.onnx.export(model.model, example_input, onnx_model_path)
        print("Model exported to ONNX format successfully:", onnx_model_path)
    except ModuleNotFoundError:
        print("ONNX module is not installed. Please install it with 'pip install onnx' to export the model.")

def train_model():
    """Train the model with new data."""
    print("Checking training directory...")
    cache_path = os.path.join(ANNOTATIONS_DIR, "train.cache")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Removed existing cache file: {cache_path}")

    if not os.listdir(ANNOTATIONS_DIR):
        print("No labels found in the annotations directory. Please check your dataset.")
        return False

    if os.path.exists(NEW_TRAINING_PATH):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"new_training_{timestamp}")
        shutil.move(NEW_TRAINING_PATH, backup_path)
        print(f"Moved previous new_training to {backup_path}")
        manage_backups()

    model = YOLO(os.path.join(BASE_MODEL_PATH, "best.pt"))  # Load the best model from the last training
    training_results = model.train(data=NEW_CONFIG_PATH, epochs=2, name='new_training')  # Increase epochs
    print("Model fine-tuning started with the new data.")
    return training_results is not None

def process_video():
    # Initialize variables and directories
    session_id = uuid.uuid4().hex
    model = YOLO(os.path.join(BASE_MODEL_PATH, "best.pt"))  # Load initial model
    setup_directories()

    frame_count = 0
    video_writer = None
    pre_training_writer = None

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((TCP_IP, TCP_PORT))
            sock.listen(1)
            conn, addr = sock.accept()
            print(f"Connected by {addr}")

            while True:
                frame = receive_frame(conn)
                if frame is None:
                    print("No frame received, exiting...")
                    break

                if video_writer is None:
                    height, width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30
                    video_writer = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, fps, (width, height))
                    pre_training_writer = cv2.VideoWriter(PRE_TRAINING_VIDEO_SAVE_PATH, fourcc, fps, (width, height))
                    if not video_writer.isOpened():
                        print(f"Failed to open video writer for {VIDEO_SAVE_PATH}")
                        return
                    if not pre_training_writer.isOpened():
                        print(f"Failed to open video writer for {PRE_TRAINING_VIDEO_SAVE_PATH}")
                        return

                video_writer.write(frame)
                print(f"Saved frame {frame_count} to video file.")

                annotate_and_save_frame(model, frame, frame_count, pre_training_writer)
                save_and_annotate_frame(frame, frame_count, session_id, model, width, height)

                frame_count += 1

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if video_writer is not None:
            video_writer.release()
        if pre_training_writer is not None:
            pre_training_writer.release()

        if os.listdir(FRAMES_DIR) and os.listdir(ANNOTATIONS_DIR):
            if train_model():
                # Reload the newly trained model
                model = YOLO(os.path.join(NEW_TRAINING_WEIGHTS_PATH, "best.pt"))
                # Annotate the video with the new model
                annotate_video(VIDEO_SAVE_PATH, POST_TRAINING_VIDEO_SAVE_PATH, model)
                transfer_random_subset()
        else:
            print("No data available for training. Skipping training.")

        cleanup_directories([FRAMES_DIR, ANNOTATIONS_DIR])
        export_to_onnx(model)


def combine_videos(video1_path, video2_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # Add labels to the frames
        label_new = "New Model"
        label_old = "Old Model"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)  # Red color
        thickness = 2
        cv2.putText(frame1, label_new, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame2, label_old, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)

        combined_frame = cv2.hconcat([frame1, frame2])
        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()


def transfer_random_subset():
    """Transfer a random subset of images and labels to the training directory."""
    image_files = os.listdir(FRAMES_DIR)
    label_files = os.listdir(ANNOTATIONS_DIR)

    if not image_files or not label_files:
        print("No images or labels found to transfer.")
        return

    label_files_set = set(os.path.splitext(f)[0] for f in label_files)
    image_files_set = set(os.path.splitext(f)[0] for f in image_files)
    common_files = label_files_set.intersection(image_files_set)

    if not common_files:
        print("No matching image-label pairs found.")
        return

    num_images = len(common_files)
    num_to_select = max(num_images // 6, 1)

    selected_files = random.sample(list(common_files), num_to_select)

    for file_base in selected_files:
        image_path = os.path.join(FRAMES_DIR, f"{file_base}.jpg")
        label_path = os.path.join(ANNOTATIONS_DIR, f"{file_base}.txt")
        new_image_path = os.path.join(TRAIN_IMAGES_DIR, os.path.basename(image_path))
        new_label_path = os.path.join(TRAIN_LABELS_DIR, os.path.basename(label_path))
        shutil.move(image_path, new_image_path)
        shutil.move(label_path, new_label_path)
        print(f"Moved {file_base}.jpg and {file_base}.txt to training data")

    shutil.rmtree(NEW_DATA_DIR)
    os.makedirs(NEW_DATA_DIR)
    print("Cleaned up the new_data directory.")

def play_combined_video(video_path):
    """Play the combined video and handle ESC key to close the window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Combined Training Video", frame)
        
        # Wait for 1 ms and check for ESC key press
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


def main_loop():
    """Main loop to process video frames continuously."""
    while True:
        try:
            print("Waiting for video frames...")
            process_video()
            combine_videos(POST_TRAINING_VIDEO_SAVE_PATH, PRE_TRAINING_VIDEO_SAVE_PATH, COMBINED_VIDEO_SAVE_PATH)

            # Delete post-training and pre-training videos after combining
            if os.path.exists(POST_TRAINING_VIDEO_SAVE_PATH):
                os.remove(POST_TRAINING_VIDEO_SAVE_PATH)
            if os.path.exists(PRE_TRAINING_VIDEO_SAVE_PATH):
                os.remove(PRE_TRAINING_VIDEO_SAVE_PATH)

            # Play the combined video
            for _ in range(2):  # Play the video twice
                play_combined_video(COMBINED_VIDEO_SAVE_PATH)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main_loop()
