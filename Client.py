import cv2
import socket
import numpy as np
import time

# Constants
tcp_ip = "192.168.1.3"  # Replace with your receiver IP address
tcp_port = 5005

# Open video file or capture device
video_path = "input.mp4"  # Change this to your video file path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Create TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Attempt to connect to the receiver
while True:
    try:
        sock.connect((tcp_ip, tcp_port))
        print(f"Connected to {tcp_ip}:{tcp_port}...")
        break
    except socket.error:
        print(f"Failed to connect to {tcp_ip}:{tcp_port}, retrying...")
        time.sleep(1)  # Wait for 1 second before retrying

# Once connected, send video frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Resize frame if needed
        frame = cv2.resize(frame, (640, 480))  # Resize to match the receiver's frame shape

        # Encode frame
        encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()

        # Send frame size
        frame_size = len(encoded_frame).to_bytes(4, byteorder='big')
        sock.sendall(frame_size)

        # Send frame data
        sock.sendall(encoded_frame)

        print(f"Frame sent, size: {len(encoded_frame)} bytes.")

finally:
    sock.close()
    cap.release()

print("Video transmission complete.")