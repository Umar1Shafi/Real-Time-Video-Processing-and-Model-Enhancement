# Real-Time-Video-Processing-and-Model-Enhancement
system that processes video frames in real-time using YOLO, improving model accuracy through dynamic retraining.

## Overview

This project focuses on real-time video inference and deployment
optimization for object detection models, with an emphasis on
latency reduction, memory efficiency, and numerical stability.

The system explores:
- ONNX export constraints for CNN-based detectors
- TensorRT engine optimization (FP16, kernel fusion)
- CUDA-based preprocessing to eliminate CPU bottlenecks
- Accuracy–latency trade-offs introduced by precision reduction

The primary objective was not model training, but understanding
how deployment constraints affect inference behavior in practice.

## Performance Summary

- Latency reduced from ~45 ms to ~18 ms per frame
- Throughput increased to ~22–28 FPS on mid-range GPU
- Memory usage reduced by approximately 30%
- Accuracy degradation remained below 1.5%

These results highlight that pipeline-level optimization often
dominates performance gains over model-level changes.

