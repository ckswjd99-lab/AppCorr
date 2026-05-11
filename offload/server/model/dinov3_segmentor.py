"""Backward-compatible import path for the ADE20K M2F segmentor executor."""

from .dinov3_segmentor_m2f import DINOv3SegmentorM2FExecutor


DINOv3SegmentorExecutor = DINOv3SegmentorM2FExecutor

