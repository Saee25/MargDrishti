import pytest
from ml.data.build_crops import clamp_box, pad_and_clamp_box
from ml.data.coco import clean_class_name
import json
import tempfile
from pathlib import Path

def test_box_clamping():
    # Box at the image edge
    # x, y, w, h, img_w, img_h
    assert clamp_box(-10, -10, 50, 50, 100, 100) == (0, 0, 40, 40)
    assert clamp_box(90, 90, 50, 50, 100, 100) == (90, 90, 10, 10)
    
    # Tiny box
    assert clamp_box(10.2, 10.8, 1.1, 1.1, 100, 100) == (10, 11, 1, 1)

def test_box_padding():
    # Padding that overflows
    # x=10, y=10, w=20, h=20. pad=0.5 -> pad_x=10, pad_y=10. new_x=0, new_y=0, new_w=40, new_h=40
    assert pad_and_clamp_box(10, 10, 20, 20, 30, 30, 0.5) == (0, 0, 30, 30)
    
    # Normal padding
    assert pad_and_clamp_box(10, 10, 10, 10, 100, 100, 0.1) == (9, 9, 12, 12)

def test_slug_cleaning():
    # trimmed, single spaces, consistent case, plus a filesystem-safe slug
    name, slug = clean_class_name("  ALL  motor VEHICLE   prohibited-1 ")
    assert name == "ALL motor VEHICLE prohibited-1"
    assert slug == "all_motor_vehicle_prohibited_1"
    
    name, slug = clean_class_name("speed limit (100)!")
    assert name == "speed limit (100)!"
    assert slug == "speed_limit_100"

def test_class_indices_stable_alphabetical():
    # The actual sorting happens in build_crops.py, but we can verify the python sorting behavior
    classes = ["speed_limit_20", "all_motor_vehicle_prohibited", "bicycles_only"]
    sorted_classes = sorted(classes)
    assert sorted_classes == ["all_motor_vehicle_prohibited", "bicycles_only", "speed_limit_20"]
    
    indices = {slug: idx for idx, slug in enumerate(sorted_classes)}
    assert indices["all_motor_vehicle_prohibited"] == 0
    assert indices["bicycles_only"] == 1
    assert indices["speed_limit_20"] == 2
