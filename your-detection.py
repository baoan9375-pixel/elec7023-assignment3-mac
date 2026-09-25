#!/usr/bin/env python3
"""ELEC7023 Assignment 3: two-image SSD MobileNet v2 detection on macOS.

This is a CPU/OpenCV adaptation of Lecture 4's Jetson detectNet exercise.
It uses the official TensorFlow 2018-03-29 COCO SSD MobileNet v2 weights,
not Jetson's TensorRT runtime. Run from assignment3-mac after downloading
the model described in README.md.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
MODEL = BASE / 'models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
CONFIG = BASE / 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
LABELS = BASE / 'models/mscoco_label_map.pbtxt'
FIELDS = ('ClassID', 'Confidence', 'Left', 'Top', 'Right', 'Bottom',
          'Width', 'Height', 'Area', 'Center')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_labels(path):
    content = path.read_text(encoding='utf-8')
    labels = {}
    for block in re.findall(r'item\s*\{([^{}]*)\}', content):
        class_id = re.search(r'\bid:\s*(\d+)', block)
        name = re.search(r'\bdisplay_name:\s*"([^"]+)"', block)
        if class_id and name:
            labels[int(class_id.group(1))] = name.group(1)
    if not labels or len(labels) < 80:
        raise ValueError('COCO label map is missing or incomplete')
    return labels


def detection_record(class_id, name, confidence, normalized_box, image_width, image_height):
    """Translate [left, top, right, bottom] in 0..1 to image pixel space."""
    values = (confidence, *normalized_box)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('Detector returned a non-finite score or coordinate')
    if not 0 <= confidence <= 1 or image_width <= 0 or image_height <= 0:
        raise ValueError('Invalid score or image dimensions')
    left, top, right, bottom = normalized_box
    if left > right or top > bottom:
        raise ValueError('Detector returned a reversed box')
    # The detector can return a box slightly outside [0, 1]. Limit it to the
    # actual image before measuring area or drawing its visible edges.
    left = min(1.0, max(0.0, float(left))) * image_width
    right = min(1.0, max(0.0, float(right))) * image_width
    top = min(1.0, max(0.0, float(top))) * image_height
    bottom = min(1.0, max(0.0, float(bottom))) * image_height
    width = right - left
    height = bottom - top
    return {'ClassID': int(class_id), 'ClassName': name,
            'Confidence': float(confidence), 'Left': left, 'Top': top,
            'Right': right, 'Bottom': bottom, 'Width': width,
            'Height': height, 'Area': width * height,
            'Center': [(left + right) / 2, (top + bottom) / 2]}


def select_detection(rows, class_name):
    matches = [row for row in rows if row['ClassName'].casefold() == class_name.casefold()]
    return max(matches, key=lambda row: row['Confidence']) if matches else None


def annotate(image, rows, chosen):
    canvas = image.copy()
    if chosen is not None:
        color = (40, 190, 20)
        p1 = (round(chosen['Left']), round(chosen['Top']))
        p2 = (round(chosen['Right']), round(chosen['Bottom']))
        cv2.rectangle(canvas, p1, p2, color, 3)
        title = '{} {:.1f}%'.format(chosen['ClassName'], chosen['Confidence'] * 100)
        y = min(image.shape[0] - 6, max(20, p1[1] - 7))
        cv2.putText(canvas, title, (max(0, p1[0]), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def run(images, class_name, threshold, output_dir):
    labels = load_labels(LABELS)
    if not MODEL.exists() or not CONFIG.exists():
        raise FileNotFoundError('Model files missing; follow README.md download steps')
    net = cv2.dnn.readNetFromTensorflow(str(MODEL), str(CONFIG))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {
        'model': 'TensorFlow SSD MobileNet v2 COCO 2018-03-29',
        'runtime': 'OpenCV {} on macOS CPU'.format(cv2.__version__),
        'model_sha256': digest(MODEL), 'config_sha256': digest(CONFIG),
        'label_map_sha256': digest(LABELS),
        'threshold': threshold, 'selected_class': class_name,
        'selection': 'Highest-confidence instance of selected class per image',
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'images': [], 'complete': False,
    }
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not read image: {}'.format(path))
        height, width = image.shape[:2]
        # OpenCV's published TensorFlow object detection example uses this
        # exact 300x300 RGB input and [class, score, left, top, right, bottom].
        net.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
        predictions = net.forward()
        if predictions.ndim != 4 or predictions.shape[-1] != 7:
            raise ValueError('Unexpected detector output dimensions: {}'.format(predictions.shape))
        rows = []
        for prediction in predictions[0, 0]:
            score = float(prediction[2])
            if score < threshold:
                continue
            class_id = int(prediction[1])
            if class_id not in labels:
                raise ValueError('ClassID {} absent from model label map'.format(class_id))
            coords = tuple(float(x) for x in prediction[3:7])
            rows.append(detection_record(class_id, labels[class_id], score,
                                         coords, width, height))
        chosen = select_detection(rows, class_name)
        output_name = 'image_{}_detected.jpg'.format(index)
        annotated = annotate(image, rows, chosen)
        if not cv2.imwrite(str(output_dir / output_name), annotated):
            raise OSError('Failed to write {}'.format(output_name))
        item = {
            'input': path.name, 'input_sha256': digest(path),
            'width': width, 'height': height, 'annotated': output_name,
            'all_detections': rows, 'selected_detection': chosen,
        }
        results['images'].append(item)
        results['complete'] = len(results['images']) == 2 and all(
            image_result['selected_detection'] is not None
            for image_result in results['images'])
        (output_dir / 'results.json').write_text(
            json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        print('{}: {} detections; selected {} ({})'.format(
            path.name, len(rows), class_name,
            '{:.3f}'.format(chosen['Confidence']) if chosen else 'NONE'))
    print('Results: {}'.format(output_dir))
    return 0 if results['complete'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs=2, type=Path,
                        default=[BASE / 'images/dog_0.jpg', BASE / 'images/dog_2.jpg'],
                        metavar=('IMAGE1', 'IMAGE2'))
    parser.add_argument('--class-name', default='dog')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0 < args.threshold <= 1 or not math.isfinite(args.threshold):
        parser.error('Threshold must be within (0, 1]')
    if not args.class_name.strip():
        parser.error('Class name cannot be blank')
    images = [path.expanduser().resolve() for path in args.images]
    if any(not path.is_file() for path in images):
        parser.error('Both input image files must exist')
    if digest(images[0]) == digest(images[1]):
        parser.error('Two distinct input images are required')
    output_dir = args.output_dir or (BASE / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    try:
        return run(images, args.class_name.strip(), args.threshold, output_dir.resolve())
    except (OSError, ValueError, cv2.error) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
