# ELEC7023 Assignment 3 — macOS execution

`your-detection.py` performs real inference on two supplied photographs using
TensorFlow's COCO SSD MobileNet v2 (2018-03-29) model through OpenCV 4.11 CPU.
It generates two annotated images and `results.json`, from which the PDF report
is built. It is an adaptation of Lecture 4's Jetson workflow and must be
described as such. It does not claim to have used a Jetson or TensorRT.

The final run uses `dog_0.jpg` and `dog_2.jpg` from the upstream
[jetson-inference sample images](https://github.com/dusty-nv/jetson-inference/tree/master/data/images).
The images are public sample inputs; all bounding boxes and scores are actual
outputs of this Mac run. Only the selected dog is drawn in green; all detections
are retained in results.json.

## Reproduce the run

From this folder with Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install opencv-python-headless==4.11.0.86 numpy==2.2.6
```

Download the official [model weights](https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz),
[OpenCV graph config](https://github.com/opencv/opencv_extra/blob/4.x/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt),
and [COCO label map](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)
to the paths in `models/`. Extract only `frozen_inference_graph.pb` from the
model archive to `models/ssd_mobilenet_v2_coco_2018_03_29/`.

```bash
.venv/bin/python your-detection.py
```

The script creates `results/<timestamp>/` containing `results.json` and two
annotated JPEGs. It exits with code 2 if either image has no dog above the
0.50 threshold, rather than inventing or substituting a detection.

`build_report.py` requires a real GitHub URL to the uploaded code:

```bash
.venv/bin/python build_report.py \
  --results results/<timestamp>/results.json \
  --github-url https://github.com/USER/REPOSITORY/blob/main/your-detection.py
```

The report states the actual Mac runtime and image sources. The course's
Moodle submission asks for the code on GitHub and a PDF/Word report containing
the link, detected images and all ten fields. The professor may evaluate the
choice of Mac rather than Jetson differently; the assignment text itself
specifies the outputs but does not explicitly mandate the hardware.
