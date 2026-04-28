# RoadSentinel
RoadSentinel is a real-time traffic enforcement system using YOLOv8, custom multi-object tracking, speed estimation (MAE 3.08 km/h), license plate OCR, and violation detection. Built for standard roadside cameras. Official implementation of our IITRAM research paper.

# RoadSentinel — Automated Traffic Enforcement System

> **ATES** &nbsp;·&nbsp; IITRAM, Ahmedabad &nbsp;·&nbsp; Computer Vision & Deep Learning

RoadSentinel is a modular, real-time traffic enforcement framework built entirely on computer vision and deep learning. It combines vehicle detection, multi-object tracking, multi-method speed estimation, wrong-lane monitoring, license plate recognition, traffic analytics, and a live 4K-aware dashboard — all inside a single unified pipeline designed to operate on consumer-grade hardware.

This repository accompanies the research paper:

> *"RoadSentinel: Real-Time Automated Traffic Enforcement Using Computer Vision"*  
> Suhani Brahmbhatt · Kathan Desai · Drashvi Thoriya · Parth Goswami  
> Prasun Chandra Tripathi · Ravi Bhandari  
> **IITRAM, Ahmedabad, India**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Module-by-Module Breakdown](#module-by-module-breakdown)
  - [1. Smart Calibration](#1-smart-calibration)
  - [2. Vehicle Detection](#2-vehicle-detection)
  - [3. Multi-Object Tracking](#3-multi-object-tracking)
  - [4. Speed Estimation](#4-speed-estimation)
  - [5. Wrong-Lane Detection](#5-wrong-lane-detection)
  - [6. License Plate Recognition](#6-license-plate-recognition)
  - [7. Violation Checker](#7-violation-checker)
  - [8. Traffic Analytics](#8-traffic-analytics)
  - [9. Dashboard UI](#9-dashboard-ui)
  - [10. CSV Exporter](#10-csv-exporter)
- [8-Step Pipeline Initialization](#8-step-pipeline-initialization)
- [Performance Results](#performance-results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Dataset](#dataset)
- [Hardware & Software Environment](#hardware--software-environment)
- [Note on Test Videos](#note-on-test-videos)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Authors](#authors)

---

## Overview

Traditional traffic enforcement relies on manual supervision and post-incident video review — an approach that is labour-intensive, difficult to scale, and largely reactive. RoadSentinel addresses this gap by providing a fully automated, real-time alternative capable of:

- Detecting and classifying six vehicle types relevant to Indian roads
- Estimating vehicle speed with a Mean Absolute Error of **3.08 km/h**
- Flagging overspeed and wrong-lane violations automatically
- Reading license plates using multi-frame OCR voting with character-level error correction
- Logging all events to six structured CSV reports with timestamps and session IDs
- Rendering a live annotated video with a 4K-aware dashboard overlay

The system auto-detects GPU at startup and falls back to CPU if none is available — no code changes required.

---

## System Architecture

```
┌─────────────┐    ┌───────────────┐    ┌───────────────────────┐    ┌────────────┐
│  Video      │───►│ Preprocessing │───►│  Vehicle Detection    │───►│  Tracking  │
│  Input      │    │ (normalize)   │    │  (YOLOv8x)            │    │ (IoU+Cent) │
└─────────────┘    └───────────────┘    └───────────────────────┘    └─────┬──────┘
                                                                            │
                    ┌───────────────────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────┐     ┌────────────────────┐     ┌────────────────────┐
         │ Speed Estimation │     │  Wrong-Lane        │     │  Traffic Analytics │
         │ (Kalman+Ensemble)│     │  Detection         │     │  (Density + Flow)  │
         └────────┬─────────┘     └────────────────────┘     └────────────────────┘
                  │
                  ▼
         ┌──────────────────┐     ┌────────────────────┐
         │ License Plate    │───► │  Violation Logging │
         │ Recognition (OCR)│     │  + CSV Report      │
         └──────────────────┘     └────────────────────┘
```

Each stage is a self-contained class, meaning individual modules can be upgraded or replaced without redesigning the rest of the pipeline.

---

## Module-by-Module Breakdown

### 1. Smart Calibration

`SmartCalibrator` establishes the pixel-to-meter conversion factor critical for accurate speed estimation. It offers two modes and presents the user with a choice at startup.

**Auto-Calibration** scans the first 300 frames of the video (every 5th frame), detects vehicles using the YOLO model, and estimates scene scale by comparing each detected bounding box height against known real-world vehicle dimensions stored in `VEHICLE_DIMENSIONS`. The median ratio across all candidate vehicles is the final `pixel_to_meter` value, reported with a confidence of 0.85.

| Vehicle | Real Height Used |
|---|---|
| car | 4.5 m × 0.7 = 3.15 m |
| bus | 12.0 m × 0.7 = 8.4 m |
| truck | 8.5 m × 0.7 = 5.95 m |
| motorbike | 2.2 m × 0.7 = 1.54 m |
| threewheel | 3.0 m × 0.7 = 2.1 m |
| van | 5.0 m × 0.7 = 3.5 m |

**Manual Calibration** opens an interactive OpenCV window on the first video frame. The user clicks two points with a known real-world distance between them. The window shows pixel distance in real-time and a reference table of common road measurements:

| Reference Object | Distance |
|---|---|
| Single lane width | 3.65 m |
| Double lane width | 7.30 m |
| Car length | 4.50 m |
| Truck length | 12.00 m |
| Road marking gap | 6.00 m |
| Road marking length | 3.00 m |
| Parking space | 5.50 m |

Controls: `Left-click` to place points · `ENTER/SPACE` to confirm · `R` to reset · `ESC` to cancel.

The calibration result (mode, pixel-to-meter, method, confidence, ISO timestamp) is saved and passed directly to the `SpeedEstimator`.

---

### 2. Vehicle Detection

`VehicleDetector` wraps a **YOLOv8x** model to localize traffic participants in each frame. The model was fine-tuned on the India Driving Dataset (IDD) and produces class labels, confidence scores, and bounding-box coordinates for six classes:

| Class ID | Class Name |
|---|---|
| 0 | car |
| 1 | threewheel |
| 2 | bus |
| 3 | truck |
| 4 | motorbike |
| 5 | van |

Each raw detection passes through bounding-box sanity checks before reaching the tracker:

| Filter | Value |
|---|---|
| Minimum box size | 60 × 60 px |
| Maximum box size | 3000 × 3000 px |
| Maximum aspect ratio | 6.0 |
| Confidence threshold | 0.45 |
| IOU threshold (NMS) | 0.50 |

The system uses `model.predict()` directly (not `model.track()`) to avoid a known `gmc_method` error in certain Ultralytics versions. Custom IoU + centroid tracking is implemented entirely within `VehicleDetector`.

---

### 3. Multi-Object Tracking

The tracker maintains persistent vehicle identities across frames using a combined IoU and centroid proximity score:

```
S = 0.6 × IoU + 0.4 × Dc
```

where `Dc = max(0, 1 − distance / MAX_MATCH_DIST)` is normalized centroid proximity. All detection-track pairs above a minimum score of 0.1 are sorted by score descending, and greedy matching is applied. Unmatched detections are assigned new track IDs.

**Track lifecycle parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| MAX_MATCH_DIST | 300 px | Maximum centroid distance allowed for matching |
| IOU_MATCH_THRESH | 0.30 | Minimum IoU for box-level matching |
| MAX_MISS_FRAMES | 25 | Frames before a lost track is dropped |
| MIN_FRAMES_COUNT | 30 | Minimum tracked frames before vehicle is counted |
| MIN_DIST_COUNT | 80 px | Minimum displacement before vehicle is counted |

A vehicle is added to the official count only after being tracked for ≥ 30 frames **and** having moved ≥ 80 pixels, preventing stationary false positives from inflating vehicle counts.

Trajectory history (bottom-center coordinates, used for speed calculation) is stored per track up to 200 points. Bounding box history (last 10 frames) is kept for IoU matching.

---

### 4. Speed Estimation

`SpeedEstimator` supports four calculation methods, selected via `SPEED_CALCULATION_METHOD` in `config.py`. The default is `ensemble`, which runs all three analytical approaches and returns their median:

**Kalman Filter** — A 4-state Kalman filter (position x, position y, velocity x, velocity y) with a constant-velocity motion model is initialized per track. Speed is derived from the velocity state components `vx`, `vy` after each predict-update cycle.

**Linear Regression** — `scipy.stats.linregress` is applied to the last 30 trajectory points in x and y separately, and the combined slope magnitude gives the pixel velocity.

**Polynomial Fitting** — A degree-2 polynomial is fitted to trajectory coordinates via `numpy.polyfit`. The derivative evaluated at the latest frame gives instantaneous pixel velocity.

**Pixel velocity → km/h conversion:**

```
speed (km/h) = pixel_velocity × perspective_factor × pixel_to_meter × fps × 3.6
```

**Perspective correction** compensates for depth distortion. Vehicles appearing higher in the frame (further from camera) receive a larger correction factor:

```
correction = 1.0 + (1.0 − normalized_y) × PERSPECTIVE_CORRECTION_STRENGTH
```

clipped to [0.8, 2.5]. `PERSPECTIVE_CORRECTION_STRENGTH` defaults to 0.3.

**Post-processing per estimate (in order):**
1. Reject if outside realistic range: 2.0 – 150.0 km/h
2. Z-score outlier rejection against per-track speed history (threshold: 3.0σ, requires ≥ 5 samples)
3. Exponential smoothing: `vt = α·xt + (1 − α)·vt−1` with α = 0.15 (lower = smoother)

Final reported speed per track = **median** of the smoothed speed buffer (up to last 30 frames).

---

### 5. Wrong-Lane Detection

Tracked trajectories are compared against the dominant traffic flow direction learned from the first frames of video.

| Parameter | Value |
|---|---|
| WRONG_LANE_MIN_TRAJECTORY | 20 points minimum before analysis |
| WRONG_LANE_MIN_DISPLACEMENT | 100 px minimum movement to trigger check |
| WRONG_LANE_ANGLE_THRESHOLD | 135° deviation from expected flow |
| WRONG_LANE_LEARN_FRAMES | 150 frames used to learn dominant direction |

Any track whose average trajectory direction deviates more than 135° from the learned dominant flow, while having moved more than 100 px over at least 20 trajectory points, is flagged as a wrong-lane violation.

---

### 6. License Plate Recognition

`UniversalPlateReader` uses **EasyOCR** with a custom pre-processing and multi-frame voting pipeline optimized for real-world roadside conditions.

**Pre-processing pipeline** (two methods run in parallel on each crop):

- *Adaptive method:* Gaussian denoising (`fastNlMeansDenoising`, h=8) → CLAHE (clipLimit=3.0, tileGrid=8×8) → Adaptive Gaussian thresholding (block=13, C=3)
- *Gamma method:* Gamma correction (γ=1.3, LUT-based) → CLAHE (clipLimit=5.0, tileGrid=5×5) → Otsu thresholding

Each vehicle crop is upscaled 2× via Lanczos4 interpolation (for boxes narrower than 80 px) before pre-processing. Both the binary and grayscale outputs of each method are passed to OCR — up to **4 image variants** per plate reading attempt.

OCR is restricted to the character allowlist `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`.

**Character confusion correction** applies position-aware substitution. In the first half of the plate (expected letters), ambiguous digit characters are replaced with their letter counterparts; in the second half (expected digits), the reverse is applied:

| Pair |
|---|
| 0 ↔ O |
| 1 ↔ I / L |
| 5 ↔ S |
| 8 ↔ B |
| 2 ↔ Z |
| 6 ↔ G |

**Plate validation** uses four regex patterns covering Indian plate formats (e.g., `GJ05AB1234`, `GJ05A1234`). Any reading shorter than 4 or longer than 12 alphanumeric characters, or containing only letters or only digits, is rejected.

**Multi-frame voting** accumulates all valid readings for a track across the session. The final plate is assembled character-by-character: readings are filtered to the most common length, then a majority vote is taken at each character position. Final confidence = average vote fraction per character.

The plate reader runs in a background thread with a lock (`threading.Lock`) to avoid blocking the main processing loop.

---

### 7. Violation Checker

`ViolationChecker` evaluates two violation types per tracked vehicle per frame:

**Overspeed** — Only checked if estimated speed exceeds `OVERSPEED_MIN_CHECK` (25 km/h). Violation confidence scales linearly with excess speed above the limit (0 → 100% over a 20 km/h excess range):

| Class | Speed Limit |
|---|---|
| car | 40 km/h |
| truck | 40 km/h |
| bus | 40 km/h |
| motorbike | 40 km/h |
| threewheel | 35 km/h |
| van | 40 km/h |

Example violation message: `OVERSPEED 52kmh (limit:40)`

**Wrong-lane** — Flagged based on trajectory direction analysis described in Section 5.

Each violation record includes: track ID, frame number, vehicle class, violation type, descriptive message, confidence score, speed reading, and ISO 8601 timestamp.

---

### 8. Traffic Analytics

`TrafficAnalyzer` estimates real-time traffic density and dominant flow direction from the active track set.

**Density levels** based on simultaneously active tracks:

| Active Vehicles | Level |
|---|---|
| ≤ 3 | Low |
| 4 – 7 | Medium |
| 8 – 12 | High |
| > 12 | Very High |

**Flow direction** is derived from displacement vectors of the 10 most recently updated tracks (minimum 20 px movement, over a rolling buffer of 100 direction samples). The mean angle is mapped to a cardinal direction: East, South, West, or North.

---

### 9. Dashboard UI

`Dashboard` renders a real-time overlay onto every output frame. The scale factor is computed as `min(width/1920, height/1080)` so all panel geometry, fonts, and line thicknesses scale proportionally across any resolution including 4K.

**Left panel** (420 px base width, semi-transparent dark background, α = 0.82):
- System name and version
- Current frame number and compute device
- Active track count, total vehicles counted, unique IDs
- Per-class vehicle breakdown
- Traffic density level and flow direction

**Right panel** (550 px base width) — live vehicle table with columns `ID | TYPE | SPEED | PLATE | STATUS`:
- Vehicles with active violations appear in **red-highlighted rows**
- Compliant vehicles alternate between dark green and transparent rows
- Violations are always sorted to the top of the table
- Vehicle entries persist for 120 frames after last detection

**Speed colour coding in the table:**

| Speed | Colour |
|---|---|
| < 35 km/h | Green |
| 35 – 50 km/h | Orange |
| > 50 km/h | Red |

**Plate text colour coding:**
- Cyan = confidence > 0.60 (high confidence)
- Light grey = confidence ≤ 0.60 (low confidence)
- Grey = still scanning

**On-frame bounding boxes:**
- Green border = compliant · Red border (thicker) = violation
- Label above box: `ID:{id} {class} {speed:.1f}km/h`
- Plate text drawn 30 px below box
- Violation text (e.g., `! OVERSPEED 52kmh`) drawn below plate

All text uses drop-shadow rendering (2 px offset black background) for readability on any background.

---

### 10. CSV Exporter

`CSVExporter` writes six timestamped CSV files at session end. All write operations are thread-safe via `threading.Lock`.

| File | Contents |
|---|---|
| `*_detections.csv` | Per-frame detection: track ID, frame, class, bbox coords, bbox area, confidence, speed, raw plate |
| `*_violations.csv` | Per-violation: track ID, frame, class, violation type, message, confidence, speed, timestamp |
| `*_plates_raw.csv` | Every individual OCR reading per track with confidence score |
| `*_speed_summary.csv` | Per-track aggregated speed stats: avg, min, max, std dev, sample count, best plate (voted), vehicle class |
| `*_frame_stats.csv` | Per-frame traffic state: active count, density level, flow direction, violation count, timestamp |
| `*_summary.csv` | Session-level totals: session ID, start/end time, total detections, unique vehicles, unique plates, total violations, avg/max/min speed across all vehicles |

All filenames follow the pattern: `ates_traffic_YYYYMMDD_HHMMSS_<type>.csv`

---

## 8-Step Pipeline Initialization

When `ATESPipelineIntegrated` is instantiated, it runs the following sequence before frame processing begins:

```
[STEP 1/8]  VideoHandler            — open video, read resolution, FPS, total frames
[STEP 2/8]  SmartCalibrator         — auto or manual calibration → pixel_to_meter ratio
[STEP 3/8]  VehicleDetector         — load YOLOv8x weights, initialize tracking state
[STEP 4/8]  SpeedEstimator          — configure FPS, pixel_to_meter, frame dimensions
[STEP 5/8]  UniversalPlateReader    — initialize EasyOCR on GPU or CPU
[STEP 6/8]  TrafficAnalyzer         — initialize flow direction and density buffers
            ViolationChecker        — load speed limits and thresholds from config
[STEP 7/8]  Dashboard               — compute scale factor, initialize panel geometry
[STEP 8/8]  CSVExporter             — create output directories, generate session ID
            VideoHandler (writer)   — initialize MP4 writer for annotated output
```

---

## Performance Results

Evaluated on benchmark traffic videos covering urban roads, campus roads, and mixed-traffic scenarios under varied lighting, vehicle density, and scene perspective conditions.

| Metric | Value |
|---|---|
| Precision | 68.9% |
| Recall | 100.0% |
| F1-Score | 81.6% |
| mAP | 84.2% |
| MOTA | 54.9% |
| MOTP | 82.3% |
| Speed MAE | 3.08 km/h |
| Speed RMSE | 4.50 km/h |
| Plate Accuracy | 52.6% |
| Average FPS | 1.4 |

Perfect recall (100.0%) means vehicles were almost never missed during evaluation. Moderate precision (68.9%) reflects some false positives in dense, heavily occluded scenes. The 3.08 km/h speed MAE was additionally validated in a controlled campus experiment with 4 two-wheelers and 1 passenger car driven at known speedometer-recorded speeds through the monitored zone.

---

## Project Structure

```
RoadSentinel/
├── main_integrated.py          # Complete pipeline — all 10 modules in one file
│   ├── SpeedEstimator          #  Module 1 — multi-method speed estimation
│   ├── VehicleDetector         #  Module 2 — YOLOv8x detection + custom tracking
│   ├── SmartCalibrator         #  Module 3 — auto/manual pixel-to-meter calibration
│   ├── UniversalPlateReader    #  Module 4 — EasyOCR with voting pipeline
│   ├── TrafficAnalyzer         #  Module 5 — density & flow direction
│   ├── ViolationChecker        #  Module 6 — speed limit & wrong-lane checks
│   ├── Dashboard               #  Module 7 — 4K-aware overlay renderer
│   ├── VideoHandler            #  Module 8 — video I/O wrapper
│   ├── CSVExporter             #  Module 9 — six-file thread-safe CSV logger
│   └── ATESPipelineIntegrated  #  Module 10 — master orchestrator
├── config.py                   # ATESConfig — all tunable parameters in one place
├── ATES_output/
│   ├── ates_YYYYMMDD_HHMMSS.mp4
│   └── csv/
│       ├── ates_traffic_*_detections.csv
│       ├── ates_traffic_*_violations.csv
│       ├── ates_traffic_*_plates_raw.csv
│       ├── ates_traffic_*_speed_summary.csv
│       ├── ates_traffic_*_frame_stats.csv
│       └── ates_traffic_*_summary.csv
└── README.md
```

---

## Requirements

**Python:** 3.8 or higher

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8x model loading and inference |
| `opencv-python` | Video I/O, frame rendering, dashboard overlay |
| `torch` | Deep learning backend (CUDA GPU recommended) |
| `easyocr` | License plate OCR |
| `filterpy` | Kalman filter for speed estimation |
| `scipy` | Linear and polynomial regression for speed |
| `numpy` | Numerical computation throughout |
| `pandas` | CSV construction and speed aggregation |
| `tqdm` | Progress bar during video processing |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/RoadSentinel.git
cd RoadSentinel

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install core dependencies
pip install ultralytics opencv-python easyocr filterpy scipy numpy pandas tqdm

# 4. Install PyTorch with CUDA (for GPU acceleration)
# Find the right command for your CUDA version at: https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Configuration

All system parameters are centralized in `config.py` inside `ATESConfig`. No command-line arguments are needed for configuration — edit the class attributes directly.

**Required before first run:**

```python
# Path to your trained YOLOv8x weights
MODEL_PATH = r"path/to/your/best.pt"
```

**Key parameters to review:**

```python
# Speed estimation method — 'ensemble' is recommended
SPEED_CALCULATION_METHOD = 'ensemble'   # kalman | linear | polynomial | ensemble

# Per-class speed limits (km/h)
SPEED_LIMITS = {
    'car': 40, 'truck': 40, 'bus': 40,
    'motorbike': 40, 'threewheel': 35, 'van': 40,
}

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD        = 0.50

# Realistic speed range — readings outside this are silently discarded
MIN_REALISTIC_SPEED = 2.0     # km/h
MAX_REALISTIC_SPEED = 150.0   # km/h

# Exponential smoothing factor (0.1–0.2 recommended; lower = smoother)
SPEED_SMOOTHING_ALPHA = 0.15

# GPU device ID — auto-falls back to CPU if unavailable
DEVICE = 0
```

Validate your configuration before running:

```bash
python config.py
```

This prints a parameter summary and checks for a missing model file, out-of-range thresholds, and an invalid speed range.

---

## Usage

```bash
python main_integrated.py <path_to_video>
```

**Example:**

```bash
python main_integrated.py traffic.mp4
```

On startup the system will:
1. Auto-detect GPU or fall back to CPU and print the result
2. Run the 8-step initialization sequence
3. Present a calibration menu (auto or manual mode)
4. Begin frame-by-frame processing with a tqdm progress bar in the terminal
5. Open a resized OpenCV display window (scaled to 90% of your screen)

**Keyboard controls during playback:**

| Key | Action |
|---|---|
| `P` | Pause / Resume |
| `Q` or `Esc` | Quit and flush all CSV outputs |

On quit or end of video, all six CSV files are saved, and a session summary (total detections, violations, speed stats) is printed to the console.

---

## Output Files

All outputs are written to `ATES_output/` automatically on every run:

```
ATES_output/
├── ates_20240615_143022.mp4                               # Annotated output video (mp4v codec)
└── csv/
    ├── ates_traffic_20240615_143022_detections.csv        # Per-frame detection records
    ├── ates_traffic_20240615_143022_violations.csv        # All flagged violations
    ├── ates_traffic_20240615_143022_plates_raw.csv        # Raw per-reading OCR data
    ├── ates_traffic_20240615_143022_speed_summary.csv     # Per-track speed statistics
    ├── ates_traffic_20240615_143022_frame_stats.csv       # Per-frame traffic state
    └── ates_traffic_20240615_143022_summary.csv           # Session-level totals
```

---

## Dataset

The YOLO model was trained on the **India Driving Dataset (IDD)**, a benchmark built for complex and unstructured Indian traffic scenarios, chosen specifically because conventional Western datasets underrepresent vehicle types and road conditions common in India.

**Key characteristics:**
- Diverse urban and peri-urban road scenes with heterogeneous traffic
- Partial occlusions, mixed lane usage, and varying illumination
- Includes three-wheelers (auto-rickshaws) — a class absent from most global datasets

**Training setup:**
- Classes: car, three-wheeler, bus, truck, motorbike, van
- Input resolution: 640 × 640 px
- Augmentation: horizontal flip, scaling, translation, mosaic

---

## Hardware & Software Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5 |
| RAM | 8 GB |
| GPU | NVIDIA RTX 3050 |
| OS | Windows (tested) |
| Language | Python 3.x |
| Detection | Ultralytics YOLOv8x |
| OCR | EasyOCR |
| Video I/O | OpenCV |
| Speed Filtering | SciPy + FilterPy |
| Data Export | Pandas |
| Deep Learning | PyTorch |

The system auto-detects the best available device at startup via `torch.cuda.is_available()`. No manual switching between CPU and GPU is required.

---

## Note on Test Videos

> **Test videos and sample output files are not included in this repository.**

The speed validation experiment and all primary performance evaluations were conducted using surveillance footage recorded within the IITRAM campus premises. These recordings contain identifiable individuals, vehicles, and location-specific information.

**Due to privacy concerns and institutional data policy, we are not permitted to publicly distribute this footage.** Releasing campus surveillance recordings would violate the privacy of individuals captured on camera, and is inconsistent with responsible and ethical data handling practices.

If you would like to test RoadSentinel on your own setup, we recommend using publicly available traffic datasets such as:

- [UA-DETRAC](https://detrac-db.rit.albany.edu/) — multi-vehicle detection and tracking benchmark
- [CityFlow](https://www.aicitychallenge.org/) — city-scale multi-camera traffic dataset
- [MIO-TCD](http://podoce.dinf.usherbrooke.ca/challenge/dataset/) — motorway and intersection dataset

You may also record your own footage — please ensure you have appropriate consent and comply with surveillance laws applicable in your region.

---

## Limitations

- Average processing speed of **1.4 FPS** on the tested RTX 3050 hardware limits real-time deployment at full 4K resolution
- Detection accuracy can degrade in adverse conditions: rain, fog, glare, strong shadows, or night scenes without infrared illumination
- Heavy vehicle overlap and prolonged occlusion in dense traffic can cause identity switches or trajectory fragmentation, reflected in the MOTA score of 54.9%
- Speed estimation accuracy is dependent on accurate camera placement and scene calibration — camera tilt, unstable mounting, or significant perspective distortion introduce proportional speed error
- Wrong-lane detection assumes the dominant permitted traffic direction can be reliably learned from the first 150 frames; roads with ambiguous lane markings or frequently changing flow patterns may require manual region configuration
- License plate recognition (52.6% character-level accuracy) is sensitive to motion blur, low resolution, dirty or damaged plates, and extreme viewing angles — it is the most challenging module under unconstrained roadside conditions
- Current violation catalogue covers only **overspeeding** and **wrong-lane behaviour**; other common violations are not yet implemented

---

## Future Work

- **Expanded violations:** helmet non-compliance, red-light jumping, seatbelt monitoring, mobile phone usage while driving, illegal parking detection
- **Night-time and adverse weather robustness:** infrared imaging, low-light enhancement, dehazing, and domain-adaptive training on challenging environmental data
- **Improved OCR:** dedicated plate super-resolution models and transformer-based OCR (e.g., TrOCR) for blurred or distant plates
- **Advanced tracking:** StrongSORT, OC-SORT, or transformer-based association to reduce identity switches in dense scenes
- **Edge deployment:** optimization for NVIDIA Jetson devices (Nano / Orin) for cost-effective large-scale field rollout without server-grade hardware
- **Multi-camera coordination:** consistent vehicle tracking across multiple intersections or road segments for city-scale monitoring
- **Backend integration:** centralized database connectivity, automated challan generation, and live analytics dashboards for municipal traffic authorities

---

## Authors

| Name | Role | Email |
|---|---|---|
| Suhani Brahmbhatt | Author | Suhani.Brahmbhatt.23co@iitram.ac.in |
| Kathan Desai | Author | Kathan.Desai.23co@iitram.ac.in |
| Drashvi Thoriya | Author | Drashvi.Thoriya.23co@iitram.ac.in |
| Parth Goswami | Author | Parthgiri.23co@iitram.ac.in |
| Prasun Chandra Tripathi | Guide | prasunchandratripathi@iitram.ac.in |
| Ravi Bhandari | Guide | ravibhandari@iitram.ac.in |

**Institution:** Institute of Infrastructure, Technology, Research and Management (IITRAM), Ahmedabad, Gujarat, India
