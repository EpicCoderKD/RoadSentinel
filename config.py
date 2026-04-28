from pathlib import Path


class ATESConfig:

    # ════════════════════════════════════════════════════════════════════════
    # PATHS
    # ════════════════════════════════════════════════════════════════════════

    MODEL_PATH        = r"E:\runs\traffic_model5\weights\best.pt"
    OUTPUT_DIR        = "ATES_output"
    OUTPUT_VIDEO_PATH = "ATES_output/output.mp4"
    CSV_DIR           = "ATES_output/csv"
    CALIBRATION_DIR   = r"E:\anaconda3\envs\ML\Traffic\ATES_output\calibration"

    # ════════════════════════════════════════════════════════════════════════
    # VEHICLE CLASSES (MUST MATCH YOUR YOLO TRAINING)
    # ════════════════════════════════════════════════════════════════════════

    VEHICLE_CLASSES = {
        0: 'car',
        1: 'threewheel',
        2: 'bus',
        3: 'truck',
        4: 'motorbike',
        5: 'van',
    }

    # ════════════════════════════════════════════════════════════════════════
    # DETECTION THRESHOLDS
    # ════════════════════════════════════════════════════════════════════════

    CONFIDENCE_THRESHOLD  = 0.45  # 0.45 is good balance
    IOU_THRESHOLD         = 0.50  # Standard IOU threshold
    MIN_BOX_WIDTH         = 60
    MIN_BOX_HEIGHT        = 60
    MAX_BOX_WIDTH         = 3000
    MAX_BOX_HEIGHT        = 3000
    MAX_BOX_ASPECT_RATIO  = 6.0

    # ════════════════════════════════════════════════════════════════════════
    # SPEED ESTIMATION
    # ════════════════════════════════════════════════════════════════════════

    SPEED_CALCULATION_METHOD = 'ensemble'  # kalman, linear, polynomial, ensemble
    MIN_TRACK_LENGTH = 8
    TRAJECTORY_ANALYSIS_POINTS = 30
    SPEED_SMOOTHING_ALPHA = 0.15  # 0.1-0.2 is best (lower = smoother)
    SPEED_BUFFER_SIZE = 30
    USE_SPEED_OUTLIER_REJECTION = True
    SPEED_OUTLIER_THRESHOLD = 3.0
    MIN_REALISTIC_SPEED = 2.0
    MAX_REALISTIC_SPEED = 150.0
    USE_PERSPECTIVE_CORRECTION = True
    PERSPECTIVE_CORRECTION_STRENGTH = 0.3  # Lower = less correction
    DEFAULT_PIXEL_TO_METER = 0.025  # Will be overridden by calibration

    # ════════════════════════════════════════════════════════════════════════
    # CALIBRATION
    # ════════════════════════════════════════════════════════════════════════

    CALIBRATION_SPEED_MIN = 10.0
    CALIBRATION_SPEED_MAX = 200.0
    CALIBRATION_AUTO_FRAMES = 300

    VEHICLE_DIMENSIONS = {
        'car': 4.5,
        'bus': 12.0,
        'truck': 8.5,
        'motorbike': 2.2,
        'threewheel': 3.0,
        'van': 5.0,
    }

    # ════════════════════════════════════════════════════════════════════════
    # TRACKING
    # ════════════════════════════════════════════════════════════════════════

    TRACKER_CONFIG = 'botsort.yaml'  # ultralytics tracker
    TRACK_LOST_THRESHOLD = 5
    REID_LOST_TIMEOUT = 90
    REID_IOU_THRESH = 0.30
    REID_FEAT_THRESH = 0.70
    REID_DIST_THRESH = 200
    TRACK_ASSIGN_MIN_SCORE = 0.20
    TRAJECTORY_MAX_LEN = 300
    SHOW_TRAJECTORY = True
    TRAJECTORY_LENGTH = 30

    # ════════════════════════════════════════════════════════════════════════
    # PLATE READING
    # ════════════════════════════════════════════════════════════════════════

    PLATE_DETECTION_INTERVAL = 5  # Check every 5 frames
    PLATE_UPSCALE = 2  # 2x upscaling (fast, good quality)
    PLATE_OCR_LANGUAGES = ['en']
    PLATE_MIN_CONFIDENCE = 0.50
    PLATE_HIGH_CONFIDENCE = 0.60
    PLATE_PREPROCESS_METHODS = 2  # 2 preprocessing methods (fast)
    PLATE_OCR_THREADS = 1

    # ════════════════════════════════════════════════════════════════════════
    # VIOLATION DETECTION
    # ════════════════════════════════════════════════════════════════════════

    SPEED_LIMITS = {
        'car': 40,
        'truck': 40,
        'bus': 40,
        'motorbike': 40,
        'threewheel': 35,
        'van': 40,
    }

    OVERSPEED_MIN_CHECK = 25  # Only check speeds above 25 km/h
    WRONG_LANE_MIN_TRAJECTORY = 20
    WRONG_LANE_MIN_DISPLACEMENT = 100
    WRONG_LANE_ANGLE_THRESHOLD = 135
    WRONG_LANE_LEARN_FRAMES = 150

    # ════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ════════════════════════════════════════════════════════════════════════

    DASHBOARD_BASE_WIDTH = 1920
    DASHBOARD_BASE_HEIGHT = 1080
    DASHBOARD_LEFT_PANEL_W = 420
    DASHBOARD_RIGHT_PANEL_W = 550
    DASHBOARD_MARGIN = 20
    DASHBOARD_ROW_HEIGHT = 48
    DASHBOARD_VEHICLE_PERSIST_FRAMES = 120
    DASHBOARD_PANEL_ALPHA = 0.82

    # Colors (BGR format for OpenCV)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_BLUE = (255, 100, 0)
    COLOR_BLACK = (0, 0, 0)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_GRAY = (128, 128, 128)
    COLOR_DARK = (20, 20, 20)
    COLOR_ACCENT = (255, 140, 0)
    COLOR_LIGHT_GRAY = (200, 200, 200)
    COLOR_DARK_RED = (0, 0, 180)
    COLOR_DARK_GREEN = (0, 180, 0)

    # ════════════════════════════════════════════════════════════════════════
    # TRAFFIC ANALYSIS
    # ════════════════════════════════════════════════════════════════════════

    DENSITY_LOW = 3
    DENSITY_MEDIUM = 7
    DENSITY_HIGH = 12
    HISTORY_MAX_VEHICLES = 20
    HISTORY_DISPLAY_FRAMES = 150

    # ════════════════════════════════════════════════════════════════════════
    # PERFORMANCE
    # ════════════════════════════════════════════════════════════════════════

    DEVICE = 0  # GPU device ID
    USE_HALF_PRECISION = True
    FRAME_SKIP = 1
    EXPECTED_SEC_PER_FRAME = 12.0

    # ════════════════════════════════════════════════════════════════════════
    # SYSTEM INFO
    # ════════════════════════════════════════════════════════════════════════

    SYSTEM_NAME = "ATES Phase-1 Integrated V1"
    SYSTEM_VERSION = "1.0.0"
    OPTIMIZED_FOR = "RTX 3050 / 4K Traffic Video"

    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("\n" + "=" * 70)
        print(f"  {cls.SYSTEM_NAME} v{cls.SYSTEM_VERSION}")
        print(f"  {cls.OPTIMIZED_FOR}")
        print("=" * 70)
        print(f"  Model              : {cls.MODEL_PATH}")
        print(f"  Confidence         : {cls.CONFIDENCE_THRESHOLD}")
        print(f"  IOU                : {cls.IOU_THRESHOLD}")
        print(f"  Min Box Size       : {cls.MIN_BOX_WIDTH}x{cls.MIN_BOX_HEIGHT}")
        print(f"  Speed Method       : {cls.SPEED_CALCULATION_METHOD}")
        print(f"  Min Track Length   : {cls.MIN_TRACK_LENGTH}")
        print(f"  Speed Smoothing    : {cls.SPEED_SMOOTHING_ALPHA}")
        print(f"  Tracker            : {cls.TRACKER_CONFIG}")
        print(f"  Plate Upscale      : {cls.PLATE_UPSCALE}x")
        print("=" * 70 + "\n")

    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        # Model path
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"MODEL_PATH not found: {cls.MODEL_PATH}")
        
        # Thresholds
        if not (0.1 <= cls.CONFIDENCE_THRESHOLD <= 0.9):
            errors.append(f"CONFIDENCE_THRESHOLD out of range: {cls.CONFIDENCE_THRESHOLD}")
        
        if not (0.1 <= cls.IOU_THRESHOLD <= 0.9):
            errors.append(f"IOU_THRESHOLD out of range: {cls.IOU_THRESHOLD}")
        
        # Speed ranges
        if cls.MIN_REALISTIC_SPEED >= cls.MAX_REALISTIC_SPEED:
            errors.append("Speed range invalid: MIN >= MAX")
        
        # Smoothing
        if not (0.0 <= cls.SPEED_SMOOTHING_ALPHA <= 1.0):
            errors.append(f"SPEED_SMOOTHING_ALPHA out of range: {cls.SPEED_SMOOTHING_ALPHA}")

        if errors:
            print("\n[Config] ⚠ VALIDATION ERRORS:")
            for e in errors:
                print(f"  ✗ {e}")
            return False

        print("[Config] ✓ All settings validated")
        return True


if __name__ == "__main__":
    ATESConfig.print_summary()
    ATESConfig.validate()
