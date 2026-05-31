import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TEMPLATE_PATH = Path(r"C:\Users\konsy\Documents\pythonなどツール\yokusoku_template.png")
DEFAULT_SOURCE = "camera"
DEFAULT_DETECTION_MODE = "both"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720

MATCH_THRESHOLD = 0.88
GREEN_RATIO_THRESHOLD = 0.60
FRAME_RED_SCORE_THRESHOLD = 0.60
INTERVAL_SEC = 0.20
MIN_SCALE = 0.50
MAX_SCALE = 2.50
SCALE_STEP = 0.05
PERSPECTIVE_MATCH_THRESHOLD = 0.70
MIN_CANDIDATE_AREA = 1000
MAX_CANDIDATES = 12
CANDIDATE_CROP_MARGIN = 0.30


@dataclass
class Detection:
    detected: bool
    score: float
    green_ratio: float
    scale: float
    mode: str
    left: int
    top: int
    width: int
    height: int
    corners: np.ndarray = None
    elapsed_ms: float = 0.0


@dataclass
class TemplateVariant:
    scale: float
    bgr: np.ndarray
    gray: np.ndarray


def load_template(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    template = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"テンプレート画像を読み込めません: {path}")
    return template


def equalized_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def make_green_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 60, 40], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def build_template_variants(
    template_bgr: np.ndarray,
    min_scale: float,
    max_scale: float,
    scale_step: float,
) -> list[TemplateVariant]:
    if min_scale <= 0 or max_scale <= 0 or scale_step <= 0:
        raise ValueError("scale values must be positive.")
    if min_scale > max_scale:
        raise ValueError("min_scale must be less than or equal to max_scale.")

    variants = []
    base_h, base_w = template_bgr.shape[:2]
    scale_count = int(round((max_scale - min_scale) / scale_step)) + 1

    for i in range(scale_count):
        scale = min_scale + scale_step * i
        width = max(1, int(round(base_w * scale)))
        height = max(1, int(round(base_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(template_bgr, (width, height), interpolation=interpolation)
        variants.append(
            TemplateVariant(
                scale=scale,
                bgr=resized,
                gray=equalized_gray(resized),
            )
        )

    return variants


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, getattr(cv2, "CAP_DSHOW", 0))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: index={index}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def capture_camera(cap: cv2.VideoCapture, mirror: bool) -> tuple[np.ndarray, dict]:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("カメラ画像を取得できませんでした。")

    if mirror:
        frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    return frame, {"left": 0, "top": 0, "width": width, "height": height}


def capture_screen(sct) -> tuple[np.ndarray, dict]:
    monitor = sct.monitors[0]
    shot = sct.grab(monitor)
    img = np.array(shot)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), monitor


def calc_green_ratio(bgr: np.ndarray) -> float:
    mask = make_green_mask(bgr)
    return np.count_nonzero(mask) / mask.size


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def find_green_quad_candidates(
    screen_bgr: np.ndarray,
    min_area: int,
    max_candidates: int,
) -> list[np.ndarray]:
    mask = make_green_mask(screen_bgr)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screen_area = screen_bgr.shape[0] * screen_bgr.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > screen_area * 0.80:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width < 20 or height < 20:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            corners = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

        candidates.append((area, order_points(corners)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [corners for _, corners in candidates[:max_candidates]]


def warp_candidate(
    screen_bgr: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
    corners_are_ordered: bool = False,
) -> np.ndarray:
    src = corners.astype(np.float32) if corners_are_ordered else order_points(corners)
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(screen_bgr, matrix, (width, height))


def calc_same_size_score(image_bgr: np.ndarray, template_gray: np.ndarray) -> float:
    image_gray = equalized_gray(image_bgr)
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def calc_best_match(
    screen_bgr: np.ndarray,
    template_variants: list[TemplateVariant],
) -> tuple[float, tuple[int, int], TemplateVariant]:
    screen_gray = equalized_gray(screen_bgr)
    screen_h, screen_w = screen_gray.shape[:2]

    best_score = -1.0
    best_loc = (0, 0)
    best_variant = template_variants[0]

    for variant in template_variants:
        template_h, template_w = variant.gray.shape[:2]
        if template_w > screen_w or template_h > screen_h:
            continue

        result = cv2.matchTemplate(screen_gray, variant.gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = float(max_val)
            best_loc = max_loc
            best_variant = variant

    return best_score, best_loc, best_variant


def detect_anywhere(
    screen_bgr: np.ndarray,
    template_variants: list[TemplateVariant],
    monitor: dict,
    match_threshold: float,
    green_threshold: float,
) -> Detection:
    score, (x, y), variant = calc_best_match(screen_bgr, template_variants)
    template_h, template_w = variant.bgr.shape[:2]

    matched = screen_bgr[y : y + template_h, x : x + template_w]
    green_ratio = calc_green_ratio(matched)
    detected = score >= match_threshold and green_ratio >= green_threshold

    return Detection(
        detected=detected,
        score=score,
        green_ratio=green_ratio,
        scale=variant.scale,
        mode="scale",
        left=monitor["left"] + x,
        top=monitor["top"] + y,
        width=template_w,
        height=template_h,
    )


def detect_with_perspective(
    screen_bgr: np.ndarray,
    template_bgr: np.ndarray,
    monitor: dict,
    match_threshold: float,
    green_threshold: float,
    min_candidate_area: int,
    max_candidates: int,
) -> Detection:
    template_h, template_w = template_bgr.shape[:2]
    template_gray = equalized_gray(template_bgr)
    candidates = find_green_quad_candidates(screen_bgr, min_candidate_area, max_candidates)

    best_detection = None
    for corners in candidates:
        warped = warp_candidate(screen_bgr, corners, template_w, template_h)
        score = calc_same_size_score(warped, template_gray)
        green_ratio = calc_green_ratio(warped)
        detected = score >= match_threshold and green_ratio >= green_threshold

        x, y, width, height = cv2.boundingRect(corners.astype(np.int32))
        absolute_corners = corners.copy()
        absolute_corners[:, 0] += monitor["left"]
        absolute_corners[:, 1] += monitor["top"]

        detection = Detection(
            detected=detected,
            score=score,
            green_ratio=green_ratio,
            scale=1.0,
            mode="perspective",
            left=monitor["left"] + x,
            top=monitor["top"] + y,
            width=width,
            height=height,
            corners=absolute_corners,
        )

        if best_detection is None or detection.score > best_detection.score:
            best_detection = detection

    if best_detection is None:
        return Detection(
            detected=False,
            score=-1.0,
            green_ratio=0.0,
            scale=1.0,
            mode="perspective",
            left=monitor["left"],
            top=monitor["top"],
            width=0,
            height=0,
        )

    return best_detection


def crop_rect_with_margin(
    width: int,
    height: int,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    margin_x = int(round(rect_width * margin_ratio))
    margin_y = int(round(rect_height * margin_ratio))
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(width, x + rect_width + margin_x)
    bottom = min(height, y + rect_height + margin_y)
    return left, top, right, bottom


def detect_in_green_candidates(
    screen_bgr: np.ndarray,
    template_variants: list[TemplateVariant],
    monitor: dict,
    match_threshold: float,
    green_threshold: float,
    min_candidate_area: int,
    max_candidates: int,
    crop_margin: float,
) -> Detection:
    screen_h, screen_w = screen_bgr.shape[:2]
    candidates = find_green_quad_candidates(screen_bgr, min_candidate_area, max_candidates)

    best_detection = None
    for corners in candidates:
        x, y, width, height = cv2.boundingRect(corners.astype(np.int32))
        left, top, right, bottom = crop_rect_with_margin(
            screen_w,
            screen_h,
            x,
            y,
            width,
            height,
            crop_margin,
        )
        crop = screen_bgr[top:bottom, left:right]
        if crop.size == 0:
            continue

        score, (local_x, local_y), variant = calc_best_match(crop, template_variants)
        template_h, template_w = variant.bgr.shape[:2]
        matched = crop[local_y : local_y + template_h, local_x : local_x + template_w]
        green_ratio = calc_green_ratio(matched)
        detected = score >= match_threshold and green_ratio >= green_threshold

        detection = Detection(
            detected=detected,
            score=score,
            green_ratio=green_ratio,
            scale=variant.scale,
            mode="candidate",
            left=monitor["left"] + left + local_x,
            top=monitor["top"] + top + local_y,
            width=template_w,
            height=template_h,
        )

        if best_detection is None or detection.score > best_detection.score:
            best_detection = detection

    if best_detection is None:
        return Detection(
            detected=False,
            score=-1.0,
            green_ratio=0.0,
            scale=1.0,
            mode="candidate",
            left=monitor["left"],
            top=monitor["top"],
            width=0,
            height=0,
        )

    return best_detection


def choose_detection(scale_detection: Detection, perspective_detection: Detection) -> Detection:
    if perspective_detection.detected and not scale_detection.detected:
        return perspective_detection
    if perspective_detection.detected == scale_detection.detected and perspective_detection.score > scale_detection.score:
        return perspective_detection
    return scale_detection


def put_outlined_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_detection(screen_bgr: np.ndarray, detection: Detection, monitor: dict) -> np.ndarray:
    view = screen_bgr.copy()
    x = detection.left - monitor["left"]
    y = detection.top - monitor["top"]
    color = (0, 0, 255) if detection.score > FRAME_RED_SCORE_THRESHOLD else (128, 128, 128)
    label = "DETECTED" if detection.detected else "BEST CANDIDATE"

    if detection.corners is not None:
        corners = detection.corners.copy()
        corners[:, 0] -= monitor["left"]
        corners[:, 1] -= monitor["top"]
        cv2.polylines(view, [corners.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(view, (x, y), (x + detection.width, y + detection.height), color, 2)

    status_lines = [
        f"{label}  score={detection.score:.3f}",
        f"mode={detection.mode}  green={detection.green_ratio:.3f}  scale={detection.scale:.2f}",
        f"time={detection.elapsed_ms:.1f}ms  fps={1000.0 / detection.elapsed_ms if detection.elapsed_ms > 0 else 0.0:.1f}",
        f"frame={'red' if detection.score > FRAME_RED_SCORE_THRESHOLD else 'gray'}  threshold={FRAME_RED_SCORE_THRESHOLD:.2f}  q: quit",
    ]
    for index, line in enumerate(status_lines):
        put_outlined_text(view, line, (8, 24 + index * 24), color)

    put_outlined_text(
        view,
        f"score={detection.score:.3f}",
        (max(0, x), max(24, y - 8)),
        color,
        scale=0.7,
        thickness=2,
    )
    return view


def resize_for_preview(bgr: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    height, width = bgr.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return bgr
    return cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def run_monitor_loop(
    args,
    template: np.ndarray,
    template_variants: list[TemplateVariant],
    capture_frame,
) -> None:
    window_name = "Yokusoku Camera Monitor"
    while True:
        started_at = time.perf_counter()
        frame, monitor = capture_frame()

        scale_detection = None
        if args.detection_mode in ("scale", "both"):
            scale_detection = detect_anywhere(
                frame,
                template_variants,
                monitor,
                match_threshold=args.match_threshold,
                green_threshold=args.green_threshold,
            )

        if args.detection_mode == "scale":
            detection = scale_detection
        else:
            perspective_detection = detect_with_perspective(
                frame,
                template,
                monitor,
                match_threshold=args.perspective_threshold,
                green_threshold=args.green_threshold,
                min_candidate_area=args.min_candidate_area,
                max_candidates=args.max_candidates,
            )
            if args.detection_mode == "candidate":
                candidate_detection = detect_in_green_candidates(
                    frame,
                    template_variants,
                    monitor,
                    match_threshold=args.match_threshold,
                    green_threshold=args.green_threshold,
                    min_candidate_area=args.min_candidate_area,
                    max_candidates=args.max_candidates,
                    crop_margin=args.candidate_crop_margin,
                )
                detection = choose_detection(candidate_detection, perspective_detection)
            elif args.detection_mode == "both" and scale_detection is not None:
                detection = choose_detection(scale_detection, perspective_detection)
            else:
                detection = perspective_detection

        detection.elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if args.no_window:
            status = "検出" if detection.detected else "未検出"
            print(
                f"{status} | mode={detection.mode} | score={detection.score:.3f} | green={detection.green_ratio:.3f} | "
                f"scale={detection.scale:.2f} | time={detection.elapsed_ms:.1f}ms | "
                f"x={detection.left}, y={detection.top}, w={detection.width}, h={detection.height}"
            )

        if not args.no_window:
            view = draw_detection(frame, detection, monitor)
            cv2.imshow(window_name, resize_for_preview(view))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        if args.once:
            break

        time.sleep(INTERVAL_SEC)


def main() -> None:
    parser = argparse.ArgumentParser(description="カメラまたは画面内に抑速アイコンが写っているか判定します。")
    parser.add_argument("--source", choices=["camera", "screen"], default=DEFAULT_SOURCE, help="入力元です。")
    parser.add_argument(
        "--detection-mode",
        choices=["candidate", "perspective", "scale", "both"],
        default=DEFAULT_DETECTION_MODE,
        help="検出方法です。bothは精度優先、candidateは速度と精度のバランス型です。",
    )
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="使用するカメラ番号です。")
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH, help="カメラ取得幅です。0なら指定しません。")
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT, help="カメラ取得高さです。0なら指定しません。")
    parser.add_argument("--mirror-camera", action="store_true", help="カメラ画像を左右反転してから判定します。")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--once", action="store_true", help="1回だけ判定して終了します。")
    parser.add_argument("--no-window", action="store_true", help="判定結果のウィンドウを表示しません。")
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE, help="探索する最小倍率です。")
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE, help="探索する最大倍率です。")
    parser.add_argument("--scale-step", type=float, default=SCALE_STEP, help="倍率探索の刻み幅です。")
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD, help="通常のテンプレート一致度しきい値です。")
    parser.add_argument("--green-threshold", type=float, default=GREEN_RATIO_THRESHOLD, help="緑色の割合しきい値です。")
    parser.add_argument("--no-perspective", action="store_true", help="緑の四角を補正して照合する処理を使いません。")
    parser.add_argument("--perspective-threshold", type=float, default=PERSPECTIVE_MATCH_THRESHOLD, help="歪み補正後の一致度しきい値です。")
    parser.add_argument("--min-candidate-area", type=int, default=MIN_CANDIDATE_AREA, help="歪み補正候補にする緑領域の最小面積です。")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES, help="歪み補正で調べる候補数の上限です。")
    parser.add_argument("--candidate-crop-margin", type=float, default=CANDIDATE_CROP_MARGIN, help="候補周辺を切り出す余白倍率です。")
    args = parser.parse_args()
    if args.no_perspective:
        args.detection_mode = "scale"

    template = load_template(args.template)
    if args.detection_mode in ("candidate", "scale", "both"):
        template_variants = build_template_variants(
            template,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            scale_step=args.scale_step,
        )
    else:
        template_variants = []
    template_h, template_w = template.shape[:2]
    if args.no_window:
        print(f"source = {args.source}")
        print(f"detection_mode = {args.detection_mode}")
        print(f"template = {args.template} ({template_w}x{template_h})")
        if template_variants:
            print(f"scale = {args.min_scale:.2f}..{args.max_scale:.2f} step={args.scale_step:.2f} ({len(template_variants)} variants)")
        print(f"threshold = match={args.match_threshold:.2f} green={args.green_threshold:.2f}")
        if args.detection_mode in ("candidate", "perspective", "both"):
            print(
                f"perspective = on threshold={args.perspective_threshold:.2f} "
                f"min_area={args.min_candidate_area} max_candidates={args.max_candidates}"
            )
        print("監視開始: q で終了")

    try:
        if args.source == "camera":
            cap = open_camera(args.camera_index, args.camera_width, args.camera_height)
            try:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if args.no_window:
                    print(
                        f"camera = index={args.camera_index} "
                        f"size={actual_width}x{actual_height} mirror={args.mirror_camera}"
                    )
                run_monitor_loop(
                    args,
                    template,
                    template_variants,
                    lambda: capture_camera(cap, args.mirror_camera),
                )
            finally:
                cap.release()
        else:
            import mss

            with mss.mss() as sct:
                run_monitor_loop(
                    args,
                    template,
                    template_variants,
                    lambda: capture_screen(sct),
                )
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
