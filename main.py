"""Motorcycle-garage suspicious-event demo.

Two modes:
  1) --demo: self-contained synthetic scene; tests the temporal alert rules only.
  2) --source VIDEO: YOLO person/motorcycle detection + ByteTrack + temporal rules.

An alert means "review this event", not a conclusion that a crime occurred.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    track_id: int
    box: Tuple[float, float, float, float]
    confidence: float = 1.0


def box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Minimum Euclidean distance between two axis-aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return math.hypot(dx, dy)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class AlertEngine:
    """Rule-based event layer on top of tracked person/motorcycle detections."""

    def __init__(
        self,
        near_px_at_1280: float = 80.0,
        alert_seconds: float = 8.0,
        motion_window_seconds: float = 1.5,
        moving_ratio: float = 0.04,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.near_px_at_1280 = near_px_at_1280
        self.alert_seconds = alert_seconds
        self.motion_window_seconds = motion_window_seconds
        self.moving_ratio = moving_ratio
        self.cooldown_seconds = cooldown_seconds
        self.pair_started: Dict[Tuple[int, int], float] = {}
        self.pair_last_seen: Dict[Tuple[int, int], float] = {}
        self.last_alert: Dict[Tuple[int, int], float] = {}
        self.bike_history: Dict[int, Deque[Tuple[float, float, float]]] = defaultdict(deque)

    def update(
        self,
        detections: Iterable[Detection],
        timestamp_sec: float,
        frame_width: int,
    ) -> Tuple[List[dict], set]:
        dets = list(detections)
        persons = [d for d in dets if d.label == "person" and d.track_id >= 0]
        bikes = [d for d in dets if d.label == "motorcycle" and d.track_id >= 0]
        near_px = self.near_px_at_1280 * max(frame_width, 1) / 1280.0
        move_px = self.moving_ratio * max(frame_width, 1)

        for bike in bikes:
            cx, cy = box_center(bike.box)
            hist = self.bike_history[bike.track_id]
            hist.append((timestamp_sec, cx, cy))
            while hist and timestamp_sec - hist[0][0] > self.motion_window_seconds:
                hist.popleft()

        seen_pairs = set()
        warnings = set()
        new_events: List[dict] = []

        for person in persons:
            for bike in bikes:
                key = (person.track_id, bike.track_id)
                if box_gap(person.box, bike.box) > near_px:
                    continue
                seen_pairs.add(key)
                self.pair_started.setdefault(key, timestamp_sec)
                self.pair_last_seen[key] = timestamp_sec
                duration = max(0.0, timestamp_sec - self.pair_started[key])

                hist = list(self.bike_history.get(bike.track_id, ()))
                displacement = 0.0
                if len(hist) >= 2 and hist[-1][0] - hist[0][0] >= 0.6:
                    displacement = math.hypot(hist[-1][1] - hist[0][1], hist[-1][2] - hist[0][2])
                bike_moving = displacement >= move_px and duration >= 1.0
                prolonged_contact = duration >= self.alert_seconds

                reason = None
                if bike_moving:
                    reason = "Motorcycle moves while a person remains nearby"
                elif prolonged_contact:
                    reason = "Person remains near a motorcycle for an extended period"

                if reason:
                    warnings.add(key)
                    last = self.last_alert.get(key, -1e9)
                    if timestamp_sec - last >= self.cooldown_seconds:
                        new_events.append({
                            "timestamp_sec": round(timestamp_sec, 2),
                            "person_id": person.track_id,
                            "motorcycle_id": bike.track_id,
                            "duration_sec": round(duration, 2),
                            "reason": reason,
                        })
                        self.last_alert[key] = timestamp_sec

        # Keep state briefly through occasional missed detections, then expire it.
        for key, last_seen in list(self.pair_last_seen.items()):
            if timestamp_sec - last_seen > 1.0:
                self.pair_last_seen.pop(key, None)
                self.pair_started.pop(key, None)
                if key not in self.last_alert or timestamp_sec - self.last_alert[key] > self.cooldown_seconds * 2:
                    self.last_alert.pop(key, None)
        for track_id, hist in list(self.bike_history.items()):
            while hist and timestamp_sec - hist[0][0] > self.motion_window_seconds:
                hist.popleft()
            if not hist:
                self.bike_history.pop(track_id, None)
        return new_events, warnings


def preprocess_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    use_clahe: bool = False,
    use_denoise: bool = False,
) -> np.ndarray:
    """Conservative optional preprocessing; avoid blindly enhancing every frame."""
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame received from video decoder.")
    h, w = frame.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if use_denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
    if use_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return frame


def draw_detection(frame: np.ndarray, det: Detection, warning: bool = False) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det.box]
    color = (0, 0, 255) if warning else ((0, 200, 0) if det.label == "person" else (255, 170, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{det.label} ID:{det.track_id} {det.confidence:.2f}"
    cv2.putText(frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def draw_warning_banner(frame: np.ndarray, text: str = "SUSPICIOUS EVENT - SECURITY REVIEW REQUIRED") -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 180), -1)
    cv2.putText(frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def format_timestamp(seconds: float) -> str:
    total_ms = int(max(0, seconds) * 1000)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_events_csv(events: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "timestamp_sec", "person_id", "motorcycle_id", "duration_sec", "reason"]
    # Write an empty but valid CSV with headers when no event has been flagged.
    existing = []
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                existing = list(csv.DictReader(f))
        except (OSError, csv.Error):
            existing = []
    combined = existing + [dict(e, timestamp=format_timestamp(e["timestamp_sec"])) for e in events]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)


def draw_synthetic_garage(frame: np.ndarray, bike_box: Tuple[int, int, int, int], person_box: Tuple[int, int, int, int], elapsed: float) -> None:
    """Illustration only: draw shapes that stand in for detector boxes."""
    h, w = frame.shape[:2]
    frame[:] = (53, 58, 64)
    # ceiling lights and concrete floor
    cv2.rectangle(frame, (0, 0), (w, 155), (43, 47, 52), -1)
    for x in range(45, w, 180):
        cv2.rectangle(frame, (x, 28), (x + 95, 36), (205, 205, 185), -1)
    cv2.line(frame, (0, 175), (w, 175), (95, 98, 101), 3)
    for x in range(30, w, 200):
        cv2.line(frame, (x, 480), (x + 105, 480), (225, 190, 65), 3)
    cv2.putText(frame, "SIMULATED PARKING GARAGE - NOT REAL CCTV", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    bx1, by1, bx2, by2 = bike_box
    # Simple motorcycle illustration inside its tracked box
    wheel_y = by2 - 17
    wheel_r = max(15, int((bx2 - bx1) * 0.11))
    wx1, wx2 = bx1 + 27, bx2 - 30
    cv2.circle(frame, (wx1, wheel_y), wheel_r, (22, 22, 22), 5)
    cv2.circle(frame, (wx2, wheel_y), wheel_r, (22, 22, 22), 5)
    cv2.line(frame, (wx1, wheel_y), (bx1 + 76, by1 + 50), (45, 170, 230), 5)
    cv2.line(frame, (bx1 + 76, by1 + 50), (wx2, wheel_y), (45, 170, 230), 5)
    cv2.line(frame, (bx1 + 73, by1 + 48), (bx1 + 105, by1 + 36), (220, 220, 220), 5)
    cv2.ellipse(frame, (bx1 + 90, by1 + 64), (32, 18), 0, 0, 360, (30, 30, 200), -1)
    cv2.line(frame, (bx1 + 63, by1 + 38), (bx1 + 87, by1 + 38), (10, 10, 10), 7)

    px1, py1, px2, py2 = person_box
    cx = (px1 + px2) // 2
    cv2.circle(frame, (cx, py1 + 22), 17, (205, 180, 140), -1)
    cv2.line(frame, (cx, py1 + 40), (cx, py1 + 105), (30, 160, 80), 17)
    cv2.line(frame, (cx, py1 + 53), (cx + 35, py1 + 83), (205, 180, 140), 8)
    cv2.line(frame, (cx, py1 + 105), (cx - 18, py2 - 8), (30, 30, 160), 10)
    cv2.line(frame, (cx, py1 + 105), (cx + 20, py2 - 8), (30, 30, 160), 10)
    cv2.putText(frame, f"time={elapsed:.1f}s", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_demo(output_dir: Path, fps: int = 10, duration: float = 12.0) -> int:
    """Run a synthetic end-to-end test without installing/downloading YOLO weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    video_path = output_dir / "demo_annotated.mp4"
    frame_path = output_dir / "demo_frame.png"
    csv_path = output_dir / "alerts.csv"
    if csv_path.exists():
        csv_path.unlink()
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create demo MP4. Check that OpenCV has an MP4V video writer.")

    engine = AlertEngine(alert_seconds=8.0)
    all_events: List[dict] = []
    alert_frame_saved = False
    frame_count = int(fps * duration)
    for idx in range(frame_count):
        t = idx / float(fps)
        shift = max(0.0, t - 4.0) * 42.0
        bike_x = int(345 + shift)
        # Keep the mock motorcycle inside the frame for the demo duration.
        bike_x = min(bike_x, width - 175)
        bike_box = (bike_x, 305, bike_x + 160, 435)
        person_box = (bike_x - 102, 220, bike_x - 28, 425)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_synthetic_garage(frame, bike_box, person_box, t)
        detections = [
            Detection("person", 1, tuple(map(float, person_box)), 0.99),
            Detection("motorcycle", 2, tuple(map(float, bike_box)), 0.98),
        ]
        events, warnings = engine.update(detections, t, width)
        for det in detections:
            draw_detection(frame, det, warning=bool(warnings))
        if warnings:
            draw_warning_banner(frame)
        for event in events:
            all_events.append(event)
            print(f"[ALERT] {format_timestamp(event['timestamp_sec'])} | {event['reason']}")
            if not alert_frame_saved:
                cv2.imwrite(str(frame_path), frame)
                alert_frame_saved = True
        writer.write(frame)
    writer.release()
    write_events_csv(all_events, csv_path)
    if not frame_path.exists():
        # Save a useful final frame even when configured thresholds were not met.
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, fps * 9))
        ok, last_frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(frame_path), last_frame)
    print(f"Demo video: {video_path}")
    print(f"Alert log:  {csv_path}")
    print(f"Frame:      {frame_path}")
    print("NOTE: demo detections are simulated; this validates only the alert logic, not theft recognition or YOLO accuracy.")
    return len(all_events)


def parse_tracked_detections(result) -> List[Detection]:
    detections: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return detections
    xyxys = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = result.names
    for box, class_id, track_id, conf in zip(xyxys, classes, ids, confs):
        label = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
        if label not in ("person", "motorcycle"):
            continue
        detections.append(Detection(label, int(track_id), tuple(map(float, box)), float(conf)))
    return detections


def run_video(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'ultralytics'. Run: python -m pip install -r requirements.txt") from exc

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video/camera source: {args.source}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not math.isfinite(src_fps) or src_fps < 1.0:
        src_fps = 25.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "annotated_video.mp4"
    output_csv = output_dir / "alerts.csv"
    output_csv.write_text("timestamp,timestamp_sec,person_id,motorcycle_id,duration_sec,reason\n", encoding="utf-8-sig")

    model = YOLO(args.model)
    engine = AlertEngine(near_px_at_1280=args.near_px, alert_seconds=args.alert_seconds)
    writer: Optional[cv2.VideoWriter] = None
    all_events: List[dict] = []
    frame_idx = 0
    event_image_idx = 1
    print("Processing. A warning marks an event for human review; it is not a verdict of theft.")
    try:
        while True:
            ok, original = cap.read()
            if not ok:
                break
            timestamp_sec = frame_idx / src_fps
            frame = preprocess_frame(original, args.max_width, args.clahe, args.denoise)
            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0, 3],  # COCO: person=0, motorcycle=3
                conf=args.conf,
                verbose=False,
            )[0]
            detections = parse_tracked_detections(result)
            events, warnings = engine.update(detections, timestamp_sec, frame.shape[1])
            warn_ids = set()
            for pid, bid in warnings:
                warn_ids.add(("person", pid))
                warn_ids.add(("motorcycle", bid))
            for det in detections:
                draw_detection(frame, det, (det.label, det.track_id) in warn_ids)
            if warnings:
                draw_warning_banner(frame)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Could not create output MP4 with this OpenCV installation.")
            writer.write(frame)

            for event in events:
                all_events.append(event)
                append_event_csv(event, output_csv)
                image_name = f"alert_{event_image_idx:04d}.jpg"
                cv2.imwrite(str(output_dir / image_name), frame)
                event_image_idx += 1
                print(f"[ALERT] {format_timestamp(event['timestamp_sec'])} | person={event['person_id']} motorcycle={event['motorcycle_id']} | {event['reason']}")

            if args.show:
                cv2.imshow("Motorcycle Garage - Review Alerts", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if writer is None:
        raise RuntimeError("The source opened but returned no readable frames.")
    print(f"Processed {frame_idx} frames ({frame_idx / src_fps:.1f}s at {src_fps:.2f} FPS).")
    print(f"Annotated video: {output_video}")
    print(f"Alert log:       {output_csv}")
    print(f"Alert snapshots: {output_dir}")
    print("No model training is performed by this script. Evaluate and tune on labeled local CCTV before operational use.")
    return len(all_events)


def append_event_csv(event: dict, csv_path: Path) -> None:
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "timestamp_sec", "person_id", "motorcycle_id", "duration_sec", "reason"])
        writer.writerow(dict(event, timestamp=format_timestamp(event["timestamp_sec"])))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motorcycle garage suspicious-event alert prototype")
    parser.add_argument("--source", default="", help="Input CCTV video path or camera index (e.g. 0). Required unless --demo is used.")
    parser.add_argument("--demo", action="store_true", help="Run a self-contained synthetic demonstration without YOLO.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights; default yolo11n.pt (downloaded on first use).")
    parser.add_argument("--output-dir", default="outputs", help="Directory for annotated video, CSV, and snapshots.")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize only when wider than this; aspect ratio is preserved. Use 0 to disable.")
    parser.add_argument("--conf", type=float, default=0.25, help="Object confidence threshold.")
    parser.add_argument("--near-px", type=float, default=80.0, help="Person-to-motorcycle proximity threshold at a 1280px-wide frame.")
    parser.add_argument("--alert-seconds", type=float, default=8.0, help="Continuous proximity duration before a dwell-only alert.")
    parser.add_argument("--clahe", action="store_true", help="Apply conservative CLAHE luminance correction (may amplify noise).")
    parser.add_argument("--denoise", action="store_true", help="Apply slow color denoising before inference.")
    parser.add_argument("--show", action="store_true", help="Show a live preview; press q to stop.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.demo:
            run_demo(Path(args.output_dir))
        elif args.source:
            run_video(args)
        else:
            parser.error("Provide --demo or --source PATH_TO_VIDEO (or --source 0 for a camera).")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
