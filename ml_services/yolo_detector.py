import os, cv2, time, numpy as np
from collections import deque
from ultralytics import YOLO

from ml_services.telegram_notify import (
    send_admin, send_admin_photo, esc
)
from utils.timeutil import now_ist, utc_now


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = None


# ================================================================
# Stability configuration
# ================================================================
CONF_THRESHOLD       = 0.60   # per-frame confidence floor
STABLE_FRAMES        = 5      # injured must appear in N consecutive frames
STABILITY_WINDOW     = 8      # sliding window size (last N frames analyzed)
ALERT_COOLDOWN_SEC   = 60.0   # min seconds between two admin alerts


# ================================================================
# Runtime state
# ================================================================
_MODEL = None
_recent_frames = deque(maxlen=STABILITY_WINDOW)   # stores bools (injured present?)
_last_alert_ts = 0.0
_alert_latched = False   # prevents repeat alerts while incident continues


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(os.path.join(BASE, "injured.pt"))
    return _MODEL


# ================================================================
# Public API
# ================================================================
def detect_frame(frame, city="", notify_admin=True, save_snapshot=True):
    """
    Run YOLO on a BGR frame.

    Only sends a Telegram alert when the injured class has been
    consistently detected for the last STABLE_FRAMES frames.
    """
    global _last_alert_ts, _alert_latched

    model = _model()
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)

    # ---------- Collect per-frame detections ----------
    detections = []
    for r in results:
        for box in r.boxes:
            cid = int(box.cls[0])
            detections.append({
                "label": model.names[cid],
                "confidence": float(box.conf[0]),
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            })

    annotated = results[0].plot()

    # ---------- Update stability window ----------
    injured_in_frame = [d for d in detections
                        if d["label"].strip().lower() == "injured"]
    frame_has_injured = len(injured_in_frame) > 0
    _recent_frames.append(frame_has_injured)

    # ---------- Evaluate stability ----------
    injured_frames_count = sum(_recent_frames)
    is_stable = injured_frames_count >= STABLE_FRAMES

    # ---------- Decide whether to alert ----------
    if notify_admin and is_stable:
        now = time.time()

        if _alert_latched:
            # Already alerted for this incident — no repeat
            print(f"[YOLO] Injured stable ({injured_frames_count}/{STABILITY_WINDOW}), "
                  f"already alerted for this incident.")
        elif (now - _last_alert_ts) < ALERT_COOLDOWN_SEC:
            print(f"[YOLO] Injured stable but cooldown active "
                  f"({int(now - _last_alert_ts)}s < {int(ALERT_COOLDOWN_SEC)}s).")
        else:
            _last_alert_ts = now
            _alert_latched = True
            _notify_injured(injured_in_frame, annotated, city, save_snapshot)

    elif notify_admin and not frame_has_injured:
        # Nobody injured in current frame → reset the latch so the
        # next genuine incident can trigger a fresh alert.
        if _alert_latched:
            print("[YOLO] Scene clear — re-arming injured alert.")
        _alert_latched = False

    return detections, annotated


def reset_stability_state():
    """Call this when starting a new drone session."""
    global _last_alert_ts, _alert_latched
    _recent_frames.clear()
    _last_alert_ts = 0.0
    _alert_latched = False
    print("[YOLO] Stability state reset.")


# ================================================================
# Internal: build and send the alert
# ================================================================
def _notify_injured(injured_list, annotated_frame, city, save_snapshot):
    try:
        count = len(injured_list)
        avg_conf = sum(d["confidence"] for d in injured_list) / count
        max_conf = max(d["confidence"] for d in injured_list)

        details = "\n".join(
            f"  • Injured — {d['confidence']*100:.1f}%"
            for d in injured_list
        )

        text = (
            f"🚑 <b>INJURED PERSON DETECTED</b>\n"
            f"<b>Count:</b> {count}\n"
            f"<b>Max confidence:</b> {max_conf*100:.1f}%\n"
            f"<b>Avg confidence:</b> {avg_conf*100:.1f}%\n"
            f"<b>Location:</b> {esc(city) or 'Unknown'}\n"
            f"<b>Stability:</b> confirmed over {STABLE_FRAMES} frames\n\n"
            f"<b>Detections:</b>\n{details}\n\n"
            f"— CrisisIntel Drone Surveillance"
        )

        send_admin(text)

        ok, buf = cv2.imencode(".jpg", annotated_frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            send_admin_photo(
                buf.tobytes(),
                caption=f"📸 Confirmed incident — {count} injured"
                        + (f" · {esc(city)}" if city else "")
            )

            if save_snapshot:
                try:
                    snap_dir = os.path.join(
                        BASE, "static", "uploads", "drone_frames"
                    )
                    os.makedirs(snap_dir, exist_ok=True)
                    fname = f"injured_{int(time.time())}.jpg"
                    with open(os.path.join(snap_dir, fname), "wb") as f:
                        f.write(buf.tobytes())
                    print(f"[YOLO] Snapshot saved: {fname}")
                except Exception as e:
                    print(f"[YOLO] Snapshot save failed: {e}")
        else:
            print("[YOLO] Failed to encode annotated frame to JPEG.")

    except Exception as e:
        print(f"[YOLO] Injured alert failed: {e}")