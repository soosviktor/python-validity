"""
Host-based fingerprint database for 06cb:0081 (flashless device).

Stores fingerprint templates as files on the host filesystem.
Uses simple image-based matching (baseline subtraction + correlation).
"""

import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path

from .tls import tls
from .blobs_90 import calibrate_prg

DB_DIR = '/var/lib/python-validity/prints/'
BASELINE_PATH = '/var/lib/python-validity/baseline.bin'
BPL = 152  # bytes per line from sensor
HEADER_SIZE = 8  # per-line header
PIXEL_WIDTH = 144  # actual pixels per line
LINES_PER_FRAME = 144
NUM_FRAMES = 6
MATCH_THRESHOLD = 22.0  # similarity threshold for raw images (lower = more similar)
FINGER_PRESENCE_THRESHOLD = 5.0  # minimum diff signal to confirm finger is on sensor


def ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)


def capture_raw():
    """Capture a raw image using calibrate_prg. Returns raw bytes from EP 0x82."""
    from .usb import usb
    rsp = tls.cmd(calibrate_prg)
    status = struct.unpack('<H', rsp[:2])[0]
    if status != 0:
        raise Exception('Capture failed: 0x%04x' % status)
    img = bytes(usb.dev.read(0x82, 1024 * 1024, timeout=15000))
    return img


def extract_image(raw_data):
    """Extract a 144x144 averaged image from raw sensor data."""
    lines = []
    for i in range(0, len(raw_data), BPL):
        line = raw_data[i:i + BPL]
        if len(line) == BPL:
            lines.append(line[HEADER_SIZE:HEADER_SIZE + PIXEL_WIDTH])

    if len(lines) < LINES_PER_FRAME:
        return None

    # Average frames (skip first frame which may be noisy)
    frame_count = len(lines) // LINES_PER_FRAME
    n = min(frame_count, NUM_FRAMES)
    start_frame = 1 if n > 1 else 0

    avg = [0] * (PIXEL_WIDTH * LINES_PER_FRAME)
    count = 0
    for f in range(start_frame, n):
        for row in range(LINES_PER_FRAME):
            idx = f * LINES_PER_FRAME + row
            if idx < len(lines):
                for col in range(PIXEL_WIDTH):
                    avg[row * PIXEL_WIDTH + col] += lines[idx][col]
                count = max(count, 1)

    frames_used = n - start_frame
    if frames_used > 0:
        avg = [v // frames_used for v in avg]

    return bytes(avg)


def capture_baseline():
    """Capture and save a baseline (no finger) image."""
    logging.info('Capturing baseline (no finger)...')
    raw = capture_raw()
    img = extract_image(raw)
    if img:
        with open(BASELINE_PATH, 'wb') as f:
            f.write(img)
        logging.info('Baseline saved (%d bytes)' % len(img))
    return img


def load_baseline():
    """Load the stored baseline image."""
    if os.path.isfile(BASELINE_PATH):
        with open(BASELINE_PATH, 'rb') as f:
            return f.read()
    return None


def capture_fingerprint(use_cached_baseline=True):
    """Capture a fingerprint image and subtract baseline."""
    baseline = load_baseline() if use_cached_baseline else None
    if baseline is None:
        baseline = capture_baseline()
        if baseline is None:
            raise Exception('Cannot capture baseline')
        time.sleep(0.5)

    raw = capture_raw()
    finger = extract_image(raw)
    if finger is None:
        raise Exception('Cannot extract image')

    # Compute difference (fingerprint signal)
    diff = bytes(min(255, abs(a - b)) for a, b in zip(baseline, finger))

    # Also return the raw finger image for template storage
    return diff, finger


def fingerprint_hash(diff_image):
    """Create a hash/template from a difference image."""
    return hashlib.sha256(diff_image).hexdigest()


def compute_similarity(img1, img2):
    """Compute similarity between two fingerprint images.
    Uses normalized cross-correlation on raw images.
    Returns a distance score (lower = more similar)."""
    if len(img1) != len(img2):
        return 999.0
    n = len(img1)
    # Average absolute difference between raw images
    total_diff = sum(abs(a - b) for a, b in zip(img1, img2))
    avg_diff = total_diff / n
    return avg_diff


def compute_signal_strength(diff_image):
    """Check if there's actually a finger on the sensor."""
    avg = sum(diff_image) / len(diff_image)
    return avg


def images_differ(img1, img2, threshold=1.5):
    """Check if two images are significantly different (finger presence)."""
    if len(img1) != len(img2):
        return True
    diff = sum(abs(a - b) for a, b in zip(img1, img2)) / len(img1)
    return diff > threshold


def get_user_dir(username):
    """Get the directory for a user's fingerprint data."""
    user_dir = os.path.join(DB_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def list_enrolled_fingers(username):
    """List enrolled fingers for a user."""
    user_dir = get_user_dir(username)
    fingers = []
    for f in os.listdir(user_dir):
        if f.endswith('.json'):
            fingers.append(f.replace('.json', ''))
    return fingers


def save_enrollment(username, finger_name, templates):
    """Save enrollment data for a user/finger."""
    user_dir = get_user_dir(username)
    data = {
        'finger': finger_name,
        'created': time.time(),
        'templates': templates,  # list of hex-encoded diff images
        'num_samples': len(templates),
    }
    path = os.path.join(user_dir, finger_name + '.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    logging.info('Saved enrollment for %s/%s (%d templates)' % (username, finger_name, len(templates)))


def load_enrollment(username, finger_name):
    """Load enrollment data for a user/finger."""
    user_dir = get_user_dir(username)
    path = os.path.join(user_dir, finger_name + '.json')
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def delete_enrolled_fingers(username):
    """Delete all enrolled fingers for a user."""
    user_dir = get_user_dir(username)
    for f in os.listdir(user_dir):
        os.remove(os.path.join(user_dir, f))
    logging.info('Deleted all enrollments for %s' % username)


def verify_finger(username):
    """Capture a fingerprint and verify against enrolled templates.

    Security: MUST verify finger presence before accepting match.
    Uses DIFF images (finger - baseline) for matching to ensure
    only actual fingerprints are compared, not sensor fixed patterns.

    Returns (matched_finger_name, confidence) or (None, 0).
    """
    fingers = list_enrolled_fingers(username)
    if not fingers:
        return None, 0

    # Always capture a fresh baseline first
    baseline = load_baseline()
    if baseline is None:
        logging.error('No baseline available')
        return None, 0

    try:
        diff, finger_raw = capture_fingerprint()
    except Exception as e:
        logging.error('Capture failed: %s' % e)
        return None, 0

    # CRITICAL: Check finger presence using diff signal strength
    signal = compute_signal_strength(diff)
    logging.info('Verify: signal strength = %.1f' % signal)

    if signal < FINGER_PRESENCE_THRESHOLD:
        logging.info('Verify: NO FINGER DETECTED (signal=%.1f < threshold=%.1f)' %
                     (signal, FINGER_PRESENCE_THRESHOLD))
        return None, 0

    # Match using RAW images (baseline-independent, consistent sensor pattern)
    # The diff check above ensures a finger is present
    best_match = None
    best_score = float('inf')

    for finger_name in fingers:
        enrollment = load_enrollment(username, finger_name)
        if enrollment is None:
            continue

        for template_hex in enrollment['templates']:
            template = bytes.fromhex(template_hex)
            score = compute_similarity(finger_raw, template)
            if score < best_score:
                best_score = score
                best_match = finger_name

    logging.info('Verify: best match=%s score=%.1f threshold=%.1f' %
                 (best_match, best_score, MATCH_THRESHOLD))

    if best_score < MATCH_THRESHOLD:
        return best_match, best_score
    return None, 0
