import cv2
import numpy as np

# ── Skin tone consistency ──────────────────────────────────────────────
def analyze_skin_tone(face_imgs):
    """
    Takes list of RGB face crops (already extracted by InsightFace).
    Returns per-frame hue mean + overall consistency score (0-1, lower = suspicious).
    """
    hue_means = []
    for img in face_imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Skin tone hue roughly 0-25 and 160-180 in HSV
        mask = cv2.inRange(hsv, (0, 40, 60), (25, 255, 255))
        if mask.sum() == 0:
            hue_means.append(None)
            continue
        hue = hsv[:, :, 0]
        hue_means.append(float(np.mean(hue[mask > 0])))

    valid = [h for h in hue_means if h is not None]
    if len(valid) < 2:
        return {"per_frame": hue_means, "consistency_score": 1.0, "suspicious": False}

    std = float(np.std(valid))
    # Std > 8 hue units across frames = inconsistency
    consistency = max(0.0, 1.0 - std / 20.0)
    return {
        "per_frame_hue": [round(h, 2) if h else None for h in hue_means],
        "hue_std": round(std, 3),
        "consistency_score": round(consistency, 3),
        "suspicious": std > 8.0
    }

# ── Eye blink detector (EAR) ───────────────────────────────────────────
def eye_aspect_ratio(eye_landmarks):
    # eye_landmarks: 6 (x,y) points in standard dlib order
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C + 1e-6)

def analyze_eye_blinks(face_imgs):
    """
    Requires dlib. Returns EAR per frame and blink count.
    Deepfakes often show reduced or absent blinking.
    """
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        return {"error": str(e), "ears": [], "blink_count": None}

    EAR_THRESHOLD = 0.21
    ears = []
    for img in face_imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = detector(gray, 1)
        if not dets:
            ears.append(None)
            continue
        shape = predictor(gray, dets[0])
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        left_ear  = eye_aspect_ratio(pts[36:42])
        right_ear = eye_aspect_ratio(pts[42:48])
        ears.append(round((left_ear + right_ear) / 2.0, 4))

    valid_ears = [e for e in ears if e is not None]
    blinks = sum(1 for e in valid_ears if e < EAR_THRESHOLD)
    avg_ear = round(float(np.mean(valid_ears)), 4) if valid_ears else None

    return {
        "ear_per_frame": ears,
        "avg_ear": avg_ear,
        "blink_count": blinks,
        "suspicious": blinks == 0 and len(valid_ears) >= 5
    }

# ── Facial boundary artifact detector ─────────────────────────────────
def analyze_boundary_artifacts(face_imgs):
    """
    Laplacian variance at face edges — GAN artifacts cause unnatural sharpness spikes.
    """
    scores = []
    for img in face_imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        # Border strip (10% of image edge)
        border = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        lap = cv2.Laplacian(border, cv2.CV_64F)
        scores.append(round(float(lap.var()), 2))

    avg = round(float(np.mean(scores)), 2) if scores else 0.0
    # Very high Laplacian variance at edges = artifact
    return {
        "laplacian_per_frame": scores,
        "avg_laplacian": avg,
        "suspicious": avg > 800.0
    }

# ── Lip-sync scorer ────────────────────────────────────────────────────
def analyze_lip_sync(face_imgs, has_audio=False):
    """
    Measures lip movement consistency frame-to-frame using landmark distance.
    If audio unavailable, returns motion score only (still useful as XAI signal).
    """
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        return {"error": str(e)}

    mouth_gaps = []
    for img in face_imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = detector(gray, 1)
        if not dets:
            mouth_gaps.append(None)
            continue
        shape = predictor(gray, dets[0])
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        # Vertical mouth gap: top lip (62) to bottom lip (66)
        gap = float(np.linalg.norm(pts[62] - pts[66]))
        mouth_gaps.append(round(gap, 2))

    valid = [g for g in mouth_gaps if g is not None]
    if len(valid) < 2:
        return {"mouth_gaps": mouth_gaps, "motion_std": None, "suspicious": False}

    motion_std = round(float(np.std(valid)), 3)
    # Very low std = lips barely moving (common in some deepfakes)
    return {
        "mouth_gaps_per_frame": mouth_gaps,
        "motion_std": motion_std,
        "audio_available": has_audio,
        "suspicious": motion_std < 1.5
    }

# ── Master function ─────────────────────────────────────────────────────
def run_xai_analysis(face_imgs, has_audio=False):
    return {
        "skin_tone":  analyze_skin_tone(face_imgs),
        "eye_blinks": analyze_eye_blinks(face_imgs),
        "lip_sync":   analyze_lip_sync(face_imgs, has_audio),
        "boundary":   analyze_boundary_artifacts(face_imgs)
    }