from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import json
import cv2
import numpy as np
import torch
from werkzeug.utils import secure_filename
from torchvision import transforms
from queue import Queue

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

from model import load_model as load_video_model
from model_image import DeepfakeImgNet

MODEL_PATH_VIDEO = os.path.abspath(os.path.join(BASE_DIR, '..', 'checkpoints', 'video', 'best_model_video.pt'))
MODEL_PATH_IMAGE = os.path.abspath(os.path.join(BASE_DIR, '..', 'checkpoints', 'image', 'best_model_image.pt'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_model = load_video_model(MODEL_PATH_VIDEO, device)
image_model = DeepfakeImgNet().to(device)
image_model.load_state_dict(torch.load(MODEL_PATH_IMAGE, map_location=device))
image_model.eval()

IMG_SIZE = 160
MAX_SEQ  = 10

video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def allowed_file(filename, typ):
    if typ == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# =============================================================================
# GRAD-CAM  (fixed: numpy/torch conflict + size mismatch)
# =============================================================================
def generate_gradcam(model, input_tensor, target_layer_name):
    activations, gradients = {}, {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break

    model.zero_grad()
    output     = model(input_tensor)
    pred_class = output.argmax(dim=1)
    score      = output[:, pred_class]
    score.backward()

    if 'value' not in activations or 'value' not in gradients:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    activation = activations['value'].squeeze(0)
    heatmap    = torch.mean(activation, dim=0).cpu().detach().numpy().astype(np.float32)
    heatmap    = np.maximum(heatmap, 0)

    hmap_max = float(np.max(heatmap))
    if hmap_max == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    heatmap /= hmap_max
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8   = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)

# =============================================================================
# XAI ANALYSIS FUNCTIONS
# =============================================================================

def xai_skin_tone(face_imgs):
    hue_means = []
    for img in face_imgs:
        hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 40, 60), (25, 255, 255))
        if mask.sum() == 0:
            hue_means.append(None)
            continue
        hue = hsv[:, :, 0]
        hue_means.append(float(np.mean(hue[mask > 0])))
    valid = [h for h in hue_means if h is not None]
    if len(valid) < 2:
        return {"per_frame_hue": hue_means, "hue_std": None,
                "consistency_score": 1.0, "suspicious": False}
    std         = float(np.std(valid))
    consistency = max(0.0, 1.0 - std / 20.0)
    return {
        "per_frame_hue":     [round(h, 2) if h is not None else None for h in hue_means],
        "hue_std":           round(std, 3),
        "consistency_score": round(consistency, 3),
        "suspicious":        std > 8.0
    }


def xai_eye_blinks(face_imgs):
    try:
        import dlib
        detector  = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat"))
    except Exception as e:
        return {"error": str(e), "ears": [], "blink_count": None, "suspicious": None}

    def ear(pts):
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    EAR_THRESH = 0.21
    ears = []
    for img in face_imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = detector(gray, 1)
        if not dets:
            ears.append(None)
            continue
        shape = predictor(gray, dets[0])
        pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        val   = round((ear(pts[36:42]) + ear(pts[42:48])) / 2.0, 4)
        ears.append(val)

    valid  = [e for e in ears if e is not None]
    blinks = sum(1 for e in valid if e < EAR_THRESH)
    avg    = round(float(np.mean(valid)), 4) if valid else None
    return {
        "ear_per_frame": ears,
        "avg_ear":       avg,
        "blink_count":   blinks,
        "suspicious":    (blinks == 0 and len(valid) >= 5)
    }


def xai_boundary_artifacts(face_imgs):
    scores = []
    for img in face_imgs:
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w  = gray.shape
        inner = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        scores.append(round(float(cv2.Laplacian(inner, cv2.CV_64F).var()), 2))
    avg = round(float(np.mean(scores)), 2) if scores else 0.0
    return {
        "laplacian_per_frame": scores,
        "avg_laplacian":       avg,
        "suspicious":          avg > 800.0
    }


def xai_lip_sync(face_imgs, has_audio=False):
    try:
        import dlib
        detector  = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat"))
    except Exception as e:
        return {"error": str(e), "suspicious": None}

    gaps = []
    for img in face_imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = detector(gray, 1)
        if not dets:
            gaps.append(None)
            continue
        shape = predictor(gray, dets[0])
        pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        gaps.append(round(float(np.linalg.norm(pts[62] - pts[66])), 2))

    valid      = [g for g in gaps if g is not None]
    motion_std = round(float(np.std(valid)), 3) if len(valid) >= 2 else None
    return {
        "mouth_gaps_per_frame": gaps,
        "motion_std":           motion_std,
        "audio_available":      has_audio,
        "suspicious":           (motion_std is not None and motion_std < 1.5)
    }


def run_xai_analysis(face_imgs, has_audio=False):
    return {
        "skin_tone":  xai_skin_tone(face_imgs),
        "eye_blinks": xai_eye_blinks(face_imgs),
        "lip_sync":   xai_lip_sync(face_imgs, has_audio),
        "boundary":   xai_boundary_artifacts(face_imgs)
    }

# =============================================================================
# END XAI FUNCTIONS
# =============================================================================


def process_video(video_path, save_id, result_queue):
    from insightface import app as insight_app

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        result_queue.put({'prediction': 'No Frame Extracted', 'confidence': 0.0,
                          'gradcam_1': '', 'gradcam_2': ''})
        return

    frame_indices = np.linspace(0, total_frames - 1, MAX_SEQ).astype(int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        result_queue.put({'prediction': 'No Frame Extracted', 'confidence': 0.0,
                          'gradcam_1': '', 'gradcam_2': ''})
        return

    face_app = insight_app.FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    face_imgs = []
    for frame in frames:
        faces = face_app.get(frame)
        if faces:
            largest      = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            x1,y1,x2,y2 = map(int, largest.bbox)
            face_img     = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_imgs.append(cv2.resize(face_img, (IMG_SIZE, IMG_SIZE)))
        if len(face_imgs) >= MAX_SEQ:
            break

    if not face_imgs:
        result_queue.put({'prediction': 'No Face Detected', 'confidence': 0.0,
                          'gradcam_1': '', 'gradcam_2': ''})
        return

    inputs = torch.stack([video_transform(img) for img in face_imgs[:MAX_SEQ]])
    inputs = inputs.unsqueeze(0).to(device)

    video_model.eval()
    with torch.no_grad():
        outputs              = video_model(inputs)
        prob                 = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(prob, 1)
        confidence           = confidence.item()
        pred_idx             = pred_idx.item()
        prediction           = {0: 'Real', 1: 'Fake'}.get(pred_idx, 'Unknown')

    heatmaps     = []
    target_layer = 'xception.features.11'
    result_dir   = os.path.join(PROCESSED_FOLDER, save_id)
    os.makedirs(result_dir, exist_ok=True)

    for i in range(min(2, len(face_imgs))):
        filename = f'GradCAM{i+1}.png'
        out_path = os.path.join(result_dir, filename)
        try:
            heatmap  = generate_gradcam(video_model.xception, inputs[0, i].unsqueeze(0), target_layer)
            face_rgb = face_imgs[i]
            heatmap  = cv2.resize(heatmap, (face_rgb.shape[1], face_rgb.shape[0]))
            overlaid = cv2.addWeighted(face_rgb, 0.6, heatmap, 0.4, 0)
            cv2.imwrite(out_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        except Exception:
            cv2.imwrite(out_path, np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        heatmaps.append(filename)

    try:
        xai_results = run_xai_analysis(face_imgs, has_audio=False)
        atomic_write_json(os.path.join(result_dir, 'xai_scores.json'), xai_results)
    except Exception as xai_err:
        atomic_write_json(os.path.join(result_dir, 'xai_scores.json'), {"error": str(xai_err)})

    result_queue.put({
        'prediction': prediction,
        'confidence': confidence,
        'gradcam_1':  heatmaps[0] if len(heatmaps) > 0 else '',
        'gradcam_2':  heatmaps[1] if len(heatmaps) > 1 else '',
    })


def process_image(img_path, save_id, result_queue):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not load image file.")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        proc_img = img_transform(img_rgb).unsqueeze(0).to(device)

        image_model.eval()
        with torch.no_grad():
            outputs              = image_model(proc_img)
            prob                 = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(prob, 1)
            confidence           = confidence.item()
            pred_idx             = pred_idx.item()
            prediction           = {0: 'Real', 1: 'Fake'}.get(pred_idx, 'Unknown')

        result_dir  = os.path.join(PROCESSED_FOLDER, save_id)
        os.makedirs(result_dir, exist_ok=True)

        gradcam_img  = generate_gradcam(image_model.xception, proc_img, 'conv4')
        img_for_over = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        gradcam_img  = cv2.resize(gradcam_img, (img_for_over.shape[1], img_for_over.shape[0]))
        overlaid     = cv2.addWeighted(img_for_over, 0.6, gradcam_img, 0.4, 0)
        filename     = 'GradCAM1.png'
        cv2.imwrite(os.path.join(result_dir, filename), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

        try:
            xai_results = run_xai_analysis([cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))], has_audio=False)
            atomic_write_json(os.path.join(result_dir, 'xai_scores.json'), xai_results)
        except Exception as xai_err:
            atomic_write_json(os.path.join(result_dir, 'xai_scores.json'), {"error": str(xai_err)})

        result_queue.put({'prediction': prediction, 'confidence': confidence,
                          'gradcam_1': filename, 'gradcam_2': ''})

    except Exception as e:
        print("[IMG ERROR]", e)
        result_queue.put({'prediction': f'Processing Error: {e}', 'confidence': 0.0,
                          'gradcam_1': '', 'gradcam_2': ''})


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, 'video'):
        filename  = secure_filename(file.filename)
        save_id   = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_FOLDER, f"{save_id}_{filename}")
        file.save(save_path)
        result_dir = os.path.join(PROCESSED_FOLDER, save_id)
        os.makedirs(result_dir, exist_ok=True)
        json_path  = os.path.join(result_dir, 'result.json')
        q = Queue()
        try:
            process_video(save_path, save_id, q)
            atomic_write_json(json_path, q.get() if not q.empty() else
                              {'prediction': 'Error', 'confidence': 0.0, 'gradcam_1': '', 'gradcam_2': ''})
        except Exception as e:
            atomic_write_json(json_path, {'prediction': f'Error: {e}', 'confidence': 0.0,
                                          'gradcam_1': '', 'gradcam_2': ''})
        return jsonify({'job_id': save_id}), 202
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, 'image'):
        filename  = secure_filename(file.filename)
        save_id   = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_FOLDER, f"{save_id}_{filename}")
        file.save(save_path)
        result_dir = os.path.join(PROCESSED_FOLDER, save_id)
        os.makedirs(result_dir, exist_ok=True)
        json_path  = os.path.join(result_dir, 'result.json')
        q = Queue()
        try:
            process_image(save_path, save_id, q)
            atomic_write_json(json_path, q.get() if not q.empty() else
                              {'prediction': 'Error', 'confidence': 0.0, 'gradcam_1': '', 'gradcam_2': ''})
        except Exception as e:
            atomic_write_json(json_path, {'prediction': f'Error: {e}', 'confidence': 0.0,
                                          'gradcam_1': '', 'gradcam_2': ''})
        return jsonify({'job_id': save_id}), 202
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    result_path = os.path.join(PROCESSED_FOLDER, job_id, 'result.json')
    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        with open(result_path) as f:
            return jsonify({'status': 'completed', 'result': json.load(f)})
    return jsonify({'status': 'processing'})


@app.route('/download/<job_id>/<filename>', methods=['GET'])
def download_file(job_id, filename):
    return send_from_directory(os.path.join(PROCESSED_FOLDER, job_id), filename)


# =============================================================================
# LAYMAN LANGUAGE HELPERS  — every score gets a plain-English explanation
# =============================================================================

def explain_skin_tone(skin):
    """Translate skin tone HSV scores into plain English."""
    std   = skin.get('hue_std')
    score = skin.get('consistency_score')
    susp  = skin.get('suspicious')

    if std is None:
        return {
            "title":       "Skin Tone Consistency",
            "icon":        "🎨",
            "what":        "We checked whether the person's skin colour looked the same throughout the video.",
            "finding":     "Not enough skin detected to analyse.",
            "implication": "This check could not be completed.",
            "verdict":     "unknown"
        }

    pct = round((score or 0) * 100, 1)
    if not susp:
        finding     = f"The skin tone stayed very consistent across frames — {pct}% consistency."
        implication = "Real faces keep a steady skin tone under the same lighting. This looks natural."
    else:
        finding     = f"The skin tone shifted noticeably across frames — only {pct}% consistent."
        implication = ("Deepfake face-swaps often struggle to keep the skin tone stable frame-to-frame. "
                       "This flickering is a common sign of a manipulated video.")

    return {
        "title":       "Skin Tone Consistency",
        "icon":        "🎨",
        "what":        "We checked whether the person's skin colour looked the same throughout the video.",
        "finding":     finding,
        "implication": implication,
        "verdict":     "suspicious" if susp else "normal"
    }


def explain_eye_blinks(blinks):
    """Translate EAR / blink count into plain English."""
    count = blinks.get('blink_count')
    avg   = blinks.get('avg_ear')
    susp  = blinks.get('suspicious')
    err   = blinks.get('error')

    if err or count is None:
        return {
            "title":       "Eye Blinking",
            "icon":        "👁️",
            "what":        "We counted how many times the person blinked.",
            "finding":     "Could not detect eye landmarks reliably.",
            "implication": "This check could not be completed.",
            "verdict":     "unknown"
        }

    if not susp:
        if count == 0:
            finding     = "No blinks detected, but not enough frames to be conclusive."
            implication = "More frames would be needed for a reliable blink check."
        else:
            finding     = f"The person blinked {count} time(s) — a natural rate."
            implication = "Real people blink naturally. Normal blinking is a good sign of authenticity."
    else:
        finding     = f"The person did not blink at all across {len(blinks.get('ear_per_frame', []))} frames."
        implication = ("Early deepfake models were trained on images where people rarely blink. "
                       "Zero blinking in a video is a strong indicator of a manipulated face.")

    return {
        "title":       "Eye Blinking",
        "icon":        "👁️",
        "what":        "We counted how many times the person blinked. Real people blink 15–20 times per minute.",
        "finding":     finding,
        "implication": implication,
        "verdict":     "suspicious" if susp else "normal"
    }


def explain_lip_sync(lip):
    """Translate mouth gap / motion std into plain English."""
    std  = lip.get('motion_std')
    susp = lip.get('suspicious')
    err  = lip.get('error')

    if err or std is None:
        return {
            "title":       "Lip Movement",
            "icon":        "👄",
            "what":        "We measured how much the lips moved across the video.",
            "finding":     "Could not detect lip landmarks reliably.",
            "implication": "This check could not be completed.",
            "verdict":     "unknown"
        }

    if not susp:
        finding     = f"The lips moved naturally with a variation of {std}px between frames."
        implication = "Natural speech produces varied lip movement. This looks normal."
    else:
        finding     = f"The lips barely moved — variation was only {std}px, which is unnaturally low."
        implication = ("Deepfakes sometimes freeze or barely animate the mouth region. "
                       "Stiff, barely-moving lips are a common sign of a face-swap.")

    return {
        "title":       "Lip Movement",
        "icon":        "👄",
        "what":        "We measured how much the lips moved across frames. Stiff lips can indicate a deepfake.",
        "finding":     finding,
        "implication": implication,
        "verdict":     "suspicious" if susp else "normal"
    }


def explain_boundary(boundary):
    """Translate Laplacian variance into plain English."""
    avg  = boundary.get('avg_laplacian')
    susp = boundary.get('suspicious')

    if avg is None:
        return {
            "title":       "Face Edge Quality",
            "icon":        "🔍",
            "what":        "We examined the edges around the face for signs of digital blending.",
            "finding":     "Could not analyse edge quality.",
            "implication": "This check could not be completed.",
            "verdict":     "unknown"
        }

    if not susp:
        finding     = f"The face edges look natural with a sharpness score of {avg}."
        implication = ("Real faces blend naturally with their background. "
                       "No unusual sharp or blurry patches were found around the face.")
    else:
        finding     = f"Unusually sharp or patchy edges were detected around the face (score: {avg})."
        implication = ("AI-generated faces are often pasted onto a background, "
                       "leaving unnatural sharp or blurry boundaries around the hairline, "
                       "ears, or jawline — a classic deepfake artefact.")

    return {
        "title":       "Face Edge Quality",
        "icon":        "🔍",
        "what":        "We looked at the edges around the face for signs of digital cut-and-paste.",
        "finding":     finding,
        "implication": implication,
        "verdict":     "suspicious" if susp else "normal"
    }


def build_summary_sentence(prediction, skin_ex, blink_ex, lip_ex, boundary_ex):
    """One plain-English paragraph summarising all findings."""
    flags  = [e for e in [skin_ex, blink_ex, lip_ex, boundary_ex] if e['verdict'] == 'suspicious']
    clears = [e for e in [skin_ex, blink_ex, lip_ex, boundary_ex] if e['verdict'] == 'normal']

    if prediction == 'Fake':
        if not flags:
            return ("The AI model classified this as a deepfake based on deep visual patterns it learned "
                    "during training, even though the individual checks above appear normal. "
                    "Some sophisticated deepfakes can pass simple checks but still contain subtle "
                    "pixel-level inconsistencies the model detects.")
        flag_names = " and ".join(f['title'].lower() for f in flags)
        return (f"The model flagged this as a deepfake. The clearest signs were problems with "
                f"{flag_names}. " +
                (f"However, {clears[0]['title'].lower()} appeared normal, which shows this is a "
                 f"relatively well-made fake." if clears else
                 "Multiple checks raised concerns, making this a strong deepfake signal."))
    else:
        if not flags:
            return ("All checks passed. The skin tone was consistent, blinking was natural, "
                    "lip movement looked genuine, and no edge artefacts were found. "
                    "The AI model is confident this is an authentic, unmanipulated face.")
        flag_names = " and ".join(f['title'].lower() for f in flags)
        return (f"The model believes this is real, though {flag_names} showed minor anomalies. "
                f"These could be due to video compression, lighting changes, or camera motion "
                f"rather than manipulation.")


# =============================================================================
# XAI DASHBOARD  —  http://127.0.0.1:5000/report/<job_id>
# =============================================================================

@app.route('/report/<job_id>')
def xai_report(job_id):
    result_dir  = os.path.join(PROCESSED_FOLDER, job_id)
    result_path = os.path.join(result_dir, 'result.json')
    xai_path    = os.path.join(result_dir, 'xai_scores.json')

    if not os.path.exists(result_path):
        return "<h2 style='font-family:sans-serif;padding:40px'>Report not ready yet.</h2>", 404

    with open(result_path) as f:
        result = json.load(f)
    xai = json.load(open(xai_path)) if os.path.exists(xai_path) else {}

    skin     = xai.get('skin_tone',  {})
    blinks   = xai.get('eye_blinks', {})
    lip      = xai.get('lip_sync',   {})
    boundary = xai.get('boundary',   {})

    # Build plain-English explanations
    skin_ex     = explain_skin_tone(skin)
    blink_ex    = explain_eye_blinks(blinks)
    lip_ex      = explain_lip_sync(lip)
    boundary_ex = explain_boundary(boundary)

    prediction  = result.get('prediction', 'Unknown')
    confidence  = round(result.get('confidence', 0) * 100, 1)
    summary     = build_summary_sentence(prediction, skin_ex, blink_ex, lip_ex, boundary_ex)

    verdict_color = '#c0392b' if prediction == 'Fake' else '#1e8449'
    verdict_bg    = '#fdecea' if prediction == 'Fake' else '#eafaf1'
    verdict_emoji = '⚠️' if prediction == 'Fake' else '✅'

    gc1     = result.get('gradcam_1', '')
    gc2     = result.get('gradcam_2', '')
    gc1_url = f'/download/{job_id}/{gc1}' if gc1 else ''
    gc2_url = f'/download/{job_id}/{gc2}' if gc2 else ''

    # Sparkline bars
    def sparkline(values, color, threshold=None, invert=False):
        clean = [v for v in values if v is not None]
        if not clean:
            return '<span style="font-size:12px;color:#aaa">not enough data</span>'
        max_v = max(clean) or 1
        bars  = []
        for v in values:
            if v is None:
                bars.append('<div style="width:12px;height:36px;background:#eee;display:inline-block;'
                            'margin:1px;vertical-align:bottom;border-radius:2px"></div>')
                continue
            pct    = int((v / max_v) * 36)
            is_bad = threshold is not None and (v < threshold if invert else v > threshold)
            c      = '#e74c3c' if is_bad else color
            bars.append(f'<div title="{round(v,2)}" style="width:12px;height:{max(2,pct)}px;'
                        f'background:{c};display:inline-block;margin:1px;vertical-align:bottom;'
                        f'border-radius:2px"></div>')
        return '<div style="display:flex;align-items:flex-end;height:40px;margin-top:8px">' + ''.join(bars) + '</div>'

    ear_spark = sparkline(blinks.get('ear_per_frame', []),   '#2ecc71', threshold=0.21, invert=True)
    lip_spark = sparkline(lip.get('mouth_gaps_per_frame', []), '#9b59b6')
    hue_spark = sparkline(skin.get('per_frame_hue', []),     '#3498db')
    lap_spark = sparkline(boundary.get('laplacian_per_frame', []), '#e67e22', threshold=800)

    def feature_card(ex, spark, tech_label, tech_value):
        color_map = {'suspicious': '#fff5f5', 'normal': '#f5fff8', 'unknown': '#f9f9f9'}
        badge_map = {
            'suspicious': '<span style="background:#fde;color:#c0392b;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600">⚠ Suspicious</span>',
            'normal':     '<span style="background:#dfd;color:#1e8449;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600">✓ Normal</span>',
            'unknown':    '<span style="background:#eee;color:#888;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600">— Unknown</span>'
        }
        bg    = color_map.get(ex['verdict'], '#f9f9f9')
        badge = badge_map.get(ex['verdict'], '')
        return f"""
        <div style="background:#fff;border-radius:12px;padding:22px;box-shadow:0 1px 4px rgba(0,0,0,.07);border-top:3px solid {('#e74c3c' if ex['verdict']=='suspicious' else '#2ecc71' if ex['verdict']=='normal' else '#ccc')}">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:8px">
            <div style="font-size:17px;font-weight:600;color:#1a1a2e">{ex['icon']} {ex['title']}</div>
            {badge}
          </div>

          <div style="background:#f8f8f8;border-radius:8px;padding:12px 14px;margin-bottom:12px">
            <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">What we checked</div>
            <div style="font-size:13px;color:#444;line-height:1.6">{ex['what']}</div>
          </div>

          <div style="background:{bg};border-radius:8px;padding:12px 14px;margin-bottom:12px">
            <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">What we found</div>
            <div style="font-size:13px;color:#333;line-height:1.6;font-weight:500">{ex['finding']}</div>
          </div>

          <div style="background:#fffdf0;border-radius:8px;padding:12px 14px;margin-bottom:14px">
            <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">Why this matters</div>
            <div style="font-size:13px;color:#555;line-height:1.6">{ex['implication']}</div>
          </div>

          {spark}

          <details style="margin-top:12px">
            <summary style="font-size:11px;color:#aaa;cursor:pointer;user-select:none">Technical details</summary>
            <div style="font-size:11px;color:#999;margin-top:6px;font-family:monospace">{tech_label}: {tech_value}</div>
          </details>
        </div>"""

    skin_tech  = f"hue_std={skin.get('hue_std','N/A')}, consistency_score={skin.get('consistency_score','N/A')}"
    blink_tech = f"avg_ear={blinks.get('avg_ear','N/A')}, blink_count={blinks.get('blink_count','N/A')}"
    lip_tech   = f"motion_std={lip.get('motion_std','N/A')}, audio={lip.get('audio_available',False)}"
    bound_tech = f"avg_laplacian={boundary.get('avg_laplacian','N/A')}, threshold=800"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Deepfake Analysis Report — {job_id[:8]}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#f0f2f5;color:#1a1a2e;min-height:100vh}}
    .topbar{{background:#1a1a2e;padding:16px 32px;display:flex;align-items:center;gap:12px}}
    .topbar h1{{color:#fff;font-size:17px;font-weight:500;letter-spacing:-0.01em}}
    .topbar .jid{{color:#666;font-size:12px;margin-left:auto;font-family:monospace}}
    .wrap{{max-width:900px;margin:32px auto;padding:0 20px}}
    .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
    @media(max-width:640px){{.grid-2{{grid-template-columns:1fr}}}}
    details summary::-webkit-details-marker{{color:#aaa}}
  </style>
</head>
<body>

<div class="topbar">
  <h1>🔎 Deepfake Analysis Report</h1>
  <span class="jid">job: {job_id}</span>
</div>

<div class="wrap">

  <!-- ── Verdict ── -->
  <div style="background:{verdict_bg};border-radius:14px;padding:24px 28px;margin-bottom:28px;
              border:1px solid {'#f5c6c6' if prediction=='Fake' else '#b7e4c7'}">
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
      <div style="font-size:36px;font-weight:700;color:{verdict_color}">{verdict_emoji} {prediction}</div>
      <div>
        <div style="font-size:15px;color:#555">AI Confidence: <strong>{confidence}%</strong></div>
        <div style="font-size:12px;color:#888;margin-top:2px">Job ID: {job_id[:8]}...</div>
      </div>
    </div>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid {'#f5c6c6' if prediction=='Fake' else '#b7e4c7'}">
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">Summary</div>
      <div style="font-size:14px;color:#333;line-height:1.7">{summary}</div>
    </div>
  </div>

  <!-- ── Heatmap explanation ── -->
  <div style="background:#fff;border-radius:12px;padding:22px;box-shadow:0 1px 4px rgba(0,0,0,.07);margin-bottom:28px">
    <div style="font-size:17px;font-weight:600;color:#1a1a2e;margin-bottom:6px">🌡️ Where the AI looked</div>
    <div style="font-size:13px;color:#666;line-height:1.6;margin-bottom:16px">
      The heatmap below shows <strong>which parts of the face the AI focused on</strong> to make its decision.
      Red/yellow areas = high attention. Blue areas = less important.
      A real face usually shows attention spread naturally across features.
      A deepfake often shows unusual attention at face edges, eyes, or mouth — areas where face-swaps leave artefacts.
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      {'<div><img src="' + gc1_url + '" style="width:160px;height:160px;object-fit:cover;border-radius:8px;border:1px solid #eee"/><div style="font-size:11px;color:#888;text-align:center;margin-top:4px">Frame 1</div></div>' if gc1_url else '<div style="width:160px;height:160px;background:#f4f4f4;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:12px">No heatmap</div>'}
      {'<div><img src="' + gc2_url + '" style="width:160px;height:160px;object-fit:cover;border-radius:8px;border:1px solid #eee"/><div style="font-size:11px;color:#888;text-align:center;margin-top:4px">Frame 2</div></div>' if gc2_url else ''}
    </div>
  </div>

  <!-- ── Feature cards ── -->
  <div style="font-size:16px;font-weight:600;color:#1a1a2e;margin-bottom:14px">
    🧪 What the AI checked 
  </div>
  <div class="grid-2" style="margin-bottom:32px">
    {feature_card(skin_ex,     hue_spark,  'skin_tone',  skin_tech)}
    {feature_card(blink_ex,    ear_spark,  'eye_blinks', blink_tech)}
    {feature_card(lip_ex,      lip_spark,  'lip_sync',   lip_tech)}
    {feature_card(boundary_ex, lap_spark,  'boundary',   bound_tech)}
  </div>

  <!-- ── Raw JSON (collapsed) ── -->
  <details style="margin-bottom:40px">
    <summary style="cursor:pointer;font-size:13px;color:#888;padding:12px 16px;
                    background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.06)">
      📄 Raw technical data (for developers)
    </summary>
    <pre style="background:#f7f8fc;border-radius:0 0 8px 8px;padding:16px;font-size:11px;
                overflow-x:auto;color:#444;line-height:1.6">{json.dumps(xai, indent=2)}</pre>
  </details>

</div>
</body>
</html>"""
    return html


@app.route('/')
def index():
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/frontend/<path:filename>')
def serve_static(filename):
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
    return send_from_directory(frontend_dir, filename)

if __name__ == '__main__':
    app.run(debug=False)
