from __future__ import annotations
import io
import math
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import cv2
import mediapipe as mp


# ------------------
# Audio Cry Detector
# ------------------
class AudioDetectionResult(BaseModel):
    is_crying: bool
    score: float  # 0..1 confidence
    cry_type: Optional[str] = None  # e.g., hungry/tired/pain/unknown
    details: Optional[Dict[str, Any]] = None


class BaseAudioDetector:
    name = "base-audio-detector"

    def detect(self, wav_bytes: bytes, sr_hint: Optional[int] = None) -> AudioDetectionResult:
        raise NotImplementedError


class HeuristicAudioDetector(BaseAudioDetector):
    """Lightweight detector that works with or without numpy/librosa using simple energy/spectral rules.

    Heuristic (very naive):
      - Compute RMS loudness over frames.
      - Compute spectral centroid (if numpy available).
      - Cry likely if sustained RMS above threshold and centroid in vocal band.
      - Cry type (very naive):
           * high variability + high centroid -> pain
           * periodic envelope (slow modulation) -> hunger
           * low centroid + consistent -> tired
    """

    name = "heuristic-audio-detector"

    def _read_audio(self, wav_bytes: bytes) -> Tuple[Optional[Any], int]:
        if sf is not None:
            data, sr = sf.read(io.BytesIO(wav_bytes))
            if np is not None and data.ndim == 2:
                data = np.mean(data, axis=1)
            return data, sr
        # Fallback: try Python wave
        import wave
        with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
            sr = w.getframerate()
            n = w.getnframes()
            sampwidth = w.getsampwidth()
            nchan = w.getnchannels()
            pcm = w.readframes(n)
        if np is None:
            # Without numpy we can't really process; declare unknown
            return None, sr
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
        arr = np.frombuffer(pcm, dtype=dtype).astype(np.float32)
        if nchan > 1:
            arr = arr.reshape(-1, nchan).mean(axis=1)
        arr /= (np.max(np.abs(arr)) + 1e-9)
        return arr, sr

    def detect(self, wav_bytes: bytes, sr_hint: Optional[int] = None) -> AudioDetectionResult:
        data, sr = self._read_audio(wav_bytes)
        if data is None or np is None:
            # Not enough tooling to analyze
            return AudioDetectionResult(is_crying=False, score=0.0, cry_type=None, details={"note": "numpy/sound libs unavailable"})

        # Ensure float mono
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        # Normalize
        maxv = np.max(np.abs(data)) + 1e-9
        data = data / maxv

        # Frame-based analysis
        win = int(0.050 * sr)  # 50 ms
        hop = int(0.025 * sr)  # 25 ms
        if win <= 0:
            win, hop = 1024, 512
        frames = []
        for i in range(0, len(data) - win, hop):
            frames.append(data[i:i+win])
        if not frames:
            return AudioDetectionResult(is_crying=False, score=0.0, cry_type=None, details={"note": "audio too short"})

        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        loud = (rms > 0.08).astype(float)  # threshold tuned for normalized audio
        loud_ratio = float(loud.mean())

        # Spectral centroid (rough vocal energy proxy)
        centroids = []
        for f in frames:
            spec = np.fft.rfft(f * np.hanning(len(f)))
            mag = np.abs(spec) + 1e-9
            freqs = np.fft.rfftfreq(len(f), 1.0/sr)
            centroids.append(float((freqs @ mag) / mag.sum()))
        centroids = np.array(centroids)
        centroid_med = float(np.median(centroids))
        centroid_var = float(np.var(centroids))

        # Basic voicing/prosody
        rms_var = float(np.var(rms))

        # Decision logic
        crying_conf = 0.0
        # sustained loudness
        crying_conf += 0.6 * min(1.0, loud_ratio * 2)
        # centroid in 300-2500 Hz typical for infant cries (very broad)
        band_score = 1.0 if 300 <= centroid_med <= 2500 else max(0.0, 1.0 - abs(centroid_med - 1200)/2000)
        crying_conf += 0.3 * band_score
        # dynamics
        crying_conf += 0.1 * min(1.0, rms_var * 50)
        crying_conf = float(max(0.0, min(1.0, crying_conf)))

        is_crying = crying_conf > 0.5

        # Naive cry-type
        cry_type = None
        if is_crying:
            if centroid_med > 1800 and rms_var > 0.002:
                cry_type = "pain"
            elif 700 < centroid_med <= 1800 and rms_var < 0.001:
                cry_type = "tired"
            elif 400 < centroid_med <= 1200 and 0.001 <= rms_var <= 0.002:
                cry_type = "hunger"
            else:
                cry_type = "unknown"

        details = {
            "loud_ratio": loud_ratio,
            "centroid_med": centroid_med,
            "centroid_var": centroid_var,
            "rms_var": rms_var,
        }
        return AudioDetectionResult(is_crying=is_crying, score=crying_conf, cry_type=cry_type, details=details)


# ------------------
# Vision Analyzer
# ------------------
class VisionAnalysisResult(BaseModel):
    emotion: Optional[str] = None  # e.g., discomfort/pain/sleepy/neutral/happy/unknown
    posture: Optional[str] = None  # e.g., lying_still/moving_a_lot/upright/unknown
    details: Optional[Dict[str, Any]] = None


class BaseVisionAnalyzer:
    name = "base-vision-analyzer"

    def analyze(self, image_bytes: bytes) -> VisionAnalysisResult:
        raise NotImplementedError


class HeuristicVisionAnalyzer(BaseVisionAnalyzer):
    """Very lightweight vision heuristics.

    If OpenCV/MediaPipe are available, we do a bit more:
      - Detect face with Haar cascade (if cv2 present) to crop ROI.
      - Estimate eye aspect (closed eyes ~ sleepy) if mediapipe face mesh available.
      - Motion proxy (single image): blur + edge density as a proxy for movement/discomfort (very rough).

    Without CV libs, returns unknowns.
    """

    name = "heuristic-vision-analyzer"

    def analyze(self, image_bytes: bytes) -> VisionAnalysisResult:
        if cv2 is None or np is None:
            return VisionAnalysisResult(emotion=None, posture=None, details={"note": "opencv/numpy unavailable"})
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return VisionAnalysisResult(emotion=None, posture=None, details={"note": "decode_failed"})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Face detection (Haar) if available
        face_roi = gray
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda r: r[2]*r[3])
                face_roi = gray[y:y+fh, x:x+fw]
        except Exception:
            pass

        # Sharpness/edge density as discomfort proxy
        edges = cv2.Canny(face_roi, 50, 150)
        edge_ratio = float(edges.mean() / 255.0)

        # Blur (variance of Laplacian) – low value might mean sleepy/closed eyes
        lap_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())

        # Eye closure via MediaPipe (optional, coarse)
        eye_closed_prob = None
        if mp is not None:
            try:
                mp_face = mp.solutions.face_mesh
                with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = fm.process(rgb)
                    if res.multi_face_landmarks:
                        lms = res.multi_face_landmarks[0].landmark
                        # Simple eye aspect ratio proxy using subset of landmarks
                        def d(i, j):
                            dx = (lms[i].x - lms[j].x) * w
                            dy = (lms[i].y - lms[j].y) * h
                            return math.hypot(dx, dy)
                        # Using indices around left eye (approx MP: 33 outer corner, 159 upper lid, 145 lower lid)
                        ear = (d(159,145)) / (d(33, 133) + 1e-6)
                        eye_closed_prob = float(max(0.0, min(1.0, (0.22 - ear) / 0.15)))  # heuristic mapping
            except Exception:
                eye_closed_prob = None

        # Map to coarse labels
        emotion = "unknown"
        if edge_ratio > 0.14 and lap_var > 60:
            emotion = "discomfort"
        elif lap_var < 25:
            emotion = "sleepy"
        else:
            emotion = "neutral"

        posture = "unknown"
        # Single-frame posture guess (very rough): portrait orientation ~ upright, landscape ~ lying
        posture = "upright" if h > w else "lying_still"

        details = {
            "edge_ratio": edge_ratio,
            "lap_var": lap_var,
            "eye_closed_prob": eye_closed_prob,
            "face_detected": emotion != "unknown" or posture != "unknown",
        }
        return VisionAnalysisResult(emotion=emotion, posture=posture, details=details)


# -----------------------
# Rule-based Fusion Engine
# -----------------------
class FusionResult(BaseModel):
    predicted_reason: str
    scores: Dict[str, float]
    rationale: str


class RuleBasedFusion:
    reasons = ["hunger", "tired", "discomfort", "pain", "unknown"]

    def predict(self, audio: AudioDetectionResult, vision: VisionAnalysisResult) -> FusionResult:
        scores = {k: 0.0 for k in self.reasons}
        rationale_parts = []

        if audio.is_crying:
            # Start with audio hint
            if audio.cry_type and audio.cry_type in scores:
                scores[audio.cry_type] += 0.5
                rationale_parts.append(f"Audio suggests {audio.cry_type} (score {audio.score:.2f}).")
            else:
                scores["unknown"] += 0.2
                rationale_parts.append("Audio indicates crying but type is uncertain.")
        else:
            rationale_parts.append("Audio not strongly indicative of crying.")

        # Vision cues
        if vision.emotion == "sleepy":
            scores["tired"] += 0.4
            rationale_parts.append("Face looks sleepy.")
        elif vision.emotion == "discomfort":
            scores["discomfort"] += 0.4
            rationale_parts.append("Face shows discomfort/grimace cues.")

        if vision.posture == "lying_still":
            scores["tired"] += 0.2
            rationale_parts.append("Posture is lying still → possibly tired.")
        elif vision.posture == "upright":
            scores["discomfort"] += 0.1
            rationale_parts.append("Upright posture → slight discomfort likelihood.")

        # Normalize + fallback
        total = sum(scores.values())
        if total == 0:
            scores["unknown"] = 1.0
            total = 1.0
        scores = {k: float(v/total) for k, v in scores.items()}
        pred = max(scores, key=scores.get)
        rationale = " ".join(rationale_parts) if rationale_parts else "Insufficient signals; defaulting to unknown."
        return FusionResult(predicted_reason=pred, scores=scores, rationale=rationale)