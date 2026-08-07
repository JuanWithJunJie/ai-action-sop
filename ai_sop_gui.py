import csv
import json
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    raise ImportError("请先安装 mediapipe，例如: pip install mediapipe==0.10.14") from e


BASE_DIR = Path(__file__).resolve().parent
TIMELINE_CSV = BASE_DIR / "full_timeline_10videos_template.csv"
YOLO_MODEL_PATH = BASE_DIR / "runs_detect/yolov8s_mirror_v14_no_earlystop/weights/best.pt"
LSTM_MODEL_PATH = BASE_DIR / "lstm_runs_fine/best_lstm_fine.pt"
LSTM_CONFIG_PATH = BASE_DIR / "lstm_runs_fine/config.json"
RUNS_GUI_DIR = BASE_DIR / "runs_gui"
UI_BG_PATH = BASE_DIR / "背景图片.jpg"

CLASS_NAMES = ["base", "frame", "mirror", "screw"]
SOURCE_WINDOWS = [48, 72, 96]
S1_SWITCH_MARGIN = 0.05
S1_SWITCH_MIN_CONF = 0.40
CONFIRM_FRAMES_FIXED = 4
STEP_TIMEOUT_SEC = 6.0
YOLO_BOX_THICKNESS = 3
YOLO_TEXT_SIZE = 38
YOLO_TEXT_OFFSET_Y = 36
HAND_LANDMARK_THICKNESS = 4
HAND_CONNECTION_THICKNESS = 4
HAND_CIRCLE_RADIUS = 5
STEP_MIN_STAGE_SEC = 0.8
DEFAULT_ACTION_DEFS = [
    ("第一步：取料", "D1_pick_material"),
    ("第二步：撕膜", "D2_tear_film"),
    ("第三步：检测", "D3_inspect"),
    ("第四步：放料", "D4_place_material"),
]
ACTION_CN_MAP = {
    "D1_pick_material": "取料",
    "D2_tear_film": "撕膜",
    "D3_inspect": "检测",
    "D4_place_material": "放料",
    "background": "背景",
}
YOLO_CN_MAP = {
    "base": "底座",
    "frame": "骨架",
    "mirror": "镜面",
    "screw": "螺丝",
}


@dataclass
class RuntimeParams:
    yolo_conf: float
    lstm_conf: float
    confirm_frames: int
    show_boxes: bool
    show_keypoints: bool
    save_snapshots: bool
    save_log: bool


@dataclass
class ActionRuntime:
    index: int
    show: str
    fine_label: str
    done: bool = False
    hit_count: int = 0
    expected_conf: float = 0.0
    done_time_sec: Optional[float] = None
    snapshot_path: Optional[str] = None


class ActionLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def fine_label_from_row(step_id, _target_id, _event_type):
    s = str(step_id).strip().upper()
    if s == "D1":
        return "D1_pick_material"
    if s == "D2":
        return "D2_tear_film"
    if s == "D3":
        return "D3_inspect"
    if s == "D4":
        return "D4_place_material"
    return "background"


def build_feature_row(detections, hands, frame_w, frame_h):
    det_feat = {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in CLASS_NAMES}
    for d in detections:
        cls_name = d.get("cls_name")
        if cls_name not in det_feat:
            continue
        conf = float(d.get("conf", 0.0))
        x1, y1, x2, y2 = d.get("xyxy", [0, 0, 0, 0])
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0

        fw = max(float(frame_w), 1.0)
        fh = max(float(frame_h), 1.0)
        if conf > det_feat[cls_name][0]:
            det_feat[cls_name] = [conf, cx / fw, cy / fh, w / fw, h / fh]

    det_vec = []
    for cname in CLASS_NAMES:
        det_vec.extend(det_feat[cname])

    hand_vec = [0.0] * (2 * 21 * 3)
    for hi, hand in enumerate(hands[:2]):
        lms = hand.get("landmarks", [])
        for li, lm in enumerate(lms[:21]):
            base = hi * 63 + li * 3
            hand_vec[base] = float(lm[0]) if len(lm) > 0 else 0.0
            hand_vec[base + 1] = float(lm[1]) if len(lm) > 1 else 0.0
            hand_vec[base + 2] = float(lm[2]) if len(lm) > 2 else 0.0

    return np.array(det_vec + hand_vec, dtype=np.float32)


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    bytes_per_line = c * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def to_beijing_time_str() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def action_to_cn(label: str) -> str:
    return ACTION_CN_MAP.get(label, label)


def _load_font(font_size: int = YOLO_TEXT_SIZE) -> ImageFont.FreeTypeFont:
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_yolo_boxes_cn(frame: np.ndarray, boxes, names_map: dict):
    vis = frame.copy()
    if boxes is None:
        return vis

    pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(YOLO_TEXT_SIZE)

    for box in boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = names_map.get(cls_id, str(cls_id))
        cls_cn = YOLO_CN_MAP.get(cls_name, cls_name)
        draw.rectangle([x1, y1, x2, y2], outline=(60, 220, 120), width=YOLO_BOX_THICKNESS)
        draw.text((x1, max(2, y1 - YOLO_TEXT_OFFSET_Y)), cls_cn, fill=(60, 220, 120), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class InferenceWorker(QThread):
    sig_frame = pyqtSignal(QImage)
    sig_status = pyqtSignal(dict)
    sig_action = pyqtSignal(dict)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, video_path: str, params: RuntimeParams, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.params = params
        self._stop = False
        self._pause = False

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

        self.lstm = None
        self.yolo = None
        self.hands = None

        self.actions: List[ActionRuntime] = []
        self.expected_idx = 0
        self.current_hit = 0
        self.expected_stage_start_sec = 0.0
        self.last_expected_idx = -1
        self.cycle = 1

        self.run_dir: Optional[Path] = None
        self.events: List[dict] = []

    def stop(self):
        self._stop = True

    def toggle_pause(self):
        self._pause = not self._pause

    def _load_models(self):
        config = json.loads(LSTM_CONFIG_PATH.read_text(encoding="utf-8"))
        self.id2label = {int(x["id"]): x["label"] for x in config["label_map"]}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.lstm = ActionLSTM(
            config["input_size"],
            config["hidden_size"],
            config["num_layers"],
            config["num_classes"],
        ).to(self.device)
        self.lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
        self.lstm.eval()

        self.yolo = YOLO(str(YOLO_MODEL_PATH))
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _build_actions(self):
        self.actions = []

        if TIMELINE_CSV.exists():
            try:
                df = pd.read_csv(TIMELINE_CSV)
                video_name = self.video_path.name
                sdf = df[df["video_name"] == video_name].copy().sort_values(["start_sec", "end_sec"])
                if not sdf.empty:
                    for i, r in sdf.reset_index(drop=True).iterrows():
                        fine = fine_label_from_row(r["step_id"], r.get("target_id", ""), r.get("event_type", ""))
                        show_cn = ACTION_CN_MAP.get(fine, fine)
                        show = f"第{i + 1}步：{show_cn}"

                        self.actions.append(
                            ActionRuntime(
                                index=i,
                                show=show,
                                fine_label=fine,
                            )
                        )
            except Exception:
                self.actions = []

        if not self.actions:
            for i, (show, fine) in enumerate(DEFAULT_ACTION_DEFS):
                self.actions.append(ActionRuntime(index=i, show=show, fine_label=fine))

    def _init_run_dir(self):
        RUNS_GUI_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_GUI_DIR / f"{self.video_path.stem}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "snapshots").mkdir(exist_ok=True)

    def _save_snapshot(self, frame_bgr: np.ndarray, action: ActionRuntime, t_sec: float) -> str:
        assert self.run_dir is not None
        fname = f"{action.index + 1:02d}_{action.fine_label}_{t_sec:.2f}.jpg".replace("/", "_")
        out = self.run_dir / "snapshots" / fname
        cv2.imwrite(str(out), frame_bgr)
        return str(out)

    def _calc_expected_conf(self, probs: torch.Tensor, expected_label: str) -> float:
        idx = self.label2id.get(expected_label)
        if idx is None:
            return 0.0
        return float(probs[0, idx].item())

    def _calc_s1_sibling_conf(self, _probs: torch.Tensor, _expected_label: str) -> float:
        # 当前场景没有S1双孔位，直接返回0
        return 0.0

    def _apply_expected_rules(self, _expected_action: ActionRuntime, expected_conf: float, _sibling_conf: float, t_sec: float) -> float:
        # 最小步骤持续时间门槛，避免切换瞬间误判
        stage_elapsed = t_sec - self.expected_stage_start_sec
        if stage_elapsed < STEP_MIN_STAGE_SEC:
            return 0.0

        return expected_conf

    def run(self):
        try:
            self._load_models()
            self._build_actions()
            self._init_run_dir()

            self.sig_status.emit({"action_defs": [a.show for a in self.actions], "total": len(self.actions)})

            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_id = 0
            feat_queue = deque(maxlen=max(SOURCE_WINDOWS))
            pred_history = deque(maxlen=5)

            current_pred = "background"

            while not self._stop:
                if self._pause:
                    self.msleep(30)
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                t_sec = frame_id / fps
                frame_h, frame_w = frame.shape[:2]

                res = self.yolo.predict(frame, verbose=False, conf=self.params.yolo_conf, device=self.device)[0]
                if self.params.show_boxes:
                    vis = draw_yolo_boxes_cn(frame, res.boxes, self.yolo.names)
                else:
                    vis = frame.copy()

                detections = []
                if res.boxes is not None:
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        detections.append(
                            {
                                "cls_name": self.yolo.names.get(cls_id, str(cls_id)),
                                "conf": float(box.conf[0]),
                                "xyxy": [x1, y1, x2, y2],
                            }
                        )

                rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hr = self.hands.process(rgb_raw)
                hands_out = []
                if hr.multi_hand_landmarks:
                    for hidx, hand_lm in enumerate(hr.multi_hand_landmarks):
                        if self.params.show_keypoints:
                            mp_draw.draw_landmarks(
                                vis,
                                hand_lm,
                                mp_hands.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(0, 255, 140), thickness=HAND_LANDMARK_THICKNESS, circle_radius=HAND_CIRCLE_RADIUS),
                                mp_draw.DrawingSpec(color=(0, 220, 120), thickness=HAND_CONNECTION_THICKNESS, circle_radius=HAND_CIRCLE_RADIUS),
                            )
                        lms = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_lm.landmark]
                        hands_out.append({"hand_index": hidx, "landmarks": lms})

                feat_queue.append(build_feature_row(detections, hands_out, frame_w, frame_h))

                expected_show = "DONE"
                expected_conf = 0.0
                top3_text = "-"

                if self.expected_idx < len(self.actions):
                    expected_action = self.actions[self.expected_idx]
                    expected_show = expected_action.show
                else:
                    expected_action = None

                if expected_action is not None and self.expected_idx != self.last_expected_idx:
                    self.expected_stage_start_sec = t_sec
                    self.last_expected_idx = self.expected_idx

                if len(feat_queue) >= min(SOURCE_WINDOWS):
                    feat_arr = np.stack(feat_queue, axis=0)
                    scale_preds = []
                    expected_confs = []
                    sibling_confs = []
                    top3_candidates = []

                    for src_len in SOURCE_WINDOWS:
                        if len(feat_arr) < src_len:
                            continue

                        src_seq = feat_arr[-src_len:]
                        pos = np.linspace(0, src_len - 1, 48)
                        idx = np.clip(np.round(pos).astype(int), 0, src_len - 1)
                        seq = src_seq[idx]

                        xb = torch.from_numpy(seq[None, ...]).float().to(self.device)
                        with torch.no_grad():
                            logits = self.lstm(xb)
                            probs = torch.softmax(logits, dim=1)

                        probs_np = probs[0].detach().cpu().numpy()
                        for i_cls, p_cls in enumerate(probs_np):
                            lb = self.id2label.get(i_cls, "")
                            if lb and lb != "background":
                                top3_candidates.append((lb, float(p_cls)))

                        pid = int(torch.argmax(probs, dim=1).item())
                        scale_preds.append(self.id2label.get(pid, "background"))

                        if expected_action is not None:
                            expected_confs.append(self._calc_expected_conf(probs, expected_action.fine_label))
                            sibling_confs.append(self._calc_s1_sibling_conf(probs, expected_action.fine_label))


                    if scale_preds:
                        if top3_candidates:
                            agg = {}
                            for lb, p in top3_candidates:
                                agg[lb] = max(agg.get(lb, 0.0), p)
                            top3_sorted = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:3]
                            top3_text = " | ".join([f"{action_to_cn(lb)}:{p:.2f}" for lb, p in top3_sorted])

                        votes = {}
                        for p in scale_preds:
                            votes[p] = votes.get(p, 0) + 1
                        current_pred = max(votes, key=votes.get)
                        pred_history.append(current_pred)
                        smooth_votes = {}
                        for p in pred_history:
                            smooth_votes[p] = smooth_votes.get(p, 0) + 1
                        current_pred = max(smooth_votes, key=smooth_votes.get)

                        if expected_action is not None:
                            expected_conf = float(np.mean(expected_confs)) if expected_confs else 0.0
                            sibling_conf = float(np.mean(sibling_confs)) if sibling_confs else 0.0
                            expected_conf = self._apply_expected_rules(expected_action, expected_conf, sibling_conf, t_sec)

                            if expected_conf >= self.params.lstm_conf:
                                self.current_hit += 1
                            else:
                                self.current_hit = 0

                            expected_action.expected_conf = expected_conf
                            expected_action.hit_count = self.current_hit

                            if self.current_hit >= self.params.confirm_frames:
                                expected_action.done = True
                                expected_action.done_time_sec = t_sec

                                if self.params.save_snapshots:
                                    expected_action.snapshot_path = self._save_snapshot(vis, expected_action, t_sec)

                                done_time_beijing = to_beijing_time_str()
                                self.events.append(
                                    {
                                        "cycle": self.cycle,
                                        "index": expected_action.index + 1,
                                        "action": expected_action.show,
                                        "fine_label": expected_action.fine_label,
                                        "done_time_sec": round(t_sec, 3),
                                        "done_time_beijing": done_time_beijing,
                                        "snapshot_path": expected_action.snapshot_path or "",
                                        "status": "完成",
                                    }
                                )

                                self.sig_action.emit(
                                    {
                                        "index": expected_action.index,
                                        "status": "完成",
                                        "info": f"完成时间（北京时间）: {done_time_beijing}",
                                        "snapshot": expected_action.snapshot_path,
                                    }
                                )

                                self.expected_idx += 1
                                self.current_hit = 0

                                if self.expected_idx >= len(self.actions):
                                    self.expected_idx = 0
                                    self.cycle += 1
                                    self.last_expected_idx = -1
                                    for a in self.actions:
                                        a.done = False
                                        a.hit_count = 0
                                        a.expected_conf = 0.0
                                        a.done_time_sec = None
                                        a.snapshot_path = None
                                    self.sig_status.emit({
                                        "action_defs": [a.show for a in self.actions],
                                        "total": len(self.actions),
                                        "cycle": self.cycle,
                                    })
                            elif t_sec - self.expected_stage_start_sec > STEP_TIMEOUT_SEC:
                                done_time_beijing = to_beijing_time_str()
                                self.events.append(
                                    {
                                        "cycle": self.cycle,
                                        "index": expected_action.index + 1,
                                        "action": expected_action.show,
                                        "fine_label": expected_action.fine_label,
                                        "done_time_sec": round(t_sec, 3),
                                        "done_time_beijing": done_time_beijing,
                                        "snapshot_path": "",
                                        "status": "超时跳过",
                                    }
                                )

                                self.sig_action.emit(
                                    {
                                        "index": expected_action.index,
                                        "status": "超时跳过",
                                        "info": f"超时跳过 ({STEP_TIMEOUT_SEC}s)",
                                        "snapshot": None,
                                    }
                                )

                                self.expected_idx += 1
                                self.current_hit = 0

                                if self.expected_idx >= len(self.actions):
                                    self.expected_idx = 0
                                    self.cycle += 1
                                    self.last_expected_idx = -1
                                    for a in self.actions:
                                        a.done = False
                                        a.hit_count = 0
                                        a.expected_conf = 0.0
                                        a.done_time_sec = None
                                        a.snapshot_path = None
                                    self.sig_status.emit({
                                        "action_defs": [a.show for a in self.actions],
                                        "total": len(self.actions),
                                        "cycle": self.cycle,
                                    })

                status = {
                    "frame_id": frame_id,
                    "time_sec": t_sec,
                    "current_pred": action_to_cn(current_pred),
                    "expected_show": expected_show,
                    "expected_conf": expected_conf,
                    "lstm_conf": self.params.lstm_conf,
                    "hit": self.current_hit,
                    "confirm_frames": self.params.confirm_frames,
                    "top3": top3_text,
                    "progress": self.expected_idx,
                    "total": len(self.actions),
                    "cycle": self.cycle,
                }
                self.sig_status.emit(status)
                self.sig_frame.emit(bgr_to_qimage(vis))

                frame_id += 1

            cap.release()
            if self.hands is not None:
                self.hands.close()

            result = {"run_dir": str(self.run_dir) if self.run_dir else "", "events": self.events}
            if self.params.save_log and self.run_dir:
                self._save_logs()

            self.sig_finished.emit(result)

        except Exception as e:
            self.sig_error.emit(str(e))

    def _save_logs(self):
        assert self.run_dir is not None
        json_path = self.run_dir / "result.json"
        csv_path = self.run_dir / "events.csv"

        result_obj = {
            "video": self.video_path.name,
            "total_cycles": self.cycle,
            "completed_events": len(self.events),
            "total_steps": len(self.actions),
            "events": self.events,
        }
        json_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["cycle", "index", "action", "fine_label", "done_time_sec", "done_time_beijing", "snapshot_path"],
            )
            writer.writeheader()
            for row in self.events:
                writer.writerow(row)


class StepCard(QFrame):
    def __init__(self, idx: int, step_id: str, name: str, en_name: str, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.step_id = step_id
        self.name = name
        self.en_name = en_name
        self.status = "pending"
        self.confidence = 0.0
        self._scale = 1.0

        self.setMinimumSize(240, 200)
        self.setMaximumHeight(240)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.lbl_num = QLabel(step_id)
        self.lbl_num.setFixedSize(28, 28)
        self.lbl_num.setAlignment(Qt.AlignCenter)
        self.lbl_num.setStyleSheet("background: rgba(255,255,255,0.06); border-radius: 5px; font-weight: 700; font-size: 13px; color: #5a6b7d;")

        name_box = QVBoxLayout()
        name_box.setSpacing(0)
        self.lbl_name = QLabel(name)
        self.lbl_name.setStyleSheet("font-size: 15px; font-weight: 600; color: #d8e4f0;")
        self.lbl_en = QLabel(en_name)
        self.lbl_en.setStyleSheet("font-size: 10px; color: #5a6b7d; font-family: 'Consolas';")
        name_box.addWidget(self.lbl_name)
        name_box.addWidget(self.lbl_en)

        header.addWidget(self.lbl_num)
        header.addLayout(name_box)
        header.addStretch()
        layout.addLayout(header)

        self.lbl_status = QLabel("未开始")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #5a6b7d; padding-top: 4px;")
        layout.addWidget(self.lbl_status)

        self.conf_widget = QWidget()
        self.conf_widget.setFixedHeight(20)
        self.conf_layout = QHBoxLayout(self.conf_widget)
        self.conf_layout.setContentsMargins(0, 4, 0, 0)
        self.conf_layout.setSpacing(6)
        self.lbl_conf_text = QLabel("")
        self.lbl_conf_text.setStyleSheet("font-size: 11px; font-family: 'Consolas'; color: #5a6b7d;")
        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet("QProgressBar{background:rgba(0,0,0,0.3);border-radius:3px;border:none;}QProgressBar::chunk{border-radius:3px;background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ffaa00,stop:1 #00ff88);}")
        self.conf_layout.addWidget(self.lbl_conf_text)
        self.conf_layout.addWidget(self.conf_bar)
        self.conf_widget.setVisible(False)
        layout.addWidget(self.conf_widget)

        self.lbl_snapshot = QLabel()
        self.lbl_snapshot.setAlignment(Qt.AlignCenter)
        self.lbl_snapshot.setStyleSheet("background: rgba(0,0,0,0.3); border: 1px solid rgba(0,212,255,0.12); border-radius: 6px; color: #3a4a5a; font-size: 10px;")
        self.lbl_snapshot.setMinimumHeight(60)
        self.lbl_snapshot.setText("截图")
        self.lbl_snapshot.setVisible(False)
        layout.addWidget(self.lbl_snapshot, 1)

    def _apply_style(self):
        border_color = "rgba(0,212,255,0.12)"
        bg = "rgba(13,20,31,0.85)"
        if self.status == "active":
            border_color = "rgba(255,170,0,0.35)"
            bg = "rgba(13,20,31,0.95)"
        elif self.status == "done":
            border_color = "rgba(0,255,136,0.3)"
        elif self.status == "timeout":
            border_color = "rgba(255,68,102,0.3)"

        self.setStyleSheet(
            f"StepCard {{ background: {bg}; border: 1px solid {border_color}; border-radius: 10px; }}"
        )

    def set_scale(self, scale: float):
        self._scale = scale
        s = lambda v: max(1, int(v * scale))
        self.lbl_name.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; color: #d8e4f0;")
        self.lbl_en.setStyleSheet(f"font-size: {s(12)}px; color: #5a6b7d; font-family: 'Consolas';")
        self.lbl_num.setStyleSheet(f"background: rgba(255,255,255,0.06); border-radius: 5px; font-weight: 700; font-size: {s(15)}px; color: #5a6b7d;")
        num_size = s(32)
        self.lbl_num.setFixedSize(num_size, num_size)
        self.lbl_status.setStyleSheet(f"font-size: {s(14)}px; color: #5a6b7d; padding-top: 4px;")
        self.lbl_conf_text.setStyleSheet(f"font-size: {s(13)}px; font-family: 'Consolas'; color: #5a6b7d;")
        self.lbl_snapshot.setStyleSheet(f"background: rgba(0,0,0,0.3); border: 1px solid rgba(0,212,255,0.12); border-radius: 6px; color: #3a4a5a; font-size: {s(12)}px;")
        self.set_status(self.status)

    def set_status(self, status: str, info: str = ""):
        self.status = status
        self._apply_style()

        status_map = {
            "pending": ("未开始", "#5a6b7d"),
            "进行中": ("进行中", "#ffaa00"),
            "active": ("进行中", "#ffaa00"),
            "完成": ("已完成", "#00ff88"),
            "done": ("已完成", "#00ff88"),
            "超时跳过": ("超时跳过", "#ff4466"),
            "timeout": ("超时跳过", "#ff4466"),
        }
        text, color = status_map.get(status, (status, "#5a6b7d"))
        s = lambda v: max(1, int(v * self._scale))
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"font-size: {s(12)}px; color: {color}; padding-top: 4px;")

        num_color = "#5a6b7d"
        num_bg = "rgba(255,255,255,0.06)"
        if status in ("进行中", "active"):
            num_color = "#ffaa00"; num_bg = "rgba(255,170,0,0.15)"
        elif status in ("完成", "done"):
            num_color = "#00ff88"; num_bg = "rgba(0,255,136,0.15)"
        elif status in ("超时跳过", "timeout"):
            num_color = "#ff4466"; num_bg = "rgba(255,68,102,0.15)"
        self.lbl_num.setStyleSheet(f"background: {num_bg}; border-radius: 5px; font-weight: 700; font-size: {s(13)}px; color: {num_color};")

        self.conf_widget.setVisible(status in ("进行中", "active"))
        if status not in ("完成", "done"):
            self.lbl_snapshot.setVisible(False)
            self.lbl_snapshot.setPixmap(QPixmap())

    def set_confidence(self, conf: float):
        self.confidence = conf
        self.conf_bar.setValue(int(conf * 100))
        self.lbl_conf_text.setText(f"置信度: {conf*100:.1f}%")

    def set_snapshot(self, snapshot_path: Optional[str]):
        if not snapshot_path:
            return
        p = Path(snapshot_path)
        if not p.exists():
            return
        pix = QPixmap(str(p))
        if pix.isNull():
            return
        scaled = pix.scaled(self.lbl_snapshot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_snapshot.setPixmap(scaled)
        self.lbl_snapshot.setText("")
        self.lbl_snapshot.setVisible(True)

    def apply_style(self, alpha: int):
        self._apply_style()


class StatRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rate = 100
        self.setFixedSize(64, 64)

    def set_rate(self, rate: int):
        self.rate = max(0, min(100, rate))
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QColor, QFont as QFont2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect_size = 50
        x = (self.width() - rect_size) // 2
        y = (self.height() - rect_size) // 2

        pen = QPen(QColor(0, 212, 255, 25))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawArc(x, y, rect_size, rect_size, 0, 360 * 16)

        color = QColor(0, 255, 136) if self.rate >= 90 else QColor(255, 170, 0) if self.rate >= 70 else QColor(255, 68, 102)
        pen = QPen(color)
        pen.setWidth(4)
        painter.setPen(pen)
        import math
        span = int(self.rate / 100 * 360 * 16)
        painter.drawArc(x, y, rect_size, rect_size, 90 * 16, -span)

        painter.setPen(QColor(216, 228, 240))
        font = QFont2("Consolas", 10, QFont2.Bold)
        painter.setFont(font)
        text = f"{self.rate}%"
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        painter.drawText((self.width() - tw) // 2, self.height() // 2 + 5, text)

        painter.end()


class EventItem(QWidget):
    def __init__(self, step_name: str, event_type: str, cycle: int, time_str: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        icon = QLabel("✓" if event_type == "done" else "!")
        icon.setFixedSize(20, 20)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(
            f"background: {'rgba(0,255,136,0.15)' if event_type == 'done' else 'rgba(255,68,102,0.15)'}; "
            f"border-radius: 4px; font-size: 11px; font-weight: 700; "
            f"color: {'#00ff88' if event_type == 'done' else '#ff4466'};"
        )
        layout.addWidget(icon)

        info = QVBoxLayout()
        info.setSpacing(0)
        lbl_action = QLabel(step_name)
        lbl_action.setStyleSheet("font-size: 12px; font-weight: 600; color: #d8e4f0;")
        lbl_time = QLabel(time_str)
        lbl_time.setStyleSheet("font-size: 10px; color: #5a6b7d; font-family: 'Consolas';")
        info.addWidget(lbl_action)
        info.addWidget(lbl_time)
        layout.addLayout(info)
        layout.addStretch()

        lbl_cycle = QLabel(f"C{cycle}")
        lbl_cycle.setStyleSheet("font-size: 10px; color: #00d4ff; font-family: 'Consolas';")
        layout.addWidget(lbl_cycle)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-SOP Vision | 手机制造智能SOP系统")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1920, 1080)
        self.video_path: Optional[Path] = None
        self.worker: Optional[InferenceWorker] = None
        self.last_run_dir: Optional[Path] = None
        self.pass_count = 0
        self.skip_count = 0

        self.cards: List[StepCard] = []
        self.event_items: List[EventItem] = []
        self._font_scale = 1.0
        self._apply_theme()
        self._build_ui()
        self._apply_font_scale(1.0)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #060a12; }
            QWidget { color: #d8e4f0; font-family: 'Microsoft YaHei'; font-size: 15px; }
            QLabel { background: transparent; color: #d8e4f0; }
            QFrame#header { background: #0d141f; border-bottom: 1px solid rgba(0,212,255,0.12); }
            QFrame#footer { background: #0d141f; border-top: 1px solid rgba(0,212,255,0.12); }
            QFrame#panel { background: rgba(13,20,31,0.85); border: 1px solid rgba(0,212,255,0.12); border-radius: 12px; }
            QFrame#videoSection { background: #000; border: 1px solid rgba(0,212,255,0.12); border-radius: 12px; }
            QFrame#titleBar { background: #0d141f; border-bottom: 1px solid rgba(0,212,255,0.15); }
            QLineEdit {
                background: rgba(0,0,0,0.3); border: 1px solid rgba(0,212,255,0.12);
                border-radius: 4px; padding: 4px 8px; color: #d8e4f0;
                font-family: 'Consolas'; font-size: 14px; max-width: 60px;
            }
            QPushButton {
                background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.35);
                border-radius: 8px; padding: 8px 20px; color: #00d4ff;
                font-size: 15px; font-weight: 600;
            }
            QPushButton:hover { background: rgba(0,212,255,0.15); border-color: #00d4ff; }
            QPushButton:pressed { background: rgba(0,212,255,0.05); }
            QPushButton:disabled { color: #2a3a4a; border-color: rgba(0,212,255,0.08); background: rgba(0,0,0,0.2); }
            QPushButton#closeBtn { background: rgba(255,68,102,0.08); border-color: rgba(255,68,102,0.3); color: #ff4466; border-radius: 0px; border: none; padding: 0px; font-size: 18px; }
            QPushButton#closeBtn:hover { background: rgba(255,68,102,0.2); }
            QPushButton#winBtn { background: transparent; border: none; color: #5a6b7d; border-radius: 0px; padding: 0px; font-size: 18px; }
            QPushButton#winBtn:hover { background: rgba(0,212,255,0.08); color: #00d4ff; }
            QCheckBox { color: #d8e4f0; font-size: 14px; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid rgba(0,212,255,0.35); background: rgba(0,0,0,0.3); }
            QCheckBox::indicator:checked { background: #00d4ff; border-color: #00d4ff; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 4px; background: transparent; }
            QScrollBar::handle:vertical { background: rgba(0,212,255,0.35); border-radius: 2px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # === Header ===
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(56)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 8, 24, 8)
        h_layout.setSpacing(20)

        logo = QLabel("AI")
        logo.setFixedSize(32, 32)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #00d4ff,stop:1 #0088aa); border-radius: 6px; font-weight: 700; font-size: 16px; color: #000;")
        h_layout.addWidget(logo)

        title = QLabel("SOP·VISION")
        title.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 1px; color: #d8e4f0;")
        title_sp = QLabel("手机制造智能SOP系统")
        title_sp.setStyleSheet("font-size: 12px; color: #5a6b7d;")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(title_sp)
        h_layout.addLayout(title_box)

        h_layout.addSpacing(30)

        for label, value in [("工厂", "深圳智造工厂"), ("产线", "A线·手机组装"), ("工位", "W-07 镜面贴合"), ("班次", "白班 A组")]:
            info = QVBoxLayout()
            info.setSpacing(1)
            lbl_l = QLabel(label)
            lbl_l.setStyleSheet("font-size: 9px; color: #5a6b7d; text-transform: uppercase; letter-spacing: 1.5px;")
            lbl_v = QLabel(value)
            lbl_v.setStyleSheet("font-size: 14px; font-weight: 600; color: #d8e4f0;")
            info.addWidget(lbl_l)
            info.addWidget(lbl_v)
            h_layout.addLayout(info)
            h_layout.addSpacing(20)

        h_layout.addStretch()

        self.lbl_status_pill = QLabel("● 系统在线")
        self.lbl_status_pill.setStyleSheet("background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: #00ff88;")
        h_layout.addWidget(self.lbl_status_pill)

        self.lbl_clock = QLabel("--:--:--")
        self.lbl_clock.setStyleSheet("font-size: 15px; color: #00d4ff; font-family: 'Consolas'; letter-spacing: 1px;")
        h_layout.addWidget(self.lbl_clock)

        h_layout.addSpacing(16)

        self.btn_min = QPushButton("—")
        self.btn_min.setObjectName("winBtn")
        self.btn_min.setFixedSize(46, 56)
        self.btn_min.clicked.connect(self.showMinimized)
        h_layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("▢")
        self.btn_max.setObjectName("winBtn")
        self.btn_max.setFixedSize(46, 56)
        self.btn_max.clicked.connect(self._toggle_max)
        h_layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("closeBtn")
        self.btn_close.setFixedSize(46, 56)
        self.btn_close.clicked.connect(self.close)
        h_layout.addWidget(self.btn_close)

        self._header_frame = header
        header.mousePressEvent = self._on_header_press
        header.mouseMoveEvent = self._on_header_move
        header.mouseDoubleClickEvent = lambda e: self._toggle_max()

        root.addWidget(header)

        # === Main Content ===
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left column
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # Video section
        video_frame = QFrame()
        video_frame.setObjectName("videoSection")
        v_layout = QVBoxLayout(video_frame)
        v_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("点击「导入视频」选择视频文件")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0a0f18,stop:1 #060a10); color: #5a6b7d; font-size: 14px; border: none;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_layout.addWidget(self.video_label)

        self.video_bottom = QFrame()
        self.video_bottom.setStyleSheet("background: qlineargradient(x1:0,y1:1,x2:0,y2:0,stop:0 rgba(0,0,0,0.85),stop:1 transparent); border: none;")
        self.video_bottom.setFixedHeight(50)
        vb_layout = QHBoxLayout(self.video_bottom)
        vb_layout.setContentsMargins(16, 6, 16, 6)
        vb_layout.setSpacing(16)

        self.bottom_labels = {}
        for key, lbl in [("current", ("当前动作", "#00d4ff")), ("expected", ("期望动作", "#ffaa00")), ("frame", ("帧号", "#5a6b7d")), ("time", ("时间", "#5a6b7d")), ("hit", ("命中", "#00ff88"))]:
            box = QVBoxLayout()
            box.setSpacing(0)
            l = QLabel(lbl[0])
            l.setStyleSheet("font-size: 9px; color: #5a6b7d; text-transform: uppercase; letter-spacing: 1px; border: none;")
            v = QLabel("--")
            v.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {lbl[1]}; font-family: {'Consolas' if key in ('frame','time','hit') else 'Microsoft YaHei'}; border: none;")
            box.addWidget(l)
            box.addWidget(v)
            self.bottom_labels[key] = v
            vb_layout.addLayout(box)

        self.video_bottom.setVisible(False)
        v_layout.addWidget(self.video_bottom)

        left_col.addWidget(video_frame, 1)

        # Step cards
        steps_frame = QFrame()
        steps_layout = QHBoxLayout(steps_frame)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(10)

        step_defs = [
            ("D1", "取料", "D1_pick_material"),
            ("D2", "撕膜", "D2_tear_film"),
            ("D3", "检测", "D3_inspect"),
            ("D4", "放料", "D4_place_material"),
        ]
        for i, (sid, name, en) in enumerate(step_defs):
            card = StepCard(i, sid, name, en)
            self.cards.append(card)
            steps_layout.addWidget(card)
        steps_layout.addStretch()

        left_col.addWidget(steps_frame)

        main_layout.addLayout(left_col, 1)

        # Right panel
        right_frame = QFrame()
        right_frame.setFixedWidth(380)
        right_col = QVBoxLayout(right_frame)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(12)

        # Cycle counter
        cycle_panel = QFrame()
        cycle_panel.setObjectName("panel")
        cp_layout = QVBoxLayout(cycle_panel)
        cp_layout.setContentsMargins(14, 14, 14, 14)
        cp_layout.setSpacing(8)

        lbl_title = QLabel("生产周期")
        lbl_title.setStyleSheet("font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #00d4ff;")
        cp_layout.addWidget(lbl_title)

        cycle_row = QHBoxLayout()
        cycle_left = QVBoxLayout()
        cycle_left.setSpacing(0)
        self.lbl_cycle = QLabel("1")
        self.lbl_cycle.setStyleSheet("font-size: 42px; font-weight: 700; font-family: 'Consolas'; color: #00d4ff;")
        lbl_cycle_l = QLabel("当前轮次")
        lbl_cycle_l.setStyleSheet("font-size: 11px; color: #5a6b7d; text-transform: uppercase; letter-spacing: 1px;")
        cycle_left.addWidget(self.lbl_cycle)
        cycle_left.addWidget(lbl_cycle_l)
        cycle_row.addLayout(cycle_left)
        cycle_row.addStretch()

        cycle_right = QVBoxLayout()
        cycle_right.setSpacing(0)
        cycle_right.setAlignment(Qt.AlignRight)
        self.lbl_completed = QLabel("0")
        self.lbl_completed.setStyleSheet("font-size: 24px; font-weight: 700; font-family: 'Consolas'; color: #00ff88;")
        lbl_completed_l = QLabel("已完成")
        lbl_completed_l.setStyleSheet("font-size: 11px; color: #5a6b7d; text-transform: uppercase; letter-spacing: 1px;")
        cycle_right.addWidget(self.lbl_completed)
        cycle_right.addWidget(lbl_completed_l)
        cycle_row.addLayout(cycle_right)
        cp_layout.addLayout(cycle_row)

        right_col.addWidget(cycle_panel)

        # Realtime metrics
        rt_panel = QFrame()
        rt_panel.setObjectName("panel")
        rt_layout = QVBoxLayout(rt_panel)
        rt_layout.setContentsMargins(14, 14, 14, 14)
        rt_layout.setSpacing(8)

        rt_title = QLabel("实时推理")
        rt_title.setStyleSheet("font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #00d4ff;")
        rt_layout.addWidget(rt_title)

        metric_grid = QGridLayout()
        metric_grid.setSpacing(8)
        for i, (key, label) in enumerate([("current", "当前动作"), ("expected", "期望动作"), ("conf", "置信度"), ("hit", "命中帧")]):
            box = QFrame()
            box.setStyleSheet("background: rgba(0,0,0,0.3); border: 1px solid rgba(0,212,255,0.12); border-radius: 8px; padding: 10px 12px;")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            bl.setSpacing(2)
            l = QLabel(label)
            l.setStyleSheet("font-size: 9px; color: #5a6b7d; text-transform: uppercase; letter-spacing: 1px; border: none;")
            v = QLabel("--")
            v.setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: {'Consolas' if key in ('conf','hit') else 'Microsoft YaHei'}; color: {'#00d4ff' if key == 'current' else '#ffaa00' if key == 'expected' else '#d8e4f0'}; border: none;")
            bl.addWidget(l)
            bl.addWidget(v)
            self.bottom_labels[f"rt_{key}"] = v
            metric_grid.addWidget(box, i // 2, i % 2)

        rt_layout.addLayout(metric_grid)

        # Top3
        lbl_top3_title = QLabel("Top3 候选")
        lbl_top3_title.setStyleSheet("font-size: 10px; color: #5a6b7d; text-transform: uppercase; letter-spacing: 1px; padding-top: 4px; border: none;")
        rt_layout.addWidget(lbl_top3_title)
        self.lbl_top3 = QLabel("等待数据...")
        self.lbl_top3.setWordWrap(True)
        self.lbl_top3.setStyleSheet("font-size: 12px; color: #d8e4f0; border: none;")
        rt_layout.addWidget(self.lbl_top3)

        right_col.addWidget(rt_panel)

        # Production stats
        stats_panel = QFrame()
        stats_panel.setObjectName("panel")
        sp_layout = QVBoxLayout(stats_panel)
        sp_layout.setContentsMargins(14, 14, 14, 14)
        sp_layout.setSpacing(8)

        stats_title = QLabel("生产统计")
        stats_title.setStyleSheet("font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #00d4ff;")
        sp_layout.addWidget(stats_title)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)

        self.stat_ring = StatRing()
        stats_row.addWidget(self.stat_ring)

        stats_info = QVBoxLayout()
        stats_info.setSpacing(4)
        self.lbl_pass = QLabel("合格: 0")
        self.lbl_pass.setStyleSheet("font-size: 14px; font-weight: 600; font-family: 'Consolas'; color: #00ff88; border: none;")
        self.lbl_skip = QLabel("超时: 0")
        self.lbl_skip.setStyleSheet("font-size: 14px; font-weight: 600; font-family: 'Consolas'; color: #ff4466; border: none;")
        self.lbl_total = QLabel("总计: 0")
        self.lbl_total.setStyleSheet("font-size: 14px; font-weight: 600; font-family: 'Consolas'; color: #d8e4f0; border: none;")
        stats_info.addWidget(self.lbl_pass)
        stats_info.addWidget(self.lbl_skip)
        stats_info.addWidget(self.lbl_total)
        stats_row.addLayout(stats_info)
        stats_row.addStretch()
        sp_layout.addLayout(stats_row)

        right_col.addWidget(stats_panel)

        # Event log
        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(14, 14, 14, 14)
        log_layout.setSpacing(8)

        log_title = QLabel("事件日志")
        log_title.setStyleSheet("font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #00d4ff;")
        log_layout.addWidget(log_title)

        self.event_scroll = QScrollArea()
        self.event_scroll.setWidgetResizable(True)
        self.event_container = QWidget()
        self.event_container.setStyleSheet("background: transparent;")
        self.event_container_layout = QVBoxLayout(self.event_container)
        self.event_container_layout.setContentsMargins(0, 0, 0, 0)
        self.event_container_layout.setSpacing(0)
        self.event_container_layout.addStretch()
        self.event_scroll.setWidget(self.event_container)
        log_layout.addWidget(self.event_scroll)

        right_col.addWidget(log_panel, 1)

        main_layout.addWidget(right_frame)

        root.addWidget(main_widget, 1)

        # === Footer ===
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(48)
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(24, 8, 24, 8)
        f_layout.setSpacing(8)

        self.btn_import = QPushButton("导入视频")
        self.btn_start = QPushButton("开始分析")
        self.btn_pause = QPushButton("暂停/继续")
        self.btn_stop = QPushButton("停止")
        self.btn_export = QPushButton("导出结果")
        f_layout.addWidget(self.btn_import)
        f_layout.addWidget(self.btn_start)
        f_layout.addWidget(self.btn_pause)
        f_layout.addWidget(self.btn_stop)
        f_layout.addWidget(self.btn_export)

        f_layout.addStretch()

        self.edit_yolo = QLineEdit("0.30")
        self.edit_lstm = QLineEdit("0.15")

        f_layout.addWidget(QLabel("YOLO阈值"))
        f_layout.addWidget(self.edit_yolo)
        f_layout.addSpacing(10)
        f_layout.addWidget(QLabel("LSTM阈值"))
        f_layout.addWidget(self.edit_lstm)
        f_layout.addSpacing(10)

        self.chk_keypoints = QCheckBox("显示关键点")
        self.chk_snapshots = QCheckBox("保存截图")
        self.chk_log = QCheckBox("保存日志")
        self.chk_boxes = QCheckBox("显示检测框")
        for cb in [self.chk_keypoints, self.chk_snapshots, self.chk_log]:
            cb.setChecked(True)
            f_layout.addWidget(cb)
        self.chk_boxes.setChecked(False)
        f_layout.addWidget(self.chk_boxes)

        root.addWidget(footer)

        # Connections
        self.btn_import.clicked.connect(self.on_import_video)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_export.clicked.connect(self.on_export)

    def _toggle_max(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _on_header_press(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos()

    def _on_header_move(self, event):
        if hasattr(self, '_drag_pos') and self._drag_pos is not None:
            if event.buttons() & Qt.LeftButton:
                if self.isMaximized():
                    return
                delta = event.globalPos() - self._drag_pos
                self._drag_pos = event.globalPos()
                self.move(self.pos() + delta)

    def _update_clock(self):
        now = datetime.now()
        self.lbl_clock.setText(now.strftime("%H:%M:%S"))

    def on_import_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择视频", str(BASE_DIR / "video"), "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if not f:
            return

        path = Path(f)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QMessageBox.warning(self, "导入失败", "该视频无法打开")
            return

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            QMessageBox.warning(self, "导入失败", "该视频无法读取帧")
            return

        h, w = frame.shape[:2]
        cap.release()

        self.video_path = path
        self.video_label.setText("")
        pix = QPixmap.fromImage(bgr_to_qimage(frame))
        self._set_video_preview_pixmap(pix)

    def _reset_cards(self, names: List[str]):
        for i, card in enumerate(self.cards):
            if i < len(names):
                card.lbl_name.setText(names[i].split("：")[-1] if "：" in names[i] else names[i])
                card.set_status("pending")
            else:
                card.set_status("pending")

    def _params(self) -> RuntimeParams:
        return RuntimeParams(
            yolo_conf=float(self.edit_yolo.text().strip()),
            lstm_conf=float(self.edit_lstm.text().strip()),
            confirm_frames=CONFIRM_FRAMES_FIXED,
            show_boxes=self.chk_boxes.isChecked(),
            show_keypoints=self.chk_keypoints.isChecked(),
            save_snapshots=self.chk_snapshots.isChecked(),
            save_log=self.chk_log.isChecked(),
        )

    def on_start(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "正在分析中")
            return
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先导入视频")
            return

        try:
            params = self._params()
        except Exception:
            QMessageBox.warning(self, "参数错误", "请检查阈值参数格式")
            return

        self.pass_count = 0
        self.skip_count = 0
        self.lbl_pass.setText("合格: 0")
        self.lbl_skip.setText("超时: 0")
        self.lbl_total.setText("总计: 0")
        self.stat_ring.set_rate(100)
        for c in self.cards:
            c.set_status("pending")
        self.video_bottom.setVisible(True)
        self.lbl_status_pill.setText("● 推理中")
        self.lbl_status_pill.setStyleSheet("background: rgba(255,170,0,0.1); border: 1px solid rgba(255,170,0,0.3); border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: #ffaa00;")

        self._clear_events()

        self.worker = InferenceWorker(str(self.video_path), params)
        self.worker.sig_frame.connect(self.on_frame)
        self.worker.sig_status.connect(self.on_status)
        self.worker.sig_action.connect(self.on_action)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.sig_error.connect(self.on_error)
        self.worker.start()

    def on_pause(self):
        if self.worker and self.worker.isRunning():
            self.worker.toggle_pause()

    def on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self.lbl_status_pill.setText("● 系统在线")
        self.lbl_status_pill.setStyleSheet("background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: #00ff88;")
        self.video_bottom.setVisible(False)

    def on_export(self):
        if not self.last_run_dir or not self.last_run_dir.exists():
            QMessageBox.information(self, "提示", "暂无可导出结果")
            return
        dst = QFileDialog.getExistingDirectory(self, "选择导出目录", str(BASE_DIR))
        if not dst:
            return
        out = Path(dst) / self.last_run_dir.name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(self.last_run_dir, out)
        QMessageBox.information(self, "导出完成", f"已导出到: {out}")

    def _set_video_preview_pixmap(self, pix: QPixmap):
        if pix.isNull():
            return
        scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
        self.video_label.setText("")

    def _apply_font_scale(self, scale: float = None):
        if scale is None:
            scale = max(1.0, min(2.5, self.width() / 1600.0))
        self._font_scale = scale
        s = lambda v: max(1, int(v * scale))

        for card in self.cards:
            card.set_scale(scale)

        if hasattr(self, 'lbl_cycle'):
            self.lbl_cycle.setStyleSheet(f"font-size: {s(52)}px; font-weight: 700; font-family: 'Consolas'; color: #00d4ff;")
        if hasattr(self, 'lbl_completed'):
            self.lbl_completed.setStyleSheet(f"font-size: {s(30)}px; font-weight: 700; font-family: 'Consolas'; color: #00ff88;")
        if hasattr(self, 'lbl_pass'):
            self.lbl_pass.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: 'Consolas'; color: #00ff88; border: none;")
        if hasattr(self, 'lbl_skip'):
            self.lbl_skip.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: 'Consolas'; color: #ff4466; border: none;")
        if hasattr(self, 'lbl_total'):
            self.lbl_total.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: 'Consolas'; color: #d8e4f0; border: none;")

        if hasattr(self, 'bottom_labels'):
            for key, lbl in self.bottom_labels.items():
                if key in ('current', 'expected'):
                    lbl.setStyleSheet(f"font-size: {s(20)}px; font-weight: 700; color: {'#00d4ff' if key == 'current' else '#ffaa00'}; border: none;")
                elif key in ('frame', 'time', 'hit'):
                    lbl.setStyleSheet(f"font-size: {s(20)}px; font-weight: 700; font-family: 'Consolas'; color: {'#00ff88' if key == 'hit' else '#5a6b7d'}; border: none;")
                elif key == 'rt_current':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; color: #00d4ff; border: none;")
                elif key == 'rt_expected':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; color: #ffaa00; border: none;")
                elif key == 'rt_conf':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; font-family: 'Consolas'; color: #00ff88; border: none;")
                elif key == 'rt_hit':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; font-family: 'Consolas'; color: #d8e4f0; border: none;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_font_scale()
        pix = self.video_label.pixmap()
        if pix is not None and not pix.isNull():
            self._set_video_preview_pixmap(pix)

    def on_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self._set_video_preview_pixmap(pix)

    def _clear_events(self):
        while self.event_container_layout.count() > 1:
            item = self.event_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.event_items.clear()

    def _add_event(self, step_name: str, event_type: str, cycle: int, time_str: str):
        item = EventItem(step_name, event_type, cycle, time_str)
        self.event_container_layout.insertWidget(0, item)
        self.event_items.append(item)

    def on_status(self, st: dict):
        if "action_defs" in st:
            self._reset_cards(st["action_defs"])
            cycle = st.get("cycle", 1)
            self.lbl_cycle.setText(str(cycle))
            self.lbl_completed.setText(str(cycle - 1))
            for c in self.cards:
                c.set_status("pending")
            if self.cards:
                self.cards[0].set_status("进行中")
            return

        self.bottom_labels["frame"].setText(str(st.get("frame_id", 0)))
        self.bottom_labels["time"].setText(f"{st.get('time_sec', 0.0):.2f}s")
        self.bottom_labels["current"].setText(st.get('current_pred', '-'))
        self.bottom_labels["expected"].setText(st.get("expected_show", "-"))
        self.bottom_labels["hit"].setText(f"{st.get('hit', 0)}/{st.get('confirm_frames', 0)}")

        self.bottom_labels["rt_current"].setText(st.get('current_pred', '-'))
        self.bottom_labels["rt_expected"].setText(st.get("expected_show", "-"))
        conf = st.get('expected_conf', 0.0)
        self.bottom_labels["rt_conf"].setText(f"{conf*100:.1f}%")
        self.bottom_labels["rt_conf"].setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: 'Consolas'; color: {'#00ff88' if conf > 0.5 else '#ffaa00'}; border: none;")
        self.bottom_labels["rt_hit"].setText(f"{st.get('hit', 0)}/{st.get('confirm_frames', 0)}")

        top3 = st.get('top3', '-')
        self.lbl_top3.setText(top3 if top3 != '-' else '等待数据...')

        p = st.get("progress", 0)
        cycle = st.get("cycle", 1)
        self.lbl_cycle.setText(str(cycle))
        self.lbl_completed.setText(str(cycle - 1))

        for i, card in enumerate(self.cards):
            if i < p:
                if card.status not in ("done", "timeout"):
                    card.set_status("完成")
            elif i == p:
                card.set_status("进行中")
                card.set_confidence(conf)
            else:
                if card.status not in ("done", "timeout"):
                    card.set_status("pending")

    def on_action(self, ev: dict):
        idx = int(ev.get("index", -1))
        status = ev.get("status", "完成")

        if 0 <= idx < len(self.cards):
            self.cards[idx].set_status(status, ev.get("info", ""))
            self.cards[idx].set_snapshot(ev.get("snapshot"))

        if status == "完成":
            self.pass_count += 1
        elif status == "超时跳过":
            self.skip_count += 1

        total = self.pass_count + self.skip_count
        rate = round(self.pass_count / total * 100) if total > 0 else 100
        self.lbl_pass.setText(f"合格: {self.pass_count}")
        self.lbl_skip.setText(f"超时: {self.skip_count}")
        self.lbl_total.setText(f"总计: {total}")
        self.stat_ring.set_rate(rate)

        if idx < len(self.cards):
            step = self.cards[idx]
            time_str = datetime.now().strftime("%H:%M:%S")
            self._add_event(step.name, "done" if status == "完成" else "skip",
                           int(self.lbl_cycle.text()), time_str)

    def on_finished(self, result: dict):
        run_dir = result.get("run_dir", "")
        self.last_run_dir = Path(run_dir) if run_dir else None
        self.lbl_status_pill.setText("● 系统在线")
        self.lbl_status_pill.setStyleSheet("background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: #00ff88;")
        QMessageBox.information(self, "完成", "视频分析完成")

    def on_error(self, msg: str):
        QMessageBox.critical(self, "错误", msg)
        self.lbl_status_pill.setText("● 系统在线")
        self.lbl_status_pill.setStyleSheet("background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: #00ff88;")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
