"""
AI-SOP 后端服务：接收视频，逐帧推理，通过 WebSocket 推送结果到前端。

用法:
    python server.py
    然后访问 http://localhost:5173
"""
import base64
import csv
import json
import os
import shutil
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    raise ImportError("请先安装 mediapipe") from e

from ai_sop_gui import (
    ActionLSTM,
    BASE_DIR,
    YOLO_MODEL_PATH,
    LSTM_MODEL_PATH,
    LSTM_CONFIG_PATH,
    RUNS_GUI_DIR,
    CLASS_NAMES,
    SOURCE_WINDOWS,
    STEP_MIN_STAGE_SEC,
    CONFIRM_FRAMES_FIXED,
    STEP_TIMEOUT_SEC,
    DEFAULT_ACTION_DEFS,
    ACTION_CN_MAP,
    build_feature_row,
    draw_yolo_boxes_cn,
    bgr_to_qimage,
    action_to_cn,
    fine_label_from_row,
    to_beijing_time_str,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "aisop-secret"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")

config = json.loads(LSTM_CONFIG_PATH.read_text(encoding="utf-8"))
id2label = {int(x["id"]): x["label"] for x in config["label_map"]}
label2id = {v: k for k, v in id2label.items()}

lstm = ActionLSTM(
    config["input_size"],
    config["hidden_size"],
    config["num_layers"],
    config["num_classes"],
).to(device)
lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
lstm.eval()

yolo = YOLO(str(YOLO_MODEL_PATH))
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

actions = [
    {"index": i, "show": show, "fine_label": fine, "done": False, "hit_count": 0, "expected_conf": 0.0, "done_time_sec": None, "snapshot_path": None}
    for i, (show, fine) in enumerate(DEFAULT_ACTION_DEFS)
]

inference_state = {
    "expected_idx": 0,
    "current_hit": 0,
    "cycle": 1,
    "expected_stage_start_sec": 0.0,
    "last_expected_idx": -1,
    "events": [],
    "running": False,
    "stop": False,
    "yolo_conf": 0.30,
    "lstm_conf": 0.15,
    "show_boxes": False,
    "show_keypoints": True,
}


def reset_state():
    global actions
    actions = [
        {"index": i, "show": show, "fine_label": fine, "done": False, "hit_count": 0, "expected_conf": 0.0, "done_time_sec": None, "snapshot_path": None}
        for i, (show, fine) in enumerate(DEFAULT_ACTION_DEFS)
    ]
    inference_state["expected_idx"] = 0
    inference_state["current_hit"] = 0
    inference_state["cycle"] = 1
    inference_state["expected_stage_start_sec"] = 0.0
    inference_state["last_expected_idx"] = -1
    inference_state["events"] = []
    inference_state["stop"] = False


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "没有视频文件"}), 400
    f = request.files["video"]
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    path = upload_dir / f.filename
    f.save(str(path))
    reset_state()
    return jsonify({"path": str(path), "name": f.filename})


@app.route("/api/params", methods=["POST"])
def set_params():
    data = request.json
    inference_state["yolo_conf"] = float(data.get("yolo_conf", 0.30))
    inference_state["lstm_conf"] = float(data.get("lstm_conf", 0.15))
    inference_state["show_boxes"] = bool(data.get("show_boxes", False))
    inference_state["show_keypoints"] = bool(data.get("show_keypoints", True))
    return jsonify({"ok": True})


@socketio.on("start_inference")
def start_inference(data):
    video_path = data.get("path")
    if not video_path or not os.path.exists(video_path):
        socketio.emit("error", {"msg": "视频路径无效"})
        return

    reset_state()
    inference_state["running"] = True
    inference_state["stop"] = False
    socketio.start_background_task(run_inference, video_path)


@socketio.on("stop_inference")
def stop_inference():
    inference_state["stop"] = True


def run_inference(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        socketio.emit("error", {"msg": "无法打开视频"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    feat_queue = deque(maxlen=max(SOURCE_WINDOWS))
    pred_history = deque(maxlen=5)
    current_pred = "background"

    action_defs = [a["show"] for a in actions]
    socketio.emit("init", {
        "action_defs": action_defs,
        "total": len(actions),
        "cycle": 1,
    })

    while not inference_state["stop"]:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_id / fps
        h, w = frame.shape[:2]

        res = yolo.predict(frame, verbose=False, conf=inference_state["yolo_conf"], device=device)[0]

        vis = frame.copy()
        if inference_state["show_boxes"]:
            vis = draw_yolo_boxes_cn(frame, res.boxes, yolo.names)

        detections = []
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append({
                    "cls_name": yolo.names.get(cls_id, str(cls_id)),
                    "conf": float(box.conf[0]),
                    "xyxy": [x1, y1, x2, y2],
                })

        rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hr = hands.process(rgb_raw)
        hands_out = []
        if hr.multi_hand_landmarks:
            for hidx, hand_lm in enumerate(hr.multi_hand_landmarks):
                if inference_state["show_keypoints"]:
                    mp_draw.draw_landmarks(
                        vis, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 140), thickness=4, circle_radius=5),
                        mp_draw.DrawingSpec(color=(0, 220, 120), thickness=4, circle_radius=5),
                    )
                lms = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_lm.landmark]
                hands_out.append({"hand_index": hidx, "landmarks": lms})

        feat_queue.append(build_feature_row(detections, hands_out, w, h))

        expected_show = "DONE"
        expected_conf = 0.0
        top3_text = "-"

        expected_idx = inference_state["expected_idx"]
        expected_action = actions[expected_idx] if expected_idx < len(actions) else None
        if expected_action:
            expected_show = expected_action["show"]

        if expected_action and expected_idx != inference_state["last_expected_idx"]:
            inference_state["expected_stage_start_sec"] = t_sec
            inference_state["last_expected_idx"] = expected_idx

        if len(feat_queue) >= min(SOURCE_WINDOWS):
            feat_arr = np.stack(feat_queue, axis=0)
            scale_preds = []
            expected_confs = []
            top3_candidates = []

            for src_len in SOURCE_WINDOWS:
                if len(feat_arr) < src_len:
                    continue
                src_seq = feat_arr[-src_len:]
                pos = np.linspace(0, src_len - 1, 48)
                idx = np.clip(np.round(pos).astype(int), 0, src_len - 1)
                seq = src_seq[idx]

                xb = torch.from_numpy(seq[None, ...]).float().to(device)
                with torch.no_grad():
                    logits = lstm(xb)
                    probs = torch.softmax(logits, dim=1)

                probs_np = probs[0].detach().cpu().numpy()
                for i_cls, p_cls in enumerate(probs_np):
                    lb = id2label.get(i_cls, "")
                    if lb and lb != "background":
                        top3_candidates.append((lb, float(p_cls)))

                pid = int(torch.argmax(probs, dim=1).item())
                scale_preds.append(id2label.get(pid, "background"))

                if expected_action:
                    idx_exp = label2id.get(expected_action["fine_label"])
                    if idx_exp is not None:
                        expected_confs.append(float(probs[0, idx_exp].item()))

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

                if expected_action:
                    expected_conf = float(np.mean(expected_confs)) if expected_confs else 0.0
                    stage_elapsed = t_sec - inference_state["expected_stage_start_sec"]
                    if stage_elapsed >= STEP_MIN_STAGE_SEC:
                        pass
                    else:
                        expected_conf = 0.0

                    if expected_conf >= inference_state["lstm_conf"]:
                        inference_state["current_hit"] += 1
                    else:
                        inference_state["current_hit"] = 0

                    expected_action["expected_conf"] = expected_conf
                    expected_action["hit_count"] = inference_state["current_hit"]

                    confirm_frames = CONFIRM_FRAMES_FIXED

                    if inference_state["current_hit"] >= confirm_frames:
                        expected_action["done"] = True
                        expected_action["done_time_sec"] = t_sec

                        done_time_beijing = to_beijing_time_str()
                        event = {
                            "cycle": inference_state["cycle"],
                            "index": expected_action["index"] + 1,
                            "action": expected_action["show"],
                            "fine_label": expected_action["fine_label"],
                            "done_time_sec": round(t_sec, 3),
                            "done_time_beijing": done_time_beijing,
                            "status": "完成",
                        }
                        inference_state["events"].append(event)

                        socketio.emit("action", {
                            "index": expected_action["index"],
                            "status": "完成",
                            "info": f"完成时间: {done_time_beijing}",
                            "event": event,
                        })

                        inference_state["expected_idx"] += 1
                        inference_state["current_hit"] = 0

                        if inference_state["expected_idx"] >= len(actions):
                            inference_state["expected_idx"] = 0
                            inference_state["cycle"] += 1
                            inference_state["last_expected_idx"] = -1
                            for a in actions:
                                a["done"] = False
                                a["hit_count"] = 0
                                a["expected_conf"] = 0.0
                                a["done_time_sec"] = None
                                a["snapshot_path"] = None
                            socketio.emit("new_cycle", {
                                "action_defs": [a["show"] for a in actions],
                                "total": len(actions),
                                "cycle": inference_state["cycle"],
                            })

                    elif t_sec - inference_state["expected_stage_start_sec"] > STEP_TIMEOUT_SEC:
                        done_time_beijing = to_beijing_time_str()
                        event = {
                            "cycle": inference_state["cycle"],
                            "index": expected_action["index"] + 1,
                            "action": expected_action["show"],
                            "fine_label": expected_action["fine_label"],
                            "done_time_sec": round(t_sec, 3),
                            "done_time_beijing": done_time_beijing,
                            "status": "超时跳过",
                        }
                        inference_state["events"].append(event)

                        socketio.emit("action", {
                            "index": expected_action["index"],
                            "status": "超时跳过",
                            "info": f"超时跳过 ({STEP_TIMEOUT_SEC}s)",
                            "event": event,
                        })

                        inference_state["expected_idx"] += 1
                        inference_state["current_hit"] = 0

                        if inference_state["expected_idx"] >= len(actions):
                            inference_state["expected_idx"] = 0
                            inference_state["cycle"] += 1
                            inference_state["last_expected_idx"] = -1
                            for a in actions:
                                a["done"] = False
                                a["hit_count"] = 0
                                a["expected_conf"] = 0.0
                                a["done_time_sec"] = None
                                a["snapshot_path"] = None
                            socketio.emit("new_cycle", {
                                "action_defs": [a["show"] for a in actions],
                                "total": len(actions),
                                "cycle": inference_state["cycle"],
                            })

        status = {
            "frame_id": frame_id,
            "time_sec": t_sec,
            "current_pred": action_to_cn(current_pred),
            "expected_show": expected_show,
            "expected_conf": expected_conf,
            "lstm_conf": inference_state["lstm_conf"],
            "hit": inference_state["current_hit"],
            "confirm_frames": CONFIRM_FRAMES_FIXED,
            "top3": top3_text,
            "progress": inference_state["expected_idx"],
            "total": len(actions),
            "cycle": inference_state["cycle"],
            "events_count": len(inference_state["events"]),
        }

        if frame_id % 2 == 0:
            small = cv2.resize(vis, (min(960, vis.shape[1]), int(vis.shape[0] * min(960, vis.shape[1]) / vis.shape[1])))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
            socketio.emit("frame_data", buf.tobytes())
            socketio.emit("frame_status", status)

        frame_id += 1

    cap.release()
    inference_state["running"] = False
    socketio.emit("finished", {
        "events": inference_state["events"],
        "total_cycles": inference_state["cycle"],
    })


if __name__ == "__main__":
    print("服务启动: http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
