"""推理 Worker —— QThread 子类，跑 YOLO + MediaPipe + LSTM + 顺序状态机。"""
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from ultralytics import YOLO

from ai_sop.core.constants import (
    ACTION_CN_MAP,
    DEFAULT_ACTION_DEFS,
    HAND_CIRCLE_RADIUS,
    HAND_CONNECTION_THICKNESS,
    HAND_LANDMARK_THICKNESS,
    LSTM_CONFIG_PATH,
    LSTM_MODEL_PATH,
    RUNS_GUI_DIR,
    SOURCE_WINDOWS,
    STEP_MIN_STAGE_SEC,
    STEP_TIMEOUT_SEC,
    TIMELINE_CSV,
    YOLO_MODEL_PATH,
)
from ai_sop.core.features import build_feature_row, draw_yolo_boxes_cn
from ai_sop.core.models import ActionLSTM, ActionRuntime, RuntimeParams
from ai_sop.core.utils import action_to_cn, bgr_to_qimage, fine_label_from_row, to_beijing_time_str

try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    raise ImportError("请先安装 mediapipe，例如: pip install mediapipe==0.10.14") from e


class InferenceWorker(QThread):
    """推理 Worker 线程 —— QThread 子类。

    主循环 run() 把 8 个阶段（暂停/读帧/YOLO/MediaPipe/拼特征/LSTM 投票/状态机/广播）
    拆成 5 个子方法，每阶段单一职责，便于调试。
    """
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

        # 推理设备：CUDA > MPS(Apple Silicon) > CPU
        self.device = (
            "cuda:0" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.id2label: Dict[int, str] = {}                # LSTM 类别 id → 标签名
        self.label2id: Dict[str, int] = {}                # 标签名 → 类别 id（反查期望动作置信度用）

        self.lstm = None
        self.yolo = None
        self.hands = None

        # === 顺序状态机变量 ===
        self.actions: List[ActionRuntime] = []            # 4 步动作列表
        self.expected_idx = 0                              # 当前期望步骤索引（推进指针）
        self.current_hit = 0                               # 当前步骤的连续命中帧数
        self.expected_stage_start_sec = 0.0                # 当前期望步骤开始时间（用于超时判断 + 最小持续时间）
        self.last_expected_idx = -1                        # 上一次的 expected_idx，用于检测「步骤刚切换」
        self.cycle = 1                                     # 多周期循环计数器（D1→D4 走完 +1）

        self.run_dir: Optional[Path] = None
        self.events: List[dict] = []                       # 事件日志（每步骤完成/超时记一条）

    def stop(self):
        """请求停止推理（run() 循环下一轮检测到 _stop=True 即退出）。"""
        self._stop = True

    def toggle_pause(self):
        """切换暂停状态。暂停时 run() 不读帧、不推进状态机，但保留所有累计状态。"""
        self._pause = not self._pause

    def _load_models(self):
        """加载 LSTM / YOLO / MediaPipe 三个模型，并构建标签映射。"""
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
        # MediaPipe Hands：model_complexity=1 比 0 慢约 2 倍但更准；tracking 模式提升连续性
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _build_actions(self):
        """构建 4 步动作列表：优先用 timeline.csv 的标注顺序，否则用 DEFAULT_ACTION_DEFS。"""
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
        """创建本次运行专属的输出目录（截图/日志都放这），用视频名 + 时间戳避免覆盖。"""
        RUNS_GUI_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_GUI_DIR / f"{self.video_path.stem}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "snapshots").mkdir(exist_ok=True)

    def _save_snapshot(self, frame_bgr: np.ndarray, action: ActionRuntime, t_sec: float) -> str:
        """步骤完成时把当前帧存为 jpg 截图，返回路径用于 ActionRuntime.snapshot_path。"""
        assert self.run_dir is not None
        fname = f"{action.index + 1:02d}_{action.fine_label}_{t_sec:.2f}.jpg".replace("/", "_")
        out = self.run_dir / "snapshots" / fname
        cv2.imwrite(str(out), frame_bgr)
        return str(out)

    def _calc_expected_conf(self, probs: torch.Tensor, expected_label: str) -> float:
        """取「期望步骤」对应的概率值（不取 argmax）。

        关键设计：顺序约束不问「LSTM 觉得在做什么」，
        而是问「LSTM 觉得在做期望步骤的概率有多大」——
        这样别的动作预测再高也不会误推进。
        """
        idx = self.label2id.get(expected_label)
        if idx is None:
            return 0.0
        return float(probs[0, idx].item())

    def _apply_expected_rules(self, _expected_action: ActionRuntime, expected_conf: float, t_sec: float) -> float:
        """对期望置信度做规则修正：步骤切换后 0.8 秒内强制清零。

        目的：步骤刚切换瞬间，LSTM 容易把上一动作的尾巴误判成新动作，
        这段静默期直接不信任何预测，避免假阳性触发完成。
        """
        stage_elapsed = t_sec - self.expected_stage_start_sec
        if stage_elapsed < STEP_MIN_STAGE_SEC:
            return 0.0

        return expected_conf

    def _reset_cycle(self):
        """多周期循环重置：4 步走完后清空所有步骤状态，cycle+1，广播 UI 重置。"""
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

    def run(self):
        """推理主循环 —— QThread 入口。

        每帧管线：读帧 → ②YOLO + ③MediaPipe → ④拼 146 维特征 →
        ⑥多尺度窗口 LSTM 投票 → ⑦顺序状态机（命中确认/超时跳过/多周期循环）→ ⑧广播。
        各阶段实现见对应子方法。
        """
        try:
            self._load_models()
            self._build_actions()
            self._init_run_dir()

            self.sig_status.emit({"action_defs": [a.show for a in self.actions], "total": len(self.actions)})

            # === 视频源 + 帧级缓冲队列 ===
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_id = 0
            feat_queue = deque(maxlen=max(SOURCE_WINDOWS))  # 96 帧特征缓冲（够最长窗口）
            pred_history = deque(maxlen=5)                  # 最近 5 帧预测（第二层时间平滑）
            current_pred = "background"

            while not self._stop:
                # === ① 暂停时不读帧，避免 frame_id 推进与画面错位 ===
                if self._pause:
                    self.msleep(30)
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                # t_sec 用帧号/fps 算（不是墙钟），保证回放时间与录制时间一致
                t_sec = frame_id / fps
                frame_h, frame_w = frame.shape[:2]

                # === ② YOLO + ③ MediaPipe + ④ 拼 146 维特征并入队 ===
                vis, detections, hands_out = self._detect_frame(frame)
                feat_queue.append(build_feature_row(detections, hands_out, frame_w, frame_h))

                # === ⑤ 取当前期望步骤 + 步骤切换时间戳更新 ===
                expected_show = "DONE"
                expected_action = None
                if self.expected_idx < len(self.actions):
                    expected_action = self.actions[self.expected_idx]
                    expected_show = expected_action.show
                    if self.expected_idx != self.last_expected_idx:
                        self.expected_stage_start_sec = t_sec
                        self.last_expected_idx = self.expected_idx

                # === ⑥ 多尺度窗口 LSTM 投票 ===
                current_pred, expected_conf, top3_text = self._run_multiscale_lstm(
                    feat_queue, pred_history, current_pred, expected_action
                )

                # === ⑦ 顺序状态机：命中确认 / 超时跳过 / 多周期循环 ===
                if expected_action is not None:
                    expected_conf = self._apply_expected_rules(expected_action, expected_conf, t_sec)

                    # 命中帧数机制：达标 +1，未达标立刻清零（不允许中断）
                    if expected_conf >= self.params.lstm_conf:
                        self.current_hit += 1
                    else:
                        self.current_hit = 0

                    expected_action.expected_conf = expected_conf
                    expected_action.hit_count = self.current_hit

                    # 分支 A：连续命中达标 → 步骤完成
                    if self.current_hit >= self.params.confirm_frames:
                        self._complete_action(expected_action, t_sec, vis)
                    # 分支 B：超时跳过（同一帧不可能既完成又超时）
                    elif t_sec - self.expected_stage_start_sec > STEP_TIMEOUT_SEC:
                        self._timeout_action(expected_action, t_sec)

                # === ⑧ 广播本帧状态 + 渲染画面到 UI ===
                self._broadcast_frame_status(
                    frame_id, t_sec, current_pred, expected_show, expected_conf, top3_text, vis
                )

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

    def _detect_frame(self, frame):
        """② YOLO 检测 + ③ MediaPipe 手部关键点，返回 (vis, detections, hands_out)。"""
        # === ② YOLO 检测物体（4 类：base/frame/mirror/screw）===
        res = self.yolo.predict(frame, verbose=False, conf=self.params.yolo_conf, device=self.device)[0]
        if self.params.show_boxes:
            vis = draw_yolo_boxes_cn(frame, res.boxes, self.yolo.names)
        else:
            vis = frame.copy()  # 注：每帧拷贝一次，渲染密集时是非主要瓶颈

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

        # === ③ MediaPipe 手部 21 关键点（最多 2 只手）===
        rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe 要 RGB 输入
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
                # 关键点坐标已归一化到 [0,1]，无需再除以宽高
                lms = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_lm.landmark]
                hands_out.append({"hand_index": hidx, "landmarks": lms})

        return vis, detections, hands_out

    def _run_multiscale_lstm(self, feat_queue, pred_history, current_pred, expected_action):
        """⑥ 多尺度窗口 LSTM 投票 → 返回 (current_pred, expected_conf, top3_text)。

        - feat_queue 至少 48 帧才预测，否则返回 (current_pred 不变, 0.0, "-")
        - 三窗口 [48, 72, 96] 各重采样到 48 长度，跑 LSTM
        - 第 1 层投票：3 个尺度预测多数票
        - 第 2 层投票：最近 5 帧多数票（时间平滑）
        """
        if len(feat_queue) < min(SOURCE_WINDOWS):
            return current_pred, 0.0, "-"

        feat_arr = np.stack(feat_queue, axis=0)
        scale_preds = []          # 3 个窗口各自的 argmax 预测（用于第 1 层投票）
        expected_confs = []       # 3 个窗口各自的「期望动作概率」（不取 argmax）
        top3_candidates = []      # 所有窗口的非 background 概率（用于 top3 显示）

        # 三窗口 [48, 72, 96]：短窗口响应快、长窗口更稳，三者投票融合
        for src_len in SOURCE_WINDOWS:
            if len(feat_arr) < src_len:
                continue

            src_seq = feat_arr[-src_len:]
            # 关键：不同长度窗口都重采样到 48 → LSTM 输入维度统一为 (1, 48, 146)
            # linspace 等距取点：96 帧窗口相当于每 2 帧取 1 帧（降采样）
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

            # 取「期望步骤」对应的概率（非 argmax）—— 顺序约束的核心
            if expected_action is not None:
                expected_confs.append(self._calc_expected_conf(probs, expected_action.fine_label))

        if not scale_preds:
            return current_pred, 0.0, "-"

        # top3 显示：所有窗口中各类最大概率，取前 3
        top3_text = "-"
        if top3_candidates:
            agg = {}
            for lb, p in top3_candidates:
                agg[lb] = max(agg.get(lb, 0.0), p)
            top3_sorted = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_text = " | ".join([f"{action_to_cn(lb)}:{p:.2f}" for lb, p in top3_sorted])

        # === 第 1 层投票：3 个尺度窗口多数票 ===
        votes = {}
        for p in scale_preds:
            votes[p] = votes.get(p, 0) + 1
        current_pred = max(votes, key=votes.get)
        # === 第 2 层投票：最近 5 帧多数票（时间平滑）===
        pred_history.append(current_pred)
        smooth_votes = {}
        for p in pred_history:
            smooth_votes[p] = smooth_votes.get(p, 0) + 1
        current_pred = max(smooth_votes, key=smooth_votes.get)

        expected_conf = float(np.mean(expected_confs)) if expected_confs else 0.0
        return current_pred, expected_conf, top3_text

    def _complete_action(self, expected_action, t_sec, vis):
        """分支 A：步骤完成 → 截图 + 事件 + 信号 + 推进 + 多周期重置。"""
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

        # 多周期循环：4 步全部完成 → 重置状态 + cycle+1
        if self.expected_idx >= len(self.actions):
            self._reset_cycle()

    def _timeout_action(self, expected_action, t_sec):
        """分支 B：超时跳过 → 事件 + 信号 + 推进 + 多周期重置。"""
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

        # 多周期循环：超时分支与完成分支一致
        if self.expected_idx >= len(self.actions):
            self._reset_cycle()

    def _broadcast_frame_status(self, frame_id, t_sec, current_pred, expected_show, expected_conf, top3_text, vis):
        """⑧ 广播本帧状态 + 渲染画面到 UI。"""
        status = {
            "frame_id": frame_id,
            "time_sec": t_sec,
            "current_pred": action_to_cn(current_pred),   # LSTM 当前预测（仅显示，不驱动状态机）
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
        self.sig_status.emit(status)                      # 状态栏更新
        self.sig_frame.emit(bgr_to_qimage(vis))            # 视频画面更新

    def _save_logs(self):
        """导出 result.json（含全部事件）+ events.csv（Excel 友好，UTF-8 BOM）。"""
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
