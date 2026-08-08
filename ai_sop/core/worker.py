"""推理 Worker —— QThread 子类，跑 MediaPipe + LSTM + 顺序状态机。"""
import csv
import json
import time
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

from ai_sop.core.constants import (
    ACTION_CN_MAP,
    DEFAULT_ACTION_DEFS,
    HAND_CIRCLE_RADIUS,
    HAND_CONNECTION_THICKNESS,
    HAND_LANDMARK_THICKNESS,
    LSTM_CONFIG_PATH,
    LSTM_MODEL_PATH,
    MAX_RECONNECT_ATTEMPTS,
    MES_ENABLED,
    MES_TOKEN,
    MES_URL,
    PENDING_FEATURES_DIR,
    PENDING_LABELS_CSV,
    RECONNECT_DELAY_SEC,
    RUNS_GUI_DIR,
    SITE_INFO,
    SLOW_RATIO_THRESHOLD,
    SOURCE_WINDOWS,
    STEP_MIN_STAGE_SEC,
    STEP_TIMEOUT_SEC,
    TIMELINE_CSV,
)
from ai_sop.core.features import build_feature_row
from ai_sop.core.models import ActionLSTM, ActionRuntime, RuntimeParams
from ai_sop.core.utils import action_to_cn, bgr_to_qimage, fine_label_from_row, http_post_json, to_beijing_time_str

try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    raise ImportError("请先安装 mediapipe，例如: pip install mediapipe==0.10.14") from e


class InferenceWorker(QThread):
    """推理 Worker 线程 —— QThread 子类。

    主循环 run() 把 7 个阶段（暂停/读帧/MediaPipe/拼特征/LSTM 投票/状态机/广播）
    拆成 5 个子方法，每阶段单一职责，便于调试。
    """
    sig_frame = pyqtSignal(QImage)
    sig_status = pyqtSignal(dict)
    sig_action = pyqtSignal(dict)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, video_source: str, params: RuntimeParams, parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self.is_live = (
            video_source.isdigit()
            or video_source.lower().startswith(("rtsp://", "http://", "https://"))
        )
        self.video_path = Path(video_source) if not self.is_live else None
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
        self.hands = None

        # === 顺序状态机变量 ===
        self.actions: List[ActionRuntime] = []            # 4 步动作列表
        self.expected_idx = 0                              # 当前期望步骤索引（推进指针）
        self.current_hit = 0                               # 当前步骤的连续命中帧数
        self.expected_stage_start_sec = 0.0                # 当前期望步骤开始时间（用于超时判断 + 最小持续时间）
        self.last_expected_idx = -1                        # 上一次的 expected_idx，用于检测「步骤刚切换」
        self.cycle = 1                                     # 多周期循环计数器（D1→D4 走完 +1）
        self.cycle_start_sec: Optional[float] = None       # 当前周期开始时间（D1 进入时记录，用于算 CT）
        self.cycle_start_beijing: Optional[str] = None     # 当前周期开始时间（北京时间）
        self.cycle_times_sec: List[float] = []             # 已完成的周期 CT（Cycle Time）列表
        self.action_durations: Dict[str, List[float]] = {} # 各动作的历史耗时（跨周期累计，用于偏慢判定基准）

        self.run_dir: Optional[Path] = None
        self.events: List[dict] = []                       # 事件日志（每步骤完成/超时记一条）

    def stop(self):
        """请求停止推理（run() 循环下一轮检测到 _stop=True 即退出）。"""
        self._stop = True

    def toggle_pause(self):
        """切换暂停状态。暂停时 run() 不读帧、不推进状态机，但保留所有累计状态。"""
        self._pause = not self._pause

    def _load_models(self):
        """加载 LSTM / MediaPipe 两个模型，并构建标签映射。"""
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

        if self.video_path is not None and TIMELINE_CSV.exists():
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
        if self.video_path is not None:
            name = self.video_path.stem
        elif self.video_source.isdigit():
            name = f"camera{self.video_source}"
        else:
            name = "rtsp"
        self.run_dir = RUNS_GUI_DIR / f"{name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "snapshots").mkdir(exist_ok=True)

    def _save_snapshot(self, frame_bgr: np.ndarray, action: ActionRuntime, t_sec: float) -> str:
        """步骤完成时把当前帧存为 jpg 截图，返回路径用于 ActionRuntime.snapshot_path。"""
        assert self.run_dir is not None
        fname = f"{action.index + 1:02d}_{action.fine_label}_{t_sec:.2f}.jpg".replace("/", "_")
        out = self.run_dir / "snapshots" / fname
        cv2.imwrite(str(out), frame_bgr)
        return str(out)

    def _reconnect(self):
        """实时源断线重连：间隔重开 VideoCapture，最多 MAX_RECONNECT_ATTEMPTS 次。

        每次尝试前广播「重连中」状态给 UI；成功广播「已重连」并返回新句柄，全部失败返回 None。
        """
        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            self.sig_status.emit({
                "source_status": "reconnecting",
                "attempt": attempt,
                "max_attempts": MAX_RECONNECT_ATTEMPTS,
            })
            self.msleep(int(RECONNECT_DELAY_SEC * 1000))
            src = int(self.video_source) if self.video_source.isdigit() else self.video_source
            new_cap = cv2.VideoCapture(src)
            if new_cap.isOpened():
                self.sig_status.emit({"source_status": "reconnected"})
                return new_cap
        return None

    def save_pending_sample(self, action_label: str, t_sec: float):
        """把当前特征缓冲（最近 48 帧）保存为待回灌训练样本。

        - 特征：train_data/pending_samples/features/<时间戳>_<标签>.npy
        - 标注：追加到 pending_labels.csv，供 ingest_pending.py 回灌训练
        """
        q = getattr(self, "feat_queue", None)
        if q is None or len(q) < min(SOURCE_WINDOWS):
            return
        PENDING_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        feat_arr = np.stack(list(q)[-min(SOURCE_WINDOWS):], axis=0)
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{action_label.replace('/', '_')}.npy"
        np.save(PENDING_FEATURES_DIR / fname, feat_arr)
        with PENDING_LABELS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if PENDING_LABELS_CSV.stat().st_size == 0:
                writer.writerow(["source", "action_label", "time_sec", "saved_at", "file"])
            writer.writerow([
                self.video_source, action_label, round(t_sec, 3),
                datetime.now().isoformat(timespec="seconds"), fname,
            ])

    def mark_current_sample(self):
        """GUI「标记样本」按钮：把当前期望动作的特征样本存为待回灌训练样本。"""
        if self.expected_idx < len(self.actions):
            label = self.actions[self.expected_idx].fine_label
            self.save_pending_sample(label, getattr(self, "_last_t_sec", 0.0))
            return label
        return None

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
        self.cycle_start_sec = None                      # 周期开始时间留空，下一轮 D1 入场时重新记录
        self.cycle_start_beijing = None
        for a in self.actions:
            a.done = False
            a.hit_count = 0
            a.expected_conf = 0.0
            a.start_time_sec = None
            a.duration_sec = None
            a.start_time_beijing = None
            a.done_time_sec = None
            a.snapshot_path = None
        self.sig_status.emit({
            "action_defs": [a.show for a in self.actions],
            "total": len(self.actions),
            "cycle": self.cycle,
        })

    def run(self):
        """推理主循环 —— QThread 入口。

        每帧管线：读帧 → ②MediaPipe → ③拼 126 维特征 →
        ④多尺度窗口 LSTM 投票 → ⑤顺序状态机（命中确认/超时跳过/多周期循环）→ ⑥广播。
        各阶段实现见对应子方法。
        """
        try:
            self._load_models()    # ① 加载 LSTM / MediaPipe 两个模型并建标签映射
            self._build_actions()  # ② 确定本次 SOP 步骤序列（优先 timeline.csv 标注，否则用默认 4 步）
            self._init_run_dir()   # ③ 创建本次运行专属输出目录（截图/日志存放处）

            # ④ 首条状态广播：把动作列表显示名 + 总数发给 UI，用于初始化步骤卡片
            self.sig_status.emit({"action_defs": [a.show for a in self.actions], "total": len(self.actions)})

            # === 视频源 + 帧级缓冲队列 ===
            if self.is_live:
                cap = cv2.VideoCapture(int(self.video_source) if self.video_source.isdigit() else self.video_source)
                live_t0 = time.monotonic()      # 实时源用墙钟计时（摄像头/RTSP 无固定帧率语义）
            else:
                cap = cv2.VideoCapture(str(self.video_path))
                live_t0 = None
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {self.video_source}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # 视频帧率；读取失败或为 0 时兜底用 30（t_sec = frame_id/fps 算时间戳用）
            frame_id = 0                              # 帧计数器：每读一帧 +1，既算时间戳也用于 UI 显示
            self.feat_queue = deque(maxlen=max(SOURCE_WINDOWS))  # 特征缓冲队列，最多存 96 帧（够最长窗口 96）
            feat_queue = self.feat_queue
            pred_history = deque(maxlen=5)                  # 最近 5 帧预测历史，用于第二层时间平滑（多数票去抖）
            current_pred = "background"                     # 当前预测动作初始值，特征攒够 48 帧前一直显示"背景"

            while not self._stop:
                # === ① 暂停时不读帧，避免 frame_id 推进与画面错位 ===
                if self._pause:
                    self.msleep(30)
                    continue

                ok, frame = cap.read()  # 取一帧
                if not ok:
                    if not self.is_live:
                        break                            # 文件模式：读完即结束
                    # 实时源断线：尝试重连，成功则清空缓冲继续，失败则结束
                    self.sig_status.emit({"source_status": "lost"})
                    new_cap = self._reconnect()
                    if new_cap is None:
                        self.sig_status.emit({"source_status": "failed"})
                        break
                    cap.release()
                    cap = new_cap
                    feat_queue.clear()                   # 清空旧特征，避免断线前后的特征混用
                    pred_history.clear()
                    self.expected_stage_start_sec = t_sec  # 重连后重置步骤计时，避免断线导致误超时
                    continue

                # t_sec：文件用帧号/fps（回放时间与录制一致）；实时源用墙钟流逝（从打开源开始算）
                t_sec = (time.monotonic() - live_t0) if live_t0 is not None else frame_id / fps
                self._last_t_sec = t_sec        # 供「标记样本」按钮记录当前时刻
                # === ② MediaPipe + ③ 拼 126 维特征并入队 ===
                vis, hands_out = self._detect_frame(frame)   # MediaPipe 手部关键点（vis=带渲染的画面）
                feat_queue.append(build_feature_row(hands_out))  # 拼成 126 维特征行并入缓冲队列

                # === ④ 取当前期望步骤 + 步骤切换时间戳更新 ===
                expected_show = "DONE"                  # 默认显示"DONE"：全部步骤走完后显示
                expected_action = None                   # 默认无期望步骤
                if self.expected_idx < len(self.actions):    # 还有未完成的步骤
                    expected_action = self.actions[self.expected_idx]  # 取当前指针指向的步骤作为期望动作
                    expected_show = expected_action.show            # 期望步骤的显示名
                    if self.expected_idx != self.last_expected_idx: # 刚切换到新步骤（首次进入/上一步完成）
                        self.expected_stage_start_sec = t_sec       # 记录本步骤开始时间（超时判断 + 最小持续用）
                        self.last_expected_idx = self.expected_idx  # 更新标记，避免每帧重复记录

                # === ⑤ 多尺度窗口 LSTM 投票：用特征缓冲跑 3 窗口 LSTM + 双轮多数票
                # 输入：feat_queue特征缓冲 / pred_history近5帧预测历史 / current_pred当前预测动作 / expected_action当前期望步骤
                current_pred, expected_conf, top3_text = self._run_multiscale_lstm(
                    feat_queue, pred_history, current_pred, expected_action
                )
                # 输出：current_pred为当前预测动作名（仅显示）/ expected_conf为期望动作概率（状态机用）/ top3_text为前3候选文本（UI 显示用）

                # === ⑥ 顺序状态机：命中确认 / 超时跳过 / 多周期循环 ===
                if expected_action is not None:
                    expected_conf = self._apply_expected_rules(expected_action, expected_conf, t_sec) #当前画面序列正在做期望步骤"的概率

                    # 命中帧数机制：达标 +1，未达标立刻清零（不允许中断）
                    if expected_conf >= self.params.lstm_conf:
                        # 首次命中：记录动作开始时间（步骤耗时/CT 从"检测到动作"开始计时）
                        if expected_action.start_time_sec is None:
                            expected_action.start_time_sec = t_sec
                            expected_action.start_time_beijing = to_beijing_time_str()
                            if self.expected_idx == 0 and self.cycle_start_sec is None:
                                self.cycle_start_sec = t_sec
                                self.cycle_start_beijing = expected_action.start_time_beijing
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

                # === ⑦ 广播本帧状态 + 渲染画面到 UI ===
                stage_start = (
                    expected_action.start_time_sec
                    if expected_action is not None and expected_action.start_time_sec is not None
                    else self.expected_stage_start_sec
                )
                self._broadcast_frame_status(
                    frame_id, t_sec, current_pred, expected_show, expected_conf, top3_text, vis, stage_start
                )

                frame_id += 1

            cap.release() # 释放视频资源
            if self.hands is not None:
                self.hands.close()

            # === 收尾：打包结果 → 可选保存日志 → 通知 UI 推理结束 ===
            # result：本次运行的输出目录（无则空串）+ 全部完成/超时事件列表
            result = {"run_dir": str(self.run_dir) if self.run_dir else "", "events": self.events}
            if self.params.save_log and self.run_dir:   # 勾选了"保存日志"且输出目录存在 → 导出 result.json + events.csv
                self._save_logs()

            self.sig_finished.emit(result)              # 把结果发给主线程 UI（on_finished 槽接收）

        except Exception as e:
            self.sig_error.emit(str(e))

    def _detect_frame(self, frame):
        """② MediaPipe 手部关键点，返回 (vis, hands_out)。

        单帧感知：这一帧里"手在哪"。
        - 输入：一帧 BGR 图像
        - 输出：
          · vis：渲染后的画面（手部关键点按参数画上），用于 UI 显示
          · hands_out：MediaPipe 手部关键点列表 [{hand_index, landmarks(21×3)}]，喂给特征拼装
        """
        vis = frame.copy()  # 注：每帧拷贝一次，渲染密集时是非主要瓶颈

        # === ② MediaPipe 手部 21 关键点（最多 2 只手）===
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

        return vis, hands_out

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
            # 关键：不同长度窗口都重采样到 48 → LSTM 输入维度统一为 (1, 48, 126)
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
        duration_sec = round(t_sec - (expected_action.start_time_sec or self.expected_stage_start_sec), 3)
        expected_action.duration_sec = duration_sec
        # 偏慢判定：耗时超过该动作历史平均耗时的 SLOW_RATIO_THRESHOLD 倍（需至少 2 个历史样本）
        hist = self.action_durations.setdefault(expected_action.fine_label, [])
        is_slow = len(hist) >= 2 and duration_sec > (sum(hist) / len(hist)) * SLOW_RATIO_THRESHOLD
        hist.append(duration_sec)
        quality = "slow" if is_slow else "normal"

        if self.params.save_snapshots:
            expected_action.snapshot_path = self._save_snapshot(vis, expected_action, t_sec)

        done_time_beijing = to_beijing_time_str()
        self.events.append(
            {
                "cycle": self.cycle,
                "index": expected_action.index + 1,
                "action": expected_action.show,
                "fine_label": expected_action.fine_label,
                "start_time_sec": round(expected_action.start_time_sec or self.expected_stage_start_sec, 3),
                "duration_sec": duration_sec,
                "start_time_beijing": expected_action.start_time_beijing or "",
                "done_time_sec": round(t_sec, 3),
                "done_time_beijing": done_time_beijing,
                "snapshot_path": expected_action.snapshot_path or "",
                "quality": quality,
                "status": "完成",
            }
        )

        self.sig_action.emit(
            {
                "index": expected_action.index,
                "status": "完成",
                "info": f"耗时 {duration_sec:.2f}s" + ("（偏慢）" if is_slow else ""),
                "snapshot": expected_action.snapshot_path,
                "duration_sec": duration_sec,
                "quality": quality,
            }
        )

        self.expected_idx += 1
        self.current_hit = 0

        # 多周期循环：4 步全部完成 → 重置状态 + cycle+1
        if self.expected_idx >= len(self.actions):
            self._finalize_cycle(t_sec)
            self._reset_cycle()

    def _timeout_action(self, expected_action, t_sec):
        """分支 B：超时跳过 → 事件 + 信号 + 推进 + 多周期重置。"""
        self.save_pending_sample(expected_action.fine_label, t_sec)   # 超时样本自动沉淀，供回灌训练提升召回
        duration_sec = round(t_sec - (expected_action.start_time_sec or self.expected_stage_start_sec), 3)
        expected_action.duration_sec = duration_sec
        done_time_beijing = to_beijing_time_str()
        self.events.append(
            {
                "cycle": self.cycle,
                "index": expected_action.index + 1,
                "action": expected_action.show,
                "fine_label": expected_action.fine_label,
                "start_time_sec": round(expected_action.start_time_sec or self.expected_stage_start_sec, 3),
                "duration_sec": duration_sec,
                "start_time_beijing": expected_action.start_time_beijing or "",
                "done_time_sec": round(t_sec, 3),
                "done_time_beijing": done_time_beijing,
                "snapshot_path": "",
                "quality": "timeout",
                "status": "超时跳过",
            }
        )

        self.sig_action.emit(
            {
                "index": expected_action.index,
                "status": "超时跳过",
                "info": f"超时 {duration_sec:.2f}s",
                "snapshot": None,
                "duration_sec": duration_sec,
                "quality": "timeout",
            }
        )

        self.expected_idx += 1
        self.current_hit = 0

        # 多周期循环：超时分支与完成分支一致
        if self.expected_idx >= len(self.actions):
            self._finalize_cycle(t_sec)
            self._reset_cycle()

    def _finalize_cycle(self, t_sec):
        """周期完成：计算 CT（D4 完成时间 − D1 开始时间），追加「周期完成」事件并广播 UI。"""
        if self.cycle_start_sec is None:
            return
        cycle_time = round(t_sec - self.cycle_start_sec, 3)
        self.cycle_times_sec.append(cycle_time)
        avg_cycle_time = round(sum(self.cycle_times_sec) / len(self.cycle_times_sec), 3)
        self.events.append(
            {
                "cycle": self.cycle,
                "event_type": "cycle",
                "action": "",
                "fine_label": "",
                "start_time_sec": round(self.cycle_start_sec, 3),
                "duration_sec": cycle_time,
                "start_time_beijing": self.cycle_start_beijing or "",
                "done_time_sec": round(t_sec, 3),
                "done_time_beijing": to_beijing_time_str(),
                "snapshot_path": "",
                "cycle_time_sec": cycle_time,
                "status": "周期完成",
            }
        )
        self.sig_action.emit(
            {
                "index": -1,
                "status": "周期完成",
                "info": f"CT {cycle_time:.2f}s",
                "snapshot": None,
                "cycle_time_sec": cycle_time,
                "avg_cycle_time_sec": avg_cycle_time,
            }
        )

        # MES / 中央系统上报：周期完成数据（含工位信息与各步耗时）
        if MES_ENABLED and MES_URL:
            payload = {
                "event_type": "cycle_completed",
                "cycle": self.cycle,
                "cycle_time_sec": cycle_time,
                "avg_cycle_time_sec": avg_cycle_time,
                "steps": [
                    {
                        "index": a.index + 1,
                        "fine_label": a.fine_label,
                        "duration_sec": a.duration_sec,
                        "status": "完成" if a.done else "超时跳过",
                    }
                    for a in self.actions
                ],
                "site": SITE_INFO,
                "video": self.video_source,
                "reported_at": to_beijing_time_str(),
            }
            ok = http_post_json(MES_URL, payload, MES_TOKEN)
            print(f"[MES] 周期 {self.cycle} 上报 {'成功' if ok else '失败'}")

    def _broadcast_frame_status(self, frame_id, t_sec, current_pred, expected_show, expected_conf, top3_text, vis, stage_start_sec=None):
        """⑧ 广播本帧状态 + 渲染画面到 UI。"""
        status = {
            "frame_id": frame_id,
            "time_sec": t_sec,
            "stage_start_sec": stage_start_sec if stage_start_sec is not None else self.expected_stage_start_sec,
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
            "video": self.video_path.name if self.video_path is not None else self.video_source,
            "site": SITE_INFO,
            "total_cycles": self.cycle,
            "completed_cycles": len(self.cycle_times_sec),
            "cycle_times_sec": self.cycle_times_sec,
            "avg_cycle_time_sec": (
                round(sum(self.cycle_times_sec) / len(self.cycle_times_sec), 3)
                if self.cycle_times_sec else None
            ),
            "completed_events": len(self.events),
            "total_steps": len(self.actions),
            "events": self.events,
        }
        json_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cycle", "index", "event_type", "action", "fine_label",
                    "start_time_sec", "duration_sec", "start_time_beijing",
                    "done_time_sec", "done_time_beijing", "snapshot_path",
                    "quality", "cycle_time_sec", "status",
                ],
                restval="",
            )
            writer.writeheader()
            for row in self.events:
                writer.writerow(row)
