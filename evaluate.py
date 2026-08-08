"""批量评估脚本：对 video/ 下所有视频跑推理，输出准确率 / 耗时报告。

用法：
    python evaluate.py                         # 评估 video/ 下全部视频
    python evaluate.py --video-dir D:/videos   # 指定视频目录
    python evaluate.py --output report.json    # 自定义报告路径

报告内容：每个视频的完成/超时/准确率/周期CT，以及全部视频的整体准确率与平均CT。
运行前提：video 目录下已有测试视频，且 train_data/timeline.csv 有对应标注（无标注则按默认 4 步评估）。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QCoreApplication

from ai_sop.core.constants import BASE_DIR
from ai_sop.core.models import RuntimeParams
from ai_sop.core.worker import InferenceWorker

DEFAULT_VIDEO_DIR = BASE_DIR / "video"


def evaluate_video(video_path: Path, app: QCoreApplication) -> dict:
    """跑单个视频的完整推理，返回评估统计。"""
    params = RuntimeParams(
        lstm_conf=0.15,
        confirm_frames=4,
        show_keypoints=False,
        save_snapshots=False,
        save_log=False,
    )
    worker = InferenceWorker(str(video_path), params)
    holder: dict = {}

    def on_finished(res: dict):
        holder["result"] = res
        app.quit()

    def on_error(msg: str):
        holder["error"] = msg
        app.quit()

    worker.sig_finished.connect(on_finished)
    worker.sig_error.connect(on_error)
    worker.start()
    app.exec_()

    if "error" in holder:
        return {"video": video_path.name, "error": holder["error"]}

    result = holder["result"]
    events = result.get("events", [])
    total_steps = result.get("total_steps", 4)
    completed = sum(1 for e in events if e.get("status") == "完成")
    timeouts = sum(1 for e in events if e.get("status") == "超时跳过")
    cycle_times = result.get("cycle_times_sec", [])
    return {
        "video": video_path.name,
        "total_steps": total_steps,
        "completed": completed,
        "timeouts": timeouts,
        "accuracy": round(completed / total_steps, 3) if total_steps else 0.0,
        "cycle_times_sec": cycle_times,
        "avg_cycle_time_sec": result.get("avg_cycle_time_sec"),
    }


def main():
    parser = argparse.ArgumentParser(description="批量评估 AI-SOP 推理准确率")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="视频目录（默认 video/）")
    parser.add_argument("--output", default="evaluate_report.json", help="报告输出路径")
    args = parser.parse_args()

    vdir = Path(args.video_dir)
    videos = sorted(vdir.glob("*.mp4"))
    if not videos:
        print(f"目录中没有视频: {vdir}")
        return

    app = QCoreApplication(sys.argv)
    report = {"date": datetime.now().isoformat(timespec="seconds"), "videos": []}

    for v in videos:
        print(f"评估: {v.name} ...")
        r = evaluate_video(v, app)
        report["videos"].append(r)
        if "error" in r:
            print(f"  -> 出错: {r['error']}")
        else:
            print(f"  -> 完成 {r['completed']}/{r['total_steps']}  超时 {r['timeouts']}  准确率 {r['accuracy']:.1%}")

    total_steps = sum(v.get("total_steps", 0) for v in report["videos"])
    completed = sum(v.get("completed", 0) for v in report["videos"])
    cts = [t for v in report["videos"] for t in v.get("cycle_times_sec", [])]
    report["overall"] = {
        "total_videos": len(report["videos"]),
        "total_steps": total_steps,
        "completed": completed,
        "accuracy": round(completed / total_steps, 3) if total_steps else 0.0,
        "avg_cycle_time_sec": round(sum(cts) / len(cts), 3) if cts else None,
    }

    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {out}")
    print(f"总准确率: {report['overall']['accuracy']:.1%}  平均CT: {report['overall'].get('avg_cycle_time_sec')}")


if __name__ == "__main__":
    main()
