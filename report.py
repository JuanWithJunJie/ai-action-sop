"""HTML 生产报表生成器：聚合 runs_gui/ 下的推理结果，输出浏览器可打开的报表。

用法：
    python report.py                          # 聚合 runs_gui/ 下全部结果
    python report.py --run-dir runs_gui/xxx   # 只看某一次运行
    python report.py --output report.html     # 自定义输出路径

报表内容：概览卡片（周期数/平均CT/合格率/超时）、CT 趋势折线、瓶颈步骤分析（平均耗时+超时次数）、周期明细表。
纯 HTML+CSS+SVG 实现，无外部依赖，工厂离线环境可直接打开。
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from ai_sop.core.constants import BASE_DIR, RUNS_GUI_DIR

STEP_LABELS = ["D1_pick_material", "D2_tear_film", "D3_inspect", "D4_place_material"]
STEP_CN = {"D1_pick_material": "取料", "D2_tear_film": "撕膜", "D3_inspect": "检测", "D4_place_material": "放料"}


def load_runs(runs_dir: Path) -> list:
    """读取 runs_dir 下所有 result.json。"""
    runs = []
    if not runs_dir.exists():
        return runs
    for rd in sorted(runs_dir.glob("*/")):
        rj = rd / "result.json"
        if rj.exists():
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
                runs.append({"name": rd.name, "data": data})
            except Exception:
                continue
    return runs


def compute_stats(runs: list) -> dict:
    """聚合所有运行的周期 CT、每步耗时、完成/超时统计。"""
    cycle_times = []
    step_durations = {k: [] for k in STEP_LABELS}
    step_timeouts = {k: 0 for k in STEP_LABELS}
    step_completed = {k: 0 for k in STEP_LABELS}
    total_events = 0
    completed = 0
    timeouts = 0

    for run in runs:
        data = run["data"]
        cycle_times.extend(data.get("cycle_times_sec", []))
        for ev in data.get("events", []):
            total_events += 1
            label = ev.get("fine_label", "")
            status = ev.get("status", "")
            if status == "完成" and label in step_durations:
                step_durations[label].append(ev.get("duration_sec", 0.0))
                step_completed[label] += 1
                completed += 1
            elif status == "超时跳过":
                if label in step_timeouts:
                    step_timeouts[label] += 1
                timeouts += 1

    return {
        "run_count": len(runs),
        "cycle_times": cycle_times,
        "avg_ct": round(sum(cycle_times) / len(cycle_times), 3) if cycle_times else None,
        "step_durations": {k: (round(sum(v) / len(v), 3) if v else None) for k, v in step_durations.items()},
        "step_timeouts": step_timeouts,
        "completed": completed,
        "timeouts": timeouts,
        "total_events": total_events,
    }


def compute_stats_by_station(runs: list) -> dict:
    """按工位（site.station）分组统计，用于多工位对比。"""
    stations: dict = {}
    for run in runs:
        st = run["data"].get("site", {}).get("station", "未知工位")
        stations.setdefault(st, []).append(run)
    return {st: compute_stats(grp) for st, grp in stations.items()}


def svg_line_chart(values: list, width=860, height=240) -> str:
    """CT 趋势折线（SVG polyline），无外部依赖。"""
    if len(values) < 2:
        return '<p style="color:#5a6b7d;font-size:13px;">数据不足，无法绘制趋势（至少需要 2 个周期）。</p>'
    pad_l, pad_r, pad_t, pad_b = 48, 24, 18, 34
    vmax = max(values) * 1.15 or 1.0
    vmin = min(values) * 0.85
    span = max(vmax - vmin, 1e-6)
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = pad_l + i * (width - pad_l - pad_r) / (n - 1)
        y = pad_t + (vmax - v) / span * (height - pad_t - pad_b)
        pts.append(f"{x:.1f},{y:.1f}")
    # Y 轴网格线
    grid = ""
    for g in range(5):
        gy = pad_t + g * (height - pad_t - pad_b) / 4
        val = vmax - g * span / 4
        grid += f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" stroke="#1d2a3a" stroke-width="1"/>'
        grid += f'<text x="{pad_l-8}" y="{gy+4:.1f}" fill="#5a6b7d" font-size="11" text-anchor="end">{val:.1f}</text>'
    # X 轴标签（周期号）
    xlabels = "".join(
        f'<text x="{pad_l + i * (width - pad_l - pad_r) / (n - 1):.1f}" y="{height-12}" fill="#5a6b7d" font-size="11" text-anchor="middle">C{i+1}</text>'
        for i in range(n)
    )
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      {grid}
      <polyline points="{' '.join(pts)}" fill="none" stroke="#00d4ff" stroke-width="2.5"/>
      {xlabels}
    </svg>"""


def svg_bar_chart(stats: dict, width=860, height=240) -> str:
    """瓶颈步骤分析：每步平均耗时柱状 + 超时次数标注。"""
    items = [(k, stats["step_durations"][k], stats["step_timeouts"][k]) for k in STEP_LABELS]
    vmax = max((v for _, v, _ in items if v is not None), default=1.0) * 1.2 or 1.0
    pad_l, pad_r, pad_t, pad_b = 48, 24, 18, 34
    bw = (width - pad_l - pad_r) / len(items) * 0.5
    html = ""
    for i, (label, avg, to) in enumerate(items):
        cx = pad_l + i * (width - pad_l - pad_r) / len(items) + (width - pad_l - pad_r) / len(items) / 2
        if avg is None:
            bh = 4
            bar_y = height - pad_b - bh
        else:
            bh = max(avg / vmax * (height - pad_t - pad_b), 3)
            bar_y = height - pad_b - bh
        color = "#00ff88" if to == 0 else "#ff4466"
        html += f"""
        <rect x="{cx-bw/2:.1f}" y="{bar_y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{color}"/>
        <text x="{cx:.1f}" y="{bar_y-6:.1f}" fill="#c9d4e0" font-size="12" text-anchor="middle">{avg if avg is not None else 0:.2f}s</text>
        <text x="{cx:.1f}" y="{height-12}" fill="#5a6b7d" font-size="12" text-anchor="middle">{STEP_CN.get(label, label)}</text>
        <text x="{cx:.1f}" y="{height-pad_b-10 if avg is None else height-pad_b-bh-20:.1f}" fill="#ff4466" font-size="10" text-anchor="middle">{"超时 " + str(to) if to else ""}</text>
        """
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      {html}
    </svg>"""


def render_html(stats: dict, sources: list, station_stats: dict = None) -> str:
    ct_line = svg_line_chart(stats["cycle_times"])
    bar_chart = svg_bar_chart(stats)
    rate = round(stats["completed"] / stats["total_events"] * 100, 1) if stats["total_events"] else 0.0
    station_rows = ""
    if station_stats:
        for st, s in station_stats.items():
            s_rate = round(s["completed"] / s["total_events"] * 100, 1) if s["total_events"] else 0.0
            station_rows += (
                f"<tr><td>{st}</td><td>{len(s['cycle_times'])}</td>"
                f"<td>{s['avg_ct'] if s['avg_ct'] is not None else '--'}s</td>"
                f"<td>{s_rate}%</td><td>{s['timeouts']}</td></tr>"
            )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>AI-SOP 生产报表</title>
<style>
  body {{ background:#0d141f; color:#c9d4e0; font-family:'Microsoft YaHei',sans-serif; margin:0; padding:24px; }}
  h1 {{ color:#00d4ff; font-size:22px; margin:0 0 4px; }}
  .sub {{ color:#5a6b7d; font-size:12px; margin-bottom:20px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:22px; }}
  .card {{ background:#131c2b; border:1px solid rgba(0,212,255,.12); border-radius:10px; padding:16px 22px; min-width:150px; }}
  .card .num {{ font-size:26px; font-weight:700; color:#00d4ff; font-family:Consolas,monospace; }}
  .card .lbl {{ font-size:11px; color:#5a6b7d; margin-top:4px; }}
  .panel {{ background:#131c2b; border:1px solid rgba(0,212,255,.12); border-radius:10px; padding:18px 20px; margin-bottom:22px; }}
  .panel h2 {{ font-size:13px; color:#00d4ff; letter-spacing:2px; margin:0 0 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:7px 10px; text-align:center; border-bottom:1px solid #1d2a3a; }}
  th {{ color:#5a6b7d; font-size:11px; }}
  td {{ font-family:Consolas,monospace; }}
  .src {{ color:#5a6b7d; font-size:12px; margin-bottom:14px; }}
</style>
</head>
<body>
  <h1>AI-SOP 生产报表</h1>
  <div class="sub">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 数据来源：{len(sources)} 次运行</div>
  <div class="cards">
    <div class="card"><div class="num">{stats['avg_ct'] if stats['avg_ct'] is not None else '--'}s</div><div class="lbl">平均 CT</div></div>
    <div class="card"><div class="num">{len(stats['cycle_times'])}</div><div class="lbl">完成周期</div></div>
    <div class="card"><div class="num" style="color:#00ff88">{rate}%</div><div class="lbl">步骤完成率</div></div>
    <div class="card"><div class="num" style="color:#ff4466">{stats['timeouts']}</div><div class="lbl">超时步骤</div></div>
  </div>
  <div class="panel">
    <h2>CT 趋势（每周期节拍）</h2>
    {ct_line}
  </div>
  <div class="panel">
    <h2>瓶颈步骤分析（平均耗时 / 超时次数）</h2>
    {bar_chart}
  </div>
  <div class="panel">
    <h2>运行明细</h2>
    {''.join(f'<div class="src">• {s["name"]}：{s["data"].get("video", "")}</div>' for s in sources)}
  </div>
  {f'''
  <div class="panel">
    <h2>多工位对比</h2>
    <table>
      <tr><th>工位</th><th>完成周期</th><th>平均CT</th><th>步骤完成率</th><th>超时步骤</th></tr>
      {station_rows}
    </table>
  </div>''' if station_rows else ""}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="生成 HTML 生产报表")
    parser.add_argument("--run-dir", default="", help="单次运行目录（默认聚合 runs_gui/ 全部）")
    parser.add_argument("--output", default="production_report.html", help="报表输出路径")
    args = parser.parse_args()

    if args.run_dir:
        runs = load_runs(Path(args.run_dir))
        label = f"运行目录 {args.run_dir}"
    else:
        runs = load_runs(RUNS_GUI_DIR)
        label = f"全部运行（{RUNS_GUI_DIR}）"

    if not runs:
        print(f"未找到任何 result.json（{label}），请先运行一次分析")
        return

    stats = compute_stats(runs)
    station_stats = compute_stats_by_station(runs)
    html = render_html(stats, runs, station_stats)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"报表已生成: {out}")
    print(f"运行 {len(runs)} 次 | 周期 {len(stats['cycle_times'])} 个 | 平均CT {stats['avg_ct']}s | 步骤完成率 "
          f"{round(stats['completed'] / stats['total_events'] * 100, 1) if stats['total_events'] else 0}%")


if __name__ == "__main__":
    main()
