#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MyRobot 通用 CSV 曲线工具。

图形模式：
    python myrobot_plot_data.py --file viewer_telemetry.csv
命令行模式：
    python myrobot_plot_data.py --file viewer_telemetry.csv --list
    python myrobot_plot_data.py --file viewer_telemetry.csv --preset torque --start 2 --end 5 --stats
    python myrobot_plot_data.py --file viewer_telemetry.csv --y left_hip_torque_Nm right_hip_torque_Nm --separate
    python myrobot_plot_data.py --file viewer_telemetry.csv --y col1,col2 --no-show --out curves.pdf

依赖：pandas、numpy、matplotlib；图形选择器使用 Python 标准库 tkinter。
不会修改原 CSV 数据。力矩曲线是 CSV 字段本身，程序不会擅自取反或滤波。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_fonts():
    """优先采用系统已安装的中文字体；不随程序打包字体文件。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
                      "Microsoft YaHei", "SimHei", "Arial Unicode MS"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


PRESETS = {
    "torque": ("torque", "effort", "_tau", "tau_"),
    "joint": ("hip", "knee", "joint"),
    "wheel": ("wheel", "omega_l", "omega_r"),
    "attitude": ("roll", "pitch", "yaw"),
    "speed": ("vel", "speed"),
    "clearance": ("clr", "clearance"),
    "contact": ("contact", "fn_", "force"),
}
TIME_HINTS = ("timestamp_s", "time_s", "timestamp", "sim_time", "time", "t")
EVENT_NAMES = (
    "TRIGGER_JUMP", "ENTER_STOPPING", "ENTER_PREPARE_JUMP", "ENTER_JUMP",
    "LIFTOFF", "AIR_APEX", "TOUCHDOWN", "TOUCHDOWN_ENTER_RECOVER",
    "NORMAL_HANDOFF", "TIMEOUT_HANDOFF", "RECOVER_ABORT", "HANDOFF_TO_BALANCE"
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV 文件：{path}")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    data.columns = [str(c).strip() for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("CSV 列名重复，需先修正日志列名以避免误选")
    if data.empty:
        raise ValueError("CSV 没有数据行")
    return data


def numeric_columns(data: pd.DataFrame) -> list[str]:
    good = []
    for c in data.columns:
        num = pd.to_numeric(data[c], errors="coerce")
        if np.isfinite(num.to_numpy(dtype=float)).any():
            good.append(c)
    return good


def auto_x(numeric: list[str]) -> str:
    lower = {c.lower(): c for c in numeric}
    for hint in TIME_HINTS:
        if hint in lower:
            return lower[hint]
    return numeric[0]


def parse_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(s.strip() for entry in names for s in entry.split(",") if s.strip()))


def preset_columns(preset: str, numeric: list[str], x: str) -> list[str]:
    if preset == "all":
        return [c for c in numeric if c != x]
    patterns = PRESETS[preset]
    return [c for c in numeric if c != x and any(p in c.lower() for p in patterns)]


def clipped_xy(data: pd.DataFrame, x: str, y: str, start: float | None, end: float | None):
    xnum = pd.to_numeric(data[x], errors="coerce").to_numpy(dtype=float)
    ynum = pd.to_numeric(data[y], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xnum) & np.isfinite(ynum)
    if start is not None:
        ok &= (xnum >= start)
    if end is not None:
        ok &= (xnum <= end)
    return xnum[ok], ynum[ok]


def event_positions(data: pd.DataFrame, x: str, start: float | None, end: float | None):
    tag_col = None
    for candidate in ("event_tag", "event"):
        if candidate in data.columns:
            tag_col = candidate
            break
    if tag_col is None:
        return []
    xx = pd.to_numeric(data[x], errors="coerce").to_numpy(dtype=float)
    tags = data[tag_col].fillna("").astype(str).to_numpy()
    out = []
    for xv, tag in zip(xx, tags):
        if not np.isfinite(xv) or not tag.strip():
            continue
        if start is not None and xv < start:
            continue
        if end is not None and xv > end:
            continue
        labels = [n for n in EVENT_NAMES if n in tag.upper()]
        out.append((float(xv), "/".join(labels) if labels else tag))
    return out[:150]  # 防止异常日志中大量事件使图表不可读


def format_stats(values: np.ndarray) -> str:
    return (f"min={values.min():.4g}, max={values.max():.4g}, "
            f"mean={values.mean():.4g}, RMS={np.sqrt(np.mean(values**2)):.4g}, n={len(values)}")


def make_plot(data: pd.DataFrame, x: str, ys: list[str], *, start=None, end=None,
              separate=False, show_events=True, stats=False, title=None,
              ylabel=None, hlines=(), out=None, show=True):
    numeric = numeric_columns(data)
    missing = [c for c in [x, *ys] if c not in numeric]
    if missing:
        raise ValueError(f"列不存在或非数值：{missing}\n可用列：{', '.join(numeric)}")
    if not ys:
        raise ValueError("请至少选一个 Y 轴数据列")
    if start is not None and end is not None and start >= end:
        raise ValueError("时间范围要求 --start 小于 --end")

    nplots = len(ys) if separate else 1
    fig, axes = plt.subplots(nplots, 1, sharex=True, figsize=(13, max(4.5, 2.7*nplots + 1)),
                             squeeze=False, layout="constrained")
    axs = axes[:, 0]
    for idx, field in enumerate(ys):
        ax = axs[idx if separate else 0]
        xx, yy = clipped_xy(data, x, field, start, end)
        if len(xx) == 0:
            print(f"警告：{field} 在所选范围内没有有效数值", file=sys.stderr)
            continue
        ax.plot(xx, yy, linewidth=1.4, label=field)
        ax.grid(True, linestyle=":", alpha=0.5)
        if stats:
            print(f"{field}: {format_stats(yy)}")
        if separate:
            ax.set_ylabel(ylabel or field)
        for limit in hlines:
            ax.axhline(limit, linestyle="--", alpha=0.6, linewidth=1.0)

    events = event_positions(data, x, start, end) if show_events else []
    for ax in axs:
        if not separate:
            ax.set_ylabel(ylabel or "数值（单位见 CSV 列名）")
        if not separate:
            for limit in hlines:
                ax.axhline(limit, linestyle="--", alpha=0.6, linewidth=1.0)
        for xv, label in events:
            ax.axvline(
                xv,
                linestyle=":",
                alpha=0.15,
                linewidth=0.8
            )
        if not separate:
            ax.legend(loc="best", fontsize=9)
        else:
            ax.legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel(x)
    if start is not None or end is not None:
        for ax in axs:
            lo, hi = ax.get_xlim()
            ax.set_xlim(start if start is not None else lo, end if end is not None else hi)
    fig.suptitle(title or "MyRobot 遥测曲线", fontsize=14)
    if out:
        target = Path(out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300)
        print(f"已保存：{target.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def launch_gui(csv_path: Path):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("当前 Python 未安装 tkinter。Ubuntu 可安装 python3-tk；或者直接用 --y 命令行绘图。") from exc

    root = tk.Tk()
    root.title("MyRobot 通用 CSV 绘图器")
    root.geometry("910x700")
    state = {"data": None, "numeric": []}

    file_var = tk.StringVar(value=str(csv_path))
    x_var = tk.StringVar()
    start_var = tk.StringVar()
    end_var = tk.StringVar()
    title_var = tk.StringVar(value="MyRobot 遥测曲线")
    ylabel_var = tk.StringVar()
    out_var = tk.StringVar(value="")
    separate_var = tk.BooleanVar(value=False)
    events_var = tk.BooleanVar(value=True)
    stats_var = tk.BooleanVar(value=False)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(3, weight=1)

    ttk.Label(frame, text="CSV 文件").grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=file_var).grid(row=0, column=1, sticky="ew", padx=6)

    def choose_file():
        path = filedialog.askopenfilename(filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")])
        if path:
            file_var.set(path)
            load_file()

    ttk.Button(frame, text="浏览…", command=choose_file).grid(row=0, column=2)

    ttk.Label(frame, text="X 轴").grid(row=1, column=0, sticky="w", pady=(10, 5))
    x_cb = ttk.Combobox(frame, textvariable=x_var, state="readonly")
    x_cb.grid(row=1, column=1, sticky="ew", padx=6)
    ttk.Button(frame, text="读取 CSV", command=lambda: load_file()).grid(row=1, column=2)

    ttk.Label(frame, text="选择 Y 轴曲线（Ctrl / Shift 多选）").grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 2))
    listframe = ttk.Frame(frame)
    listframe.grid(row=3, column=0, columnspan=3, sticky="nsew")
    listframe.columnconfigure(0, weight=1)
    listframe.rowconfigure(0, weight=1)
    lb = tk.Listbox(listframe, selectmode=tk.EXTENDED, exportselection=False, font=("DejaVu Sans Mono", 10))
    lb.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(listframe, orient="vertical", command=lb.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    lb.configure(yscrollcommand=scrollbar.set)

    def load_file():
        try:
            data = read_csv(Path(file_var.get()).expanduser())
            numeric = numeric_columns(data)
            if not numeric:
                raise ValueError("CSV 中没有可绘制的数值列")
            state.update(data=data, numeric=numeric)
            x_cb["values"] = numeric
            x_var.set(auto_x(numeric))
            lb.delete(0, tk.END)
            for name in numeric:
                lb.insert(tk.END, name)
            suggested = preset_columns("torque", numeric, x_var.get())
            if not suggested:
                suggested = [c for c in numeric if c != x_var.get()][:4]
            for i, name in enumerate(numeric):
                if name in suggested[:8]:
                    lb.selection_set(i)
            root.title(f"MyRobot 通用 CSV 绘图器 — {Path(file_var.get()).name}（{len(data)} 行）")
        except Exception as exc:
            messagebox.showerror("加载 CSV 失败", str(exc))

    options = ttk.Frame(frame)
    options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=9)
    for i in range(8):
        options.columnconfigure(i, weight=1)
    ttk.Label(options, text="起点").grid(row=0, column=0)
    ttk.Entry(options, textvariable=start_var, width=9).grid(row=0, column=1)
    ttk.Label(options, text="终点").grid(row=0, column=2)
    ttk.Entry(options, textvariable=end_var, width=9).grid(row=0, column=3)
    ttk.Checkbutton(options, text="分开子图", variable=separate_var).grid(row=0, column=4)
    ttk.Checkbutton(options, text="标记事件", variable=events_var).grid(row=0, column=5)
    ttk.Checkbutton(options, text="打印统计", variable=stats_var).grid(row=0, column=6)

    ttk.Label(frame, text="图标题").grid(row=5, column=0, sticky="w")
    ttk.Entry(frame, textvariable=title_var).grid(row=5, column=1, sticky="ew", padx=6)
    ttk.Label(frame, text="Y轴说明（可空）").grid(row=6, column=0, sticky="w")
    ttk.Entry(frame, textvariable=ylabel_var).grid(row=6, column=1, sticky="ew", padx=6)
    ttk.Label(frame, text="保存路径（可空）").grid(row=7, column=0, sticky="w")
    ttk.Entry(frame, textvariable=out_var).grid(row=7, column=1, sticky="ew", padx=6)

    def select_preset(name):
        cols = state["numeric"]
        lb.selection_clear(0, tk.END)
        matched = preset_columns(name, cols, x_var.get())
        if not matched:
            messagebox.showinfo("没有匹配列", f"当前 CSV 没有 {name} 对应的字段，可在列表中任意选择。")
        for i, c in enumerate(cols):
            if c in matched:
                lb.selection_set(i)

    presetframe = ttk.Frame(frame)
    presetframe.grid(row=8, column=0, columnspan=3, sticky="ew", pady=6)
    ttk.Label(presetframe, text="快速筛选：").pack(side="left")
    for name in (*PRESETS, "all"):
        ttk.Button(presetframe, text=name, command=lambda n=name: select_preset(n)).pack(side="left", padx=2)

    def do_plot(save_only=False):
        try:
            if state["data"] is None:
                load_file()
            if state["data"] is None:
                return
            ys = [state["numeric"][i] for i in lb.curselection() if state["numeric"][i] != x_var.get()]
            if not ys:
                raise ValueError("请选择至少一条 Y 轴曲线（X 轴字段不能同时当 Y 轴）")
            out = out_var.get().strip() or None
            if save_only and not out:
                out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if not out:
                    return
                out_var.set(out)
            make_plot(state["data"], x_var.get(), ys,
                      start=float(start_var.get()) if start_var.get().strip() else None,
                      end=float(end_var.get()) if end_var.get().strip() else None,
                      separate=separate_var.get(), show_events=events_var.get(),
                      stats=stats_var.get(), title=title_var.get(), ylabel=ylabel_var.get() or None,
                      out=out if save_only else None, show=not save_only)
        except Exception as exc:
            messagebox.showerror("绘图失败", str(exc))

    buttons = ttk.Frame(frame)
    buttons.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    ttk.Button(buttons, text="显示交互式图表（可缩放）", command=do_plot).pack(side="left", padx=4)
    ttk.Button(buttons, text="保存 PNG / PDF / SVG", command=lambda: do_plot(True)).pack(side="left", padx=4)
    ttk.Button(buttons, text="退出", command=root.destroy).pack(side="right", padx=4)

    if csv_path.exists():
        root.after(50, load_file)
    root.mainloop()


def main():
    configure_fonts()
    parser = argparse.ArgumentParser(description="MyRobot 通用 CSV 曲线绘制，支持 GUI 自选字段或 CLI")
    parser.add_argument("--file", default="viewer_telemetry.csv", help="CSV 路径")
    parser.add_argument("--list", action="store_true", help="列出 CSV 可绘制的字段")
    parser.add_argument("--x", help="X 轴列名，默认自动检测时间")
    parser.add_argument("--y", nargs="+", help="任意 Y 轴列名，空格或逗号分隔")
    parser.add_argument("--preset", choices=(*PRESETS, "all"), help="自动按关键词选择字段；也可自行 --y 精选")
    parser.add_argument("--start", type=float, help="X 轴显示区间起点")
    parser.add_argument("--end", type=float, help="X 轴显示区间终点")
    parser.add_argument("--separate", action="store_true", help="每条曲线放独立子图")
    parser.add_argument("--stats", action="store_true", help="输出 min、max、mean、RMS")
    parser.add_argument("--no-events", action="store_true", help="不显示事件竖线 (event_tag / event)")
    parser.add_argument("--hline", type=float, action="append", default=[], help="添加水平阈值线，可重复")
    parser.add_argument("--title", help="图标题")
    parser.add_argument("--ylabel", help="Y 轴标签")
    parser.add_argument("--out", help="导出 PNG / PDF / SVG 路径")
    parser.add_argument("--no-show", action="store_true", help="不显示窗口，仅输出图片")
    parser.add_argument("--gui", action="store_true", help="打开图形选择器")
    args = parser.parse_args()
    path = Path(args.file).expanduser()
    if args.gui or not (args.y or args.preset or args.list or args.out):
        launch_gui(path)
        return
    if args.out and not (args.y or args.preset or args.list):
        args.preset = "torque"
    data = read_csv(path)
    numeric = numeric_columns(data)
    if args.list:
        print(f"CSV: {path}，{len(data)} 行")
        for name in data.columns:
            print(f"{'[数值]' if name in numeric else '[文字]'} {name}")
        return
    if not numeric:
        raise ValueError("CSV 中没有数值列")
    x = args.x or auto_x(numeric)
    ys = parse_names(args.y) if args.y else preset_columns(args.preset, numeric, x)
    if not ys:
        raise ValueError(f"找不到 '{args.preset}' 对应的字段，请先用 --list 查看列名，然后使用 --y 明确指定")
    make_plot(data, x, ys, start=args.start, end=args.end, separate=args.separate,
              show_events=not args.no_events, stats=args.stats, title=args.title,
              ylabel=args.ylabel, hlines=args.hline, out=args.out, show=not args.no_show)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(2)
