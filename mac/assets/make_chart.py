#!/usr/bin/env python
"""The speed chart of mac/README.md: tokens per second of every measured system on the two workloads, our engine in
the accent color and the general engines in gray. Writes speed-light.svg and speed-dark.svg next to this file.

    python mac/assets/make_chart.py

The numbers are microseconds per token from eval/baselines_inkling-2-mini_m5max_c.json, runs-mac/baselines-20261123-ollama
and the six evaluations of the engine (medians); tokens per second = 1e6 / microseconds.
"""
from pathlib import Path

ROWS = [  # name, weights, microseconds per token on synthetic prompts, on natural text, ours
    ("megakernel (this repo)", "the checkpoint's own formats", 8100, 5143, True),
    ("oMLX + DSpark", "oQ4e requantization", 13767, 8307, False),
    ("oMLX", "oQ4e requantization", 16744, 16779, False),
    ("llama.cpp + DSpark", "Q4_K_M GGUF", 20556, 16323, False),
    ("llama.cpp", "Q4_K_M GGUF", 25130, 25030, False),
    ("mlx-lm", "the checkpoint's own formats", 30359, 30486, False),
    ("Ollama (MLX)", "the checkpoint's own formats", 31291, 30748, False),
]
THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", axis="#c3c2b7", accent="#2a78d6", gray="#898781"),
    "dark": dict(ink="#f0efec", ink2="#c3c2b7", axis="#383835", accent="#3987e5", gray="#898781"),
}
W, LABEL, PANEL, GAP, TOP, ROW, BAR, XMAX = 860, 232, 250, 64, 58, 42, 20, 200.0
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def bar(x0, y, w, h, fill):
    r = min(4.0, w)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} a{r},{r} 0 0 1 {r},{r} V{y + h - r:.1f} a{r},{r} 0 0 1 -{r},{r} H{x0:.1f} Z" '
            f'fill="{fill}"/>')


def chart(t):
    H = TOP + ROW * len(ROWS) + 14
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" font-family=\'{FONT}\' '
         f'aria-label="Tokens per second of Inkling-2-Mini on an Apple M5 Max: megakernel 123 on synthetic prompts and 194 on natural '
         f'text; oMLX with DSpark 73 and 120; llama.cpp with DSpark 49 and 61; mlx-lm 33 and 33">']
    xs = [LABEL, LABEL + PANEL + GAP]
    for x0, title in zip(xs, ("Synthetic prompts", "Natural text")):
        o.append(f'<text x="{x0}" y="20" font-size="14" font-weight="600" fill="{t["ink"]}">{title}</text>')
        o.append(f'<text x="{x0}" y="38" font-size="12" fill="{t["ink2"]}">tokens per second, one sequence</text>')
        o.append(f'<line x1="{x0}" y1="{TOP - 6}" x2="{x0}" y2="{TOP + ROW * len(ROWS) - 6}" stroke="{t["axis"]}" stroke-width="1"/>')
    for i, (name, weights, synth, natural, ours) in enumerate(ROWS):
        y = TOP + i * ROW
        mid = y + BAR / 2
        o.append(f'<text x="{LABEL - 14}" y="{mid - 1:.1f}" text-anchor="end" font-size="13" font-weight="{600 if ours else 400}" '
                 f'fill="{t["ink"]}">{name}</text>')
        o.append(f'<text x="{LABEL - 14}" y="{mid + 13:.1f}" text-anchor="end" font-size="10.5" fill="{t["ink2"]}">{weights}</text>')
        for x0, us in zip(xs, (synth, natural)):
            v = 1e6 / us
            w = v / XMAX * PANEL
            o.append(bar(x0, y, w, BAR, t["accent"] if ours else t["gray"]))
            o.append(f'<text x="{x0 + w + 7:.1f}" y="{mid + 4.5:.1f}" font-size="13" font-weight="{600 if ours else 400}" '
                     f'fill="{t["ink"]}">{v:.0f}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name, theme in THEMES.items():
        (here / f"speed-{name}.svg").write_text(chart(theme))
        print(here / f"speed-{name}.svg")
