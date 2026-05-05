"""Render the Stage 3 QwenOFT + V-JEPA architecture figure."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 980
BG = (248, 250, 252)
TEXT = (15, 23, 42)
MUTED = (71, 85, 105)
LINE = (148, 163, 184)
PRIMARY = (37, 99, 235)
SECONDARY = (16, 185, 129)
ACCENT = (245, 158, 11)
PANEL = (255, 255, 255)


def font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = font(42)
SUBTITLE = font(22)
BODY = font(24)
SMALL = font(20)


def rounded_box(draw: ImageDraw.ImageDraw, xy, fill, outline, width=3, radius=24):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def box(draw, xy, title, lines, fill, outline):
    rounded_box(draw, xy, fill=fill, outline=outline)
    x0, y0, x1, y1 = xy
    draw.text((x0 + 24, y0 + 20), title, font=BODY, fill=TEXT)
    y = y0 + 62
    for line in lines:
        draw.text((x0 + 24, y), line, font=SMALL, fill=MUTED)
        y += 34


def arrow(draw, start, end, fill, width=8):
    draw.line([start, end], fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return
    import math

    angle = math.atan2(dy, dx)
    head_len = 18
    head_angle = 0.6
    p1 = (
        end[0] - head_len * math.cos(angle - head_angle),
        end[1] - head_len * math.sin(angle - head_angle),
    )
    p2 = (
        end[0] - head_len * math.cos(angle + head_angle),
        end[1] - head_len * math.sin(angle + head_angle),
    )
    draw.polygon([end, p1, p2], fill=fill)


def render(out_path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((70, 54), "Stage 3: QwenOFT + offline / online V-JEPA path", font=TITLE, fill=TEXT)
    draw.text(
        (72, 112),
        "Training consumes offline episode caches; LIBERO sim eval falls back to an online encoder when episode/frame ids are absent.",
        font=SUBTITLE,
        fill=MUTED,
    )

    box(
        draw,
        (70, 190, 470, 420),
        "LeRobot sample / LIBERO obs",
        [
            "image: [primary, wrist]",
            "lang / optional state",
            "episode_id + frame_id during training",
        ],
        fill=(239, 246, 255),
        outline=PRIMARY,
    )
    box(
        draw,
        (570, 190, 980, 420),
        "V-JEPA source",
        [
            "train: offline .npz cache reader",
            "eval: online encoder + rolling history",
            "pooled feature (+ optional tokens)",
        ],
        fill=(236, 253, 245),
        outline=SECONDARY,
    )
    box(
        draw,
        (1080, 190, 1490, 420),
        "VJepaProjector",
        [
            "2-layer MLP + LayerNorm",
            "fusion = film_gating | concat_tokens | cross_attn",
            "separate LR / freeze-friendly",
        ],
        fill=(255, 251, 235),
        outline=ACCENT,
    )
    box(
        draw,
        (570, 520, 980, 760),
        "Qwen3-VL backbone",
        [
            "images + instruction + action tokens",
            "last hidden state",
            "action-token query extraction",
        ],
        fill=PANEL,
        outline=LINE,
    )
    box(
        draw,
        (1080, 520, 1490, 760),
        "OFT action head",
        [
            "conditioned action queries",
            "predict action_horizon x action_dim",
            "L1 loss / normalized_actions",
        ],
        fill=PANEL,
        outline=LINE,
    )

    arrow(draw, (470, 305), (570, 305), PRIMARY)
    arrow(draw, (980, 305), (1080, 305), SECONDARY)
    arrow(draw, (270, 420), (720, 520), LINE)
    arrow(draw, (1285, 420), (1285, 520), ACCENT)
    arrow(draw, (980, 640), (1080, 640), LINE)

    draw.rounded_rectangle((70, 810, 1730, 915), radius=26, fill=(255, 255, 255), outline=LINE, width=2)
    draw.text((100, 838), "Offline train path", font=BODY, fill=PRIMARY)
    draw.text((330, 838), "sample -> cache lookup -> projector fusion -> OFT loss", font=BODY, fill=MUTED)
    draw.text((930, 838), "Online eval path", font=BODY, fill=SECONDARY)
    draw.text((1140, 838), "sample -> history buffer -> V-JEPA encoder -> projector fusion -> predict_action", font=BODY, fill=MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


if __name__ == "__main__":
    render(Path(__file__).with_name("architecture.png"))
