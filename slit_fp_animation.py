"""
Slit Aperture for Fourier Ptychography - poster animation (ICCP 2026).

Manim Community Edition (tested with 0.20.1). Render inside the `fp` conda env:

    conda run -n fp manim -pql slit_fp_animation.py Full     # fast full preview
    conda run -n fp manim -pqh slit_fp_animation.py Full     # HD
    conda run -n fp manim -pql slit_fp_animation.py Stage7   # iterate on one stage

Design notes
------------
* Every tunable quantity lives in a per-stage CONFIG dict near the top so you can
  iterate on a single stage without touching the others.
* Each stage is a standalone `run_stageN(scene, carry=None)` function plus a matching
  `StageN(Scene)` wrapper (so `manim ... StageN` renders just that stage). `Full`
  chains the functions and threads a `carry` object through the stages that must
  share/persist objects (2->4 rebuild, 4->5 persist, 6->7 persist).
* Images are placeholders (`placeholder_image`) that draw a labelled gray box until
  you drop a real file at the configured path. No image files are needed to render.
"""

import os
import math

from manim import *
import numpy as np


# =============================================================================
# GLOBAL (shared look + layout)
# =============================================================================
GLOBAL = {
    "background": "#000000",
    "blue_outline": BLUE_D,
    "blue_fill": BLUE,
    "green_line": GREEN,
    "gray_placeholder": GREY_B,
    "fade_time": 0.6,          # default cross-stage fade duration
}

# Convenient screen anchors (depend on the configured frame size)
FRAME_W = config.frame_width
FRAME_H = config.frame_height
LEFT_HALF = np.array([-FRAME_W / 4.0, 0.0, 0.0])
RIGHT_HALF = np.array([FRAME_W / 4.0, 0.0, 0.0])


# ---- Stage 1: Illumination through a slit aperture --------------------------
STAGE1 = {
    "duration": 5.3,
    "angles_deg": [-28, 0, 28],      # left, center, right (time split evenly)
    "cone_stroke": GREEN,
    "cone_stroke_width": 5.0,
    "cone_fill": GREEN,
    "cone_fill_opacity": 0.15,
    "outline_color": WHITE,
    "outline_width": 3.0,
    "camera_center": np.array([0.0, 2.6, 0.0]),
    "camera_body_w": 2.4,
    "camera_body_h": 1.3,
    "lens_w": 1.0,
    "lens_h": 0.5,
    "sample_y": 0.4,                 # y of the sample slide
    "sample_w": 3.2,
    "sample_h": 0.12,
    "aperture_y": -0.5,              # y of the small aperture
    "aperture_w": 0.5,
    "aperture_h": 0.12,
    "source_y": -3.2,                # y of the illumination source (below aperture)
    "spread_on_sample": 1.2,         # half-width of the cone where it hits the sample
}

# ---- Stage 2: building a rounded rectangle from frequency-space circles ------
STAGE2 = {
    "duration": 6.5,
    "circle_radius": 0.5,
    "overlap_spacing": 1.0,          # center-to-center distance as a fraction of radius
    "region_half_w": 2.3,            # rounded-rect region the circles must fall inside
    "region_half_h": 1.4,
    "region_corner": 0.7,
    "max_spiral": 800,               # spiral candidates generated before filtering
    "outline_color": BLUE,
    "stroke_width": 3.0,
    "lag_ratio": 0.9,                # how staggered the sequential Create() is
}

# ---- Stage 3: two example images --------------------------------------------
STAGE3 = {
    "duration": 6.0,
    "medical_img": "assets/medical.png",
    "satellite_img": "assets/satellite.png",
    "second_delay": 3.0,             # satellite appears this many seconds in
    "img_width": 5.6,
    "img_height": 5.6,
}

# ---- Stage 4: circles reappear, N x N -> N^2 --------------------------------
STAGE4 = {
    "duration": 5.3,
    "bar_color": WHITE,
    "bar_stroke": 4.0,
    "bar_offset": 0.55,              # gap between circle array and the measuring bar
    "label_font_size": 40,
    "n2_font_size": 64,
    "reappear_time": 0.8,
}

# ---- Stage 5: bars pulse bold / not-bold ------------------------------------
STAGE5 = {
    "duration": 3.5,
    "bar_normal_width": 4.0,
    "bar_bold_width": 11.0,
    "pulse_time": 0.45,              # one bold<->normal transition
}

# ---- Stage 6: shaded circle with a thin vertical slit cutout ----------------
STAGE6 = {
    "duration": 4.4,
    "radius": 1.7,
    "fill_color": BLUE,
    "fill_opacity": 0.6,
    "stroke_color": BLUE_E,
    "stroke_width": 3.0,
    "slit_width": 0.28,
    "slit_height_frac": 0.85,        # slit height as a fraction of the diameter
    "grow_time": 1.2,
}

# ---- Stage 7: linear (translational) slit stack -----------------------------
STAGE7 = {
    "duration": 11.9,
    "intro_time": 1.6,               # slide + cross-fade of the Stage 6 circle
    "n_rects": 15,
    "rect_width": 0.42,
    "rect_height": 3.4,
    "rect_color": BLUE,
    "rect_stroke": 3.0,
    "overlap_frac": 0.5,             # 0.5 => neighbours overlap 50%
    "step_time": 0.5,                # one new rectangle every step_time seconds
    "rect_fade_time": 0.2,
    "image_paths": [
        "assets/stage7_recon_1.png",
        "assets/stage7_recon_2.png",
        "assets/stage7_recon_3.png",
    ],
    "img_width": 5.2,
    "img_height": 5.2,
    "rects_per_image": 5,            # each image lasts this many rectangles
}

# ---- Stage 8: rotational slit stack -----------------------------------------
STAGE8 = {
    "duration": 8.4,
    "intro_time": 1.0,
    "n_rects": 8,
    "angle_step_deg": -22.5,         # negative => clockwise
    "rect_width": 0.42,
    "rect_height": 3.4,
    "rect_color": BLUE,
    "rect_stroke": 3.0,
    "rect_fade_time": 0.25,
    "image_paths": [
        "assets/stage8_recon_1.png",
        "assets/stage8_recon_2.png",
        "assets/stage8_recon_3.png",
    ],
    "img_width": 5.2,
    "img_height": 5.2,
    "image_swap_indices": [0, 3, 6],  # which rectangle index swaps the image
}

# ---- Stage 9: reconstruction pipeline ---------------------------------------
STAGE9 = {
    "duration": 6.5,
    "row_center_x": -1.2,            # rows sit slightly left of center
    "row_y": 1.7,                    # top row y (bottom row mirrors to -row_y)
    "step_delay": 0.5,               # delay between successive left-to-right reveals
    "final_delay": 3.0,              # seconds after rightmost images before final image
    "stack_img_width": 1.3,
    "stack_img_height": 1.1,
    "large_img_width": 2.2,
    "large_img_height": 1.8,
    "final_img_width": 3.0,
    "final_img_height": 3.4,
    "final_center": np.array([4.6, 0.0, 0.0]),
    "stack_a": "assets/stage9_top_a.png",
    "stack_b": "assets/stage9_top_b.png",
    "large_top": "assets/stage9_top_large.png",
    "stack_c": "assets/stage9_bot_a.png",
    "stack_d": "assets/stage9_bot_b.png",
    "large_bot": "assets/stage9_bot_large.png",
    "final_img": "assets/stage9_final.png",
}

# ---- Stage 10: Thank you ----------------------------------------------------
STAGE10 = {
    "duration": 2.2,
    "text": "Thank you!",
    "font_size": 96,
    "color": WHITE,
    "write_time": 1.0,
}


# =============================================================================
# Shared helpers
# =============================================================================
def _apply_background(scene):
    scene.camera.background_color = GLOBAL["background"]


def placeholder_image(path, label, width=3.0, height=None, color=None):
    """Return an ImageMobject if `path` exists, else a labelled gray placeholder box.

    Both branches are plain Mobjects, so callers can `.move_to`, `FadeIn`, etc.
    """
    if height is None:
        height = width * 0.66
    if path and os.path.exists(path):
        img = ImageMobject(path)
        img.set_width(width)
        if img.height > height:
            img.set_height(height)
        return img
    color = color or GLOBAL["gray_placeholder"]
    box = Rectangle(width=width, height=height, stroke_color=color, stroke_width=2.0)
    box.set_fill(color, opacity=0.18)
    txt = Text(label, font_size=22, color=WHITE)
    if txt.width > width * 0.85:
        txt.scale_to_fit_width(width * 0.85)
    txt.move_to(box.get_center())
    return VGroup(box, txt)


def fade_out_all(scene, run_time=None):
    """Fade out everything currently on the scene (used between stages)."""
    run_time = run_time if run_time is not None else GLOBAL["fade_time"]
    movers = [m for m in scene.mobjects]
    if movers:
        scene.play(*[FadeOut(m) for m in movers], run_time=run_time)


def _spiral_coords(n):
    """Integer grid coordinates in a counter-clockwise spiral starting at the
    origin, with the *second* point to the right (matches the Stage 2 spec)."""
    pts = [(0, 0)]
    x = y = 0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, up, left, down => CCW
    d = 0
    seg = 1
    while len(pts) < n:
        for _ in range(2):
            dx, dy = dirs[d % 4]
            for _ in range(seg):
                x += dx
                y += dy
                pts.append((x, y))
                if len(pts) >= n:
                    return pts
            d += 1
        seg += 1
    return pts


def _inside_rounded_rect(x, y, a, b, cr):
    ax, ay = abs(x), abs(y)
    if ax > a or ay > b:
        return False
    if ax <= a - cr or ay <= b - cr:
        return True
    dx = ax - (a - cr)
    dy = ay - (b - cr)
    return dx * dx + dy * dy <= cr * cr


def build_frequency_circles(cfg=STAGE2):
    """VGroup of overlapping blue-outline circles, ordered as a CCW spiral,
    filling a rounded-rectangle region. Reused by Stage 2 (built) and Stage 4."""
    r = cfg["circle_radius"]
    spacing = cfg["overlap_spacing"] * r
    a, b, cr = cfg["region_half_w"], cfg["region_half_h"], cfg["region_corner"]
    circles = VGroup()
    for (ix, iy) in _spiral_coords(cfg["max_spiral"]):
        x, y = ix * spacing, iy * spacing
        if _inside_rounded_rect(x, y, a, b, cr):
            c = Circle(radius=r, color=cfg["outline_color"], stroke_width=cfg["stroke_width"])
            c.set_fill(opacity=0.0)
            c.move_to([x, y, 0.0])
            circles.add(c)
    circles.move_to(ORIGIN)
    return circles


def build_measure_bars(circles, cfg=STAGE4):
    """Width bar (top) and height bar (right) with 'N' labels around `circles`."""
    off = cfg["bar_offset"]
    top = circles.get_top()[1] + off
    right = circles.get_right()[0] + off
    left_x = circles.get_left()[0]
    right_x = circles.get_right()[0]
    bot_y = circles.get_bottom()[1]
    top_y = circles.get_top()[1]

    width_bar = DoubleArrow(
        start=[left_x, top, 0], end=[right_x, top, 0],
        color=cfg["bar_color"], stroke_width=cfg["bar_stroke"], buff=0.0,
        tip_length=0.2,
    )
    height_bar = DoubleArrow(
        start=[right, bot_y, 0], end=[right, top_y, 0],
        color=cfg["bar_color"], stroke_width=cfg["bar_stroke"], buff=0.0,
        tip_length=0.2,
    )
    n_top = Text("N", font_size=cfg["label_font_size"]).next_to(width_bar, UP, buff=0.12)
    n_right = Text("N", font_size=cfg["label_font_size"]).next_to(height_bar, RIGHT, buff=0.12)
    return width_bar, height_bar, n_top, n_right


def build_slit_circle(cfg=STAGE6):
    """Blue-shaded circle with a thin vertical rectangular slit cut out.

    Implemented as a filled circle with a background-colored rectangle overlaid on
    top (reads as a cutout on the configured background). This avoids the skia
    boolean-ops (`Difference`) path that crashes the renderer on some platforms.
    """
    r = cfg["radius"]
    circle = Circle(radius=r)
    circle.set_fill(cfg["fill_color"], opacity=cfg["fill_opacity"])
    circle.set_stroke(cfg["stroke_color"], width=cfg["stroke_width"])
    slit = Rectangle(width=cfg["slit_width"], height=cfg["slit_height_frac"] * 2 * r)
    slit.set_fill(GLOBAL["background"], opacity=1.0)
    slit.set_stroke(cfg["stroke_color"], width=cfg["stroke_width"] * 0.5)
    slit.move_to(circle.get_center())
    return VGroup(circle, slit)


# =============================================================================
# Stage 1 - Illumination
# =============================================================================
def _light_cone(angle_deg, cfg):
    """Green cone polygon from the (angled) source, through the aperture, onto sample."""
    src = np.array([math.tan(math.radians(angle_deg)) * (cfg["aperture_y"] - cfg["source_y"]),
                    cfg["source_y"], 0.0])
    sample_cx = 0.0
    base_l = np.array([sample_cx - cfg["spread_on_sample"], cfg["sample_y"], 0.0])
    base_r = np.array([sample_cx + cfg["spread_on_sample"], cfg["sample_y"], 0.0])
    cone = Polygon(src, base_l, base_r)
    cone.set_stroke(cfg["cone_stroke"], width=cfg["cone_stroke_width"])
    cone.set_fill(cfg["cone_fill"], opacity=cfg["cone_fill_opacity"])
    return cone


def build_optics_diagram(cfg=STAGE1):
    """2D outline: camera (looking down) + sample slide + small aperture."""
    body = Rectangle(width=cfg["camera_body_w"], height=cfg["camera_body_h"],
                     color=cfg["outline_color"], stroke_width=cfg["outline_width"])
    body.move_to(cfg["camera_center"])
    lens = Polygon(
        body.get_corner(DL) + RIGHT * (cfg["camera_body_w"] / 2 - cfg["lens_w"] / 2),
        body.get_corner(DR) + LEFT * (cfg["camera_body_w"] / 2 - cfg["lens_w"] / 2),
        body.get_bottom() + DOWN * cfg["lens_h"] + RIGHT * cfg["lens_w"] / 2,
        body.get_bottom() + DOWN * cfg["lens_h"] + LEFT * cfg["lens_w"] / 2,
        color=cfg["outline_color"], stroke_width=cfg["outline_width"],
    )
    sample = Rectangle(width=cfg["sample_w"], height=cfg["sample_h"],
                       color=cfg["outline_color"], stroke_width=cfg["outline_width"])
    sample.set_fill(cfg["outline_color"], opacity=0.15)
    sample.move_to([0.0, cfg["sample_y"], 0.0])
    aperture = Rectangle(width=cfg["aperture_w"], height=cfg["aperture_h"],
                         color=cfg["outline_color"], stroke_width=cfg["outline_width"])
    aperture.set_fill(cfg["outline_color"], opacity=0.15)
    aperture.move_to([0.0, cfg["aperture_y"], 0.0])
    return VGroup(body, lens, sample, aperture)


def run_stage1(scene, carry=None):
    cfg = STAGE1
    diagram = build_optics_diagram(cfg)
    scene.play(Create(diagram), run_time=0.8)

    per_angle = (cfg["duration"] - 0.8) / len(cfg["angles_deg"])
    cone = None
    for i, ang in enumerate(cfg["angles_deg"]):
        new_cone = _light_cone(ang, cfg)
        if cone is None:
            scene.play(Create(new_cone), run_time=min(0.6, per_angle * 0.5))
        else:
            scene.play(Transform(cone, new_cone), run_time=min(0.6, per_angle * 0.5))
            new_cone = cone
        cone = new_cone
        scene.wait(max(0.05, per_angle - min(0.6, per_angle * 0.5)))

    fade_out_all(scene)
    return None


class Stage1(Scene):
    def construct(self):
        _apply_background(self)
        run_stage1(self)


# =============================================================================
# Stage 2 - Frequency-space circles
# =============================================================================
def run_stage2(scene, carry=None):
    cfg = STAGE2
    circles = build_frequency_circles(cfg)
    scene.play(
        LaggedStart(*[Create(c) for c in circles], lag_ratio=cfg["lag_ratio"]),
        run_time=cfg["duration"],
    )
    fade_out_all(scene)
    return None


class Stage2(Scene):
    def construct(self):
        _apply_background(self)
        run_stage2(self)


# =============================================================================
# Stage 3 - Two example images
# =============================================================================
def run_stage3(scene, carry=None):
    cfg = STAGE3
    medical = placeholder_image(cfg["medical_img"], "Medical image",
                                cfg["img_width"], cfg["img_height"]).move_to(LEFT_HALF)
    scene.play(FadeIn(medical), run_time=GLOBAL["fade_time"])
    scene.wait(max(0.0, cfg["second_delay"] - GLOBAL["fade_time"]))

    satellite = placeholder_image(cfg["satellite_img"], "Satellite image",
                                  cfg["img_width"], cfg["img_height"]).move_to(RIGHT_HALF)
    scene.play(FadeIn(satellite), run_time=GLOBAL["fade_time"])
    scene.wait(max(0.0, cfg["duration"] - cfg["second_delay"] - GLOBAL["fade_time"]))

    fade_out_all(scene)
    return None


class Stage3(Scene):
    def construct(self):
        _apply_background(self)
        run_stage3(self)


# =============================================================================
# Stage 4 - Circles reappear; N x N -> bold N^2
# =============================================================================
def run_stage4(scene, carry=None):
    cfg = STAGE4
    circles = build_frequency_circles(STAGE2)
    scene.play(FadeIn(circles), run_time=cfg["reappear_time"])

    width_bar, height_bar, n_top, n_right = build_measure_bars(circles, cfg)
    scene.play(
        GrowFromCenter(width_bar), GrowFromCenter(height_bar),
        FadeIn(n_top), FadeIn(n_right),
        run_time=1.0,
    )
    scene.wait(max(0.0, cfg["duration"] - cfg["reappear_time"] - 1.0 - 1.8))

    # Bring the two N's together into a bold N^2 above/right of the array.
    center = circles.get_center() + UP * (circles.height / 2 + 1.2)
    n2 = Text("N²", font_size=cfg["n2_font_size"], weight=BOLD).move_to(center)
    n_top_c = n_top.copy()
    n_right_c = n_right.copy()
    scene.add(n_top_c, n_right_c)
    scene.play(n_top_c.animate.move_to(center), n_right_c.animate.move_to(center), run_time=0.9)
    scene.play(ReplacementTransform(VGroup(n_top_c, n_right_c), n2), run_time=0.9)

    return {
        "circles": circles,
        "width_bar": width_bar,
        "height_bar": height_bar,
        "n_top": n_top,
        "n_right": n_right,
        "n2": n2,
    }


class Stage4(Scene):
    def construct(self):
        _apply_background(self)
        carry = run_stage4(self)
        fade_out_all(self)


# =============================================================================
# Stage 5 - Bars pulse bold / not-bold
# =============================================================================
def run_stage5(scene, carry=None):
    cfg = STAGE5
    if carry is None:
        # Standalone: rebuild the Stage 4 end-state statically.
        circles = build_frequency_circles(STAGE2)
        width_bar, height_bar, n_top, n_right = build_measure_bars(circles, STAGE4)
        center = circles.get_center() + UP * (circles.height / 2 + 1.2)
        n2 = Text("N²", font_size=STAGE4["n2_font_size"], weight=BOLD).move_to(center)
        scene.add(circles, width_bar, height_bar, n2)
    else:
        width_bar = carry["width_bar"]
        height_bar = carry["height_bar"]

    bars = VGroup(width_bar, height_bar)
    n_cycles = max(1, int(cfg["duration"] / (2 * cfg["pulse_time"])))
    for _ in range(n_cycles):
        scene.play(bars.animate.set_stroke(width=cfg["bar_bold_width"]),
                   run_time=cfg["pulse_time"])
        scene.play(bars.animate.set_stroke(width=cfg["bar_normal_width"]),
                   run_time=cfg["pulse_time"])

    fade_out_all(scene)
    return None


class Stage5(Scene):
    def construct(self):
        _apply_background(self)
        run_stage5(self)


# =============================================================================
# Stage 6 - Shaded circle with vertical slit cutout
# =============================================================================
def run_stage6(scene, carry=None):
    cfg = STAGE6
    shape = build_slit_circle(cfg)
    scene.play(GrowFromCenter(shape), run_time=cfg["grow_time"])
    scene.wait(max(0.0, cfg["duration"] - cfg["grow_time"]))
    return {"slit_circle": shape}


class Stage6(Scene):
    def construct(self):
        _apply_background(self)
        run_stage6(self)
        fade_out_all(self)


# =============================================================================
# Stage 7 - Linear (translational) slit stack
# =============================================================================
def _vertical_slit_rect(cfg):
    r = Rectangle(width=cfg["rect_width"], height=cfg["rect_height"],
                  color=cfg["rect_color"], stroke_width=cfg["rect_stroke"])
    r.set_fill(opacity=0.0)
    return r


def _linear_offsets(n):
    """Order 0, +1, -1, +2, -2, ... in half-width units (right then left)."""
    order = [0]
    k = 1
    while len(order) < n:
        order.append(k)
        if len(order) < n:
            order.append(-k)
        k += 1
    return order[:n]


def run_stage7(scene, carry=None):
    cfg = STAGE7

    # The Stage 6 circle slides left and cross-fades away, revealing the stack.
    circle = carry["slit_circle"] if carry and "slit_circle" in carry else build_slit_circle(STAGE6)
    if circle not in scene.mobjects:
        scene.add(circle)
    scene.play(circle.animate.move_to(LEFT_HALF), run_time=cfg["intro_time"] * 0.6)
    scene.play(FadeOut(circle), run_time=cfg["intro_time"] * 0.4)

    step = cfg["rect_width"] * cfg["overlap_frac"]
    offsets = _linear_offsets(cfg["n_rects"])
    hold = max(0.0, cfg["step_time"] - cfg["rect_fade_time"])

    current_img = None
    for i, off in enumerate(offsets):
        rect = _vertical_slit_rect(cfg).move_to(LEFT_HALF + RIGHT * off * step)

        if i % cfg["rects_per_image"] == 0:
            idx = min(i // cfg["rects_per_image"], len(cfg["image_paths"]) - 1)
            new_img = placeholder_image(cfg["image_paths"][idx], f"Recon {idx + 1}",
                                        cfg["img_width"], cfg["img_height"]).move_to(RIGHT_HALF)
            if current_img is None:
                scene.play(FadeIn(rect), FadeIn(new_img), run_time=cfg["rect_fade_time"])
            else:
                scene.play(FadeIn(rect), FadeOut(current_img), FadeIn(new_img),
                           run_time=cfg["rect_fade_time"])
            current_img = new_img
        else:
            scene.play(FadeIn(rect), run_time=cfg["rect_fade_time"])
        scene.wait(hold)

    used = cfg["intro_time"] + cfg["n_rects"] * cfg["step_time"]
    scene.wait(max(0.0, cfg["duration"] - used))

    fade_out_all(scene)
    return None


class Stage7(Scene):
    def construct(self):
        _apply_background(self)
        run_stage7(self)


# =============================================================================
# Stage 8 - Rotational slit stack
# =============================================================================
def run_stage8(scene, carry=None):
    cfg = STAGE8
    base = _vertical_slit_rect(cfg).move_to(LEFT_HALF)
    center = base.get_center()

    intro_img = placeholder_image(cfg["image_paths"][0], "Recon 1",
                                  cfg["img_width"], cfg["img_height"]).move_to(RIGHT_HALF)
    scene.play(FadeIn(base), FadeIn(intro_img), run_time=cfg["intro_time"])
    current_img = intro_img

    per_rect = max(0.05, (cfg["duration"] - cfg["intro_time"]) / cfg["n_rects"])
    hold = max(0.0, per_rect - cfg["rect_fade_time"])
    for i in range(1, cfg["n_rects"]):
        rect = base.copy().rotate(math.radians(cfg["angle_step_deg"] * i), about_point=center)
        anims = [FadeIn(rect)]
        if i in cfg["image_swap_indices"]:
            idx = cfg["image_swap_indices"].index(i)
            idx = min(idx, len(cfg["image_paths"]) - 1)
            new_img = placeholder_image(cfg["image_paths"][idx], f"Recon {idx + 1}",
                                        cfg["img_width"], cfg["img_height"]).move_to(RIGHT_HALF)
            anims += [FadeOut(current_img), FadeIn(new_img)]
            current_img = new_img
        scene.play(*anims, run_time=cfg["rect_fade_time"])
        scene.wait(hold)

    fade_out_all(scene)
    return None


class Stage8(Scene):
    def construct(self):
        _apply_background(self)
        run_stage8(self)


# =============================================================================
# Stage 9 - Reconstruction pipeline
# =============================================================================
def _build_pipeline_row(cfg, y, imgs, labels):
    """Ordered list [img1, dots, img2, arrow, large], arranged left-to-right."""
    img1 = placeholder_image(imgs[0], labels[0], cfg["stack_img_width"], cfg["stack_img_height"])
    dots = Text("· · ·", font_size=40)
    img2 = placeholder_image(imgs[1], labels[1], cfg["stack_img_width"], cfg["stack_img_height"])
    arrow = Arrow(LEFT, RIGHT, buff=0.0).scale(0.8)
    large = placeholder_image(imgs[2], labels[2], cfg["large_img_width"], cfg["large_img_height"])

    row = Group(img1, dots, img2, arrow, large).arrange(RIGHT, buff=0.35)
    row.move_to([cfg["row_center_x"], y, 0.0])
    return [img1, dots, img2, arrow, large]


def run_stage9(scene, carry=None):
    cfg = STAGE9
    top = _build_pipeline_row(cfg, cfg["row_y"],
                              [cfg["stack_a"], cfg["stack_b"], cfg["large_top"]],
                              ["Img A", "Img B", "Recon"])
    bot = _build_pipeline_row(cfg, -cfg["row_y"],
                              [cfg["stack_c"], cfg["stack_d"], cfg["large_bot"]],
                              ["Img C", "Img D", "Recon"])

    # Reveal both rows in parallel, left-to-right, with step delays.
    for t_el, b_el in zip(top, bot):
        scene.play(FadeIn(t_el), FadeIn(b_el), run_time=0.25)
        scene.wait(max(0.0, cfg["step_delay"] - 0.25))

    scene.wait(cfg["final_delay"])

    final_img = placeholder_image(cfg["final_img"], "Final",
                                  cfg["final_img_width"], cfg["final_img_height"])
    final_img.move_to(cfg["final_center"])
    scene.play(FadeIn(final_img), run_time=GLOBAL["fade_time"])

    used = 5 * cfg["step_delay"] + cfg["final_delay"] + GLOBAL["fade_time"]
    scene.wait(max(0.0, cfg["duration"] - used))

    fade_out_all(scene)
    return None


class Stage9(Scene):
    def construct(self):
        _apply_background(self)
        run_stage9(self)


# =============================================================================
# Stage 10 - Thank you
# =============================================================================
def run_stage10(scene, carry=None):
    cfg = STAGE10
    text = Text(cfg["text"], font_size=cfg["font_size"], color=cfg["color"])
    scene.play(Write(text), run_time=cfg["write_time"])
    scene.wait(max(0.0, cfg["duration"] - cfg["write_time"]))
    return None


class Stage10(Scene):
    def construct(self):
        _apply_background(self)
        run_stage10(self)


# =============================================================================
# Full - continuous 60 s render (threads `carry` where objects must persist)
# =============================================================================
class Full(Scene):
    def construct(self):
        _apply_background(self)
        run_stage1(self)
        run_stage2(self)
        run_stage3(self)
        carry = run_stage4(self)     # leaves circles + bars + N^2 on screen
        run_stage5(self, carry)      # persists, pulses, then clears
        carry = run_stage6(self)     # leaves the slit circle on screen
        run_stage7(self, carry)      # slides that circle away
        run_stage8(self)
        run_stage9(self)
        run_stage10(self)
