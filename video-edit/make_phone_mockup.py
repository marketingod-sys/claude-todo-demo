"""Render an iPhone Pro Max on a desk stand as a 1920x1080 RGBA overlay.

The screen area is fully transparent, so the B-roll goes underneath the PNG.
The screen rectangle (in 1920x1080 coordinates) is written next to it as JSON.

    python3 make_phone_mockup.py OUT_DIR [--orientation portrait|landscape]
                                 [--finish black|natural|desert|white] [--height 1080]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
SS = 3  # supersampling factor

FINISHES = {  # (edge colour, highlight colour) for the titanium band
    "black": ((38, 38, 40), (120, 120, 124)),
    "natural": ((128, 124, 117), (222, 219, 212)),
    "desert": ((150, 128, 108), (230, 212, 192)),
    "white": ((176, 176, 172), (246, 246, 242)),
}
STAND = ((96, 99, 104), (214, 217, 222))  # anodised aluminium


def s(v):
    return int(round(v * SS))


def gradient(size, c0, c1, horizontal=True, mid_peak=True):
    w, h = size
    n = w if horizontal else h
    t = np.linspace(0, 1, n)
    if mid_peak:
        t = 1 - np.abs(t - 0.5) * 2
        t = t ** 0.6
    c = np.array(c0)[None, :] * (1 - t[:, None]) + np.array(c1)[None, :] * t[:, None]
    if horizontal:
        arr = np.repeat(c[None, :, :], h, axis=0)
    else:
        arr = np.repeat(c[:, None, :], w, axis=1)
    rgba = np.concatenate([arr, np.full((h, w, 1), 255)], axis=2).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def rr_mask(size, box, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=radius, fill=255)
    return m


def paste_shape(canvas, img, mask, xy=(0, 0)):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(img, xy, mask)
    canvas.alpha_composite(layer)


def phone_layer(pw, ph, finish):
    """Portrait phone on its own canvas (with margin for side buttons)."""
    margin = s(6)
    cw, ch = s(pw) + 2 * margin, s(ph) + 2 * margin
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    edge, hi = FINISHES[finish]
    x0, y0, x1, y1 = margin, margin, margin + s(pw), margin + s(ph)
    r_out = s(pw * 0.155)

    # side buttons (drawn first so the body overlaps their inner half)
    btn = tuple(int(v * 0.85) for v in hi)
    def button(side, top, length):
        bx = x0 - s(3.2) if side == "L" else x1 - s(2)
        d.rounded_rectangle((bx, y0 + s(top), bx + s(5.2), y0 + s(top + length)),
                            radius=s(2), fill=btn + (255,))
    button("L", ph * 0.155, ph * 0.045)   # action button
    button("L", ph * 0.235, ph * 0.075)   # volume up
    button("L", ph * 0.325, ph * 0.075)   # volume down
    button("R", ph * 0.255, ph * 0.11)    # side button
    button("R", ph * 0.60, ph * 0.075)    # camera control

    # titanium band
    band = gradient((cw, ch), edge, hi)
    paste_shape(img, band, rr_mask((cw, ch), (x0, y0, x1, y1), r_out))
    # thin dark line separating band from glass
    t_band = s(pw * 0.013)
    d.rounded_rectangle((x0 + t_band, y0 + t_band, x1 - t_band, y1 - t_band),
                        radius=r_out - t_band, fill=(14, 14, 16, 255))
    # black bezel
    t_bez = t_band + s(pw * 0.006)
    d.rounded_rectangle((x0 + t_bez, y0 + t_bez, x1 - t_bez, y1 - t_bez),
                        radius=r_out - t_bez, fill=(4, 4, 6, 255))
    # screen: punch a transparent hole
    t_scr = t_band + s(pw * 0.032)
    scr = (x0 + t_scr, y0 + t_scr, x1 - t_scr, y1 - t_scr)
    r_scr = r_out - t_scr
    hole = rr_mask((cw, ch), scr, r_scr)
    alpha = np.array(img.getchannel("A"))
    alpha[np.array(hole) > 0] = 0
    # keep the anti-aliased screen edge soft
    alpha = np.minimum(alpha, 255 - np.array(hole))
    img.putalpha(Image.fromarray(alpha.astype(np.uint8)))

    # Dynamic Island
    di_w, di_h = s(pw * 0.32), s(pw * 0.093)
    cx = (x0 + x1) // 2
    di_top = scr[1] + s(pw * 0.035)
    d.rounded_rectangle((cx - di_w // 2, di_top, cx + di_w // 2, di_top + di_h),
                        radius=di_h // 2, fill=(0, 0, 0, 255))

    # subtle glass glare across the upper-left of the screen
    glare = Image.new("L", (cw, ch), 0)
    gd = ImageDraw.Draw(glare)
    gd.polygon([(scr[0], scr[1]), (scr[0] + (scr[2] - scr[0]) * 0.75, scr[1]),
                (scr[0], scr[1] + (scr[3] - scr[1]) * 0.42)], fill=14)
    glare = Image.composite(glare, Image.new("L", (cw, ch), 0), hole)
    glare = glare.filter(ImageFilter.GaussianBlur(s(18)))
    white = Image.new("RGBA", (cw, ch), (255, 255, 255, 0))
    white.putalpha(glare)
    img.alpha_composite(white)

    screen = {"x": scr[0], "y": scr[1], "w": scr[2] - scr[0], "h": scr[3] - scr[1],
              "radius": r_scr}
    return img, screen, margin


def build(orientation, finish, out_h=1080):
    canvas = Image.new("RGBA", (s(W), s(H)), (0, 0, 0, 0))
    portrait = orientation == "portrait"
    if portrait:
        ph = 740
        pw = ph / 2.10
        top = 58
    else:
        pw_long = 1080
        pw, ph = pw_long / 2.10, pw_long  # drawn portrait, rotated later
        top = 205
    phone, scr, margin = phone_layer(pw, ph, finish)
    if not portrait:
        phone = phone.rotate(90, expand=True, resample=Image.BICUBIC)
        cw, ch = phone.size
        # rotate the screen rect 90deg CCW within the phone canvas
        scr = {"x": scr["y"], "y": ch - scr["x"] - scr["w"],
               "w": scr["h"], "h": scr["w"], "radius": scr["radius"]}
    vis_w = phone.size[0] - 2 * margin
    vis_h = phone.size[1] - 2 * margin
    px = s(W / 2) - phone.size[0] // 2
    py = s(top) - margin
    bottom = py + margin + vis_h  # phone's bottom edge (SS px)

    # ---- stand -------------------------------------------------------------
    cx = s(W / 2)
    base_w = s(560 if portrait else 760)
    base_h = s(64)
    base_cy = bottom + s(135 if portrait else 165)
    # floor contact shadow
    sh = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(sh).ellipse((cx - base_w * 0.55, base_cy - base_h * 0.2,
                                cx + base_w * 0.55, base_cy + base_h * 1.05), fill=120)
    sh = sh.filter(ImageFilter.GaussianBlur(s(22)))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow.putalpha(sh)
    canvas.alpha_composite(shadow)
    dark, light = STAND
    # base: side wall then top face
    side = gradient(canvas.size, tuple(int(c * 0.75) for c in dark), dark)
    m = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(m).ellipse((cx - base_w / 2, base_cy - base_h / 2 + s(16),
                               cx + base_w / 2, base_cy + base_h / 2 + s(16)), fill=255)
    paste_shape(canvas, side, m)
    top_face = gradient(canvas.size, dark, light)
    m = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(m).ellipse((cx - base_w / 2, base_cy - base_h / 2,
                               cx + base_w / 2, base_cy + base_h / 2), fill=255)
    paste_shape(canvas, top_face, m)
    # neck
    neck_w = s(84)
    neck = gradient(canvas.size, dark, light)
    m = rr_mask(canvas.size, (cx - neck_w / 2, bottom - s(40), cx + neck_w / 2, base_cy),
                s(10))
    paste_shape(canvas, neck, m)
    # neck shading where it meets the base
    sh2 = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(sh2).ellipse((cx - neck_w * 0.62, base_cy - s(7),
                                 cx + neck_w * 0.62, base_cy + s(6)), fill=55)
    sh2 = sh2.filter(ImageFilter.GaussianBlur(s(3)))
    s2 = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    s2.putalpha(sh2)
    canvas.alpha_composite(s2)

    # ---- phone -------------------------------------------------------------
    # soft drop shadow behind the phone
    ps = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(ps).rounded_rectangle(
        (px + margin + s(10), py + margin + s(18),
         px + margin + vis_w - s(10), py + margin + vis_h + s(10)), radius=s(60), fill=110)
    ps = ps.filter(ImageFilter.GaussianBlur(s(26)))
    psl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    psl.putalpha(ps)
    canvas.alpha_composite(psl)
    # the shadow must not show through the transparent screen
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(phone, (px, py))
    scr_box = (px + scr["x"], py + scr["y"], px + scr["x"] + scr["w"],
               py + scr["y"] + scr["h"])
    hole = rr_mask(canvas.size, scr_box, scr["radius"])
    a = np.array(canvas.getchannel("A")).astype(np.int32)
    a[np.array(hole) > 0] = 0
    canvas.putalpha(Image.fromarray(a.astype(np.uint8)))
    canvas.alpha_composite(layer)

    # ---- cradle lip in front of the phone's bottom edge ----------------------
    lip_w = s(vis_w / SS * 0.78)
    lip = gradient(canvas.size, dark, light, horizontal=False)
    m = rr_mask(canvas.size, (cx - lip_w / 2, bottom - s(9), cx + lip_w / 2, bottom + s(22)),
                s(8))
    paste_shape(canvas, lip, m)
    hl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle((cx - lip_w / 2 + s(6), bottom - s(8),
                                          cx + lip_w / 2 - s(6), bottom - s(6)),
                                         radius=s(1), fill=(255, 255, 255, 90))
    canvas.alpha_composite(hl)

    k = out_h / H
    out = canvas.resize((round(W * k), out_h), Image.LANCZOS)
    rect = {key: round(v / SS * k) for key, v in
            {"x": scr_box[0], "y": scr_box[1], "w": scr["w"], "h": scr["h"],
             "radius": scr["radius"]}.items()}
    return out, rect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="build/mockup")
    ap.add_argument("--orientation", default="portrait", choices=["portrait", "landscape"])
    ap.add_argument("--finish", default="natural", choices=sorted(FINISHES))
    ap.add_argument("--height", type=int, default=1080, help="output frame height")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img, rect = build(a.orientation, a.finish, a.height)
    stem = f"iphone_stand_{a.orientation}"
    img.save(out / f"{stem}.png")
    (out / f"{stem}.json").write_text(json.dumps(rect, indent=2))
    print(out / f"{stem}.png", rect)


if __name__ == "__main__":
    main()
