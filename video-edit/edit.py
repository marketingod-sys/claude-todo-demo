"""Edit the A-roll: burned-in word-highlight captions, section title cards,
B-roll inside an iPhone-on-stand mockup, and SFX (no music).

    python3 edit.py config.json [--out build/final.mp4] [--range 5:50-6:40]

Steps
  1. Transcribe the A-roll with faster-whisper (word timestamps), cached in
     build/words_raw.json. Edit that file by hand to fix a timing if needed.
  2. Align the ASR words to transcript.txt so captions use the script's
     spelling; audio that is not in the script near a [Play ...] marker is the
     app's own playback and gets no captions.
  3. Build assets (SFX, phone mockup, title cards, scene background), write
     build/captions.ass and build/edit_decisions.md, then render with ffmpeg.
"""
import argparse
import difflib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_phone_mockup  # noqa: E402
import make_sfx  # noqa: E402

PLAY = "<play>"


# ---------------------------------------------------------------- helpers --
def tsec(v):
    if v is None or isinstance(v, (int, float)):
        return v
    parts = [float(p) for p in str(v).split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower().replace("’", "'"))


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ass_color(h, alpha=0):
    r, g, b = hex_rgb(h)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def ass_time(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def font_file(spec):
    return subprocess.check_output(["fc-match", "-f", "%{file}", spec], text=True)


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,width,height,r_frame_rate:format=duration",
        "-of", "json", str(path)], text=True)
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return {"w": v["width"], "h": v["height"], "fps": num / den,
            "fps_str": v["r_frame_rate"], "duration": float(info["format"]["duration"]),
            "audio": has_audio}


# ------------------------------------------------------------ transcribe --
def transcribe(path, model_name):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True,
                                   language="en", beam_size=5)
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": round(w.start, 3),
                          "end": round(w.end, 3)})
        print(f"\r  transcribed to {seg.end:7.1f}s", end="", flush=True)
    print()
    return words


# ----------------------------------------------------------------- align --
def parse_transcript(path):
    """Tokens for every script word, plus section headings and play markers."""
    tokens, sections = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            sections.append({"title": line.lstrip("# ").strip(), "token": len(tokens)})
            continue
        if line.startswith("["):
            tokens.append({"text": PLAY, "norm": PLAY})
            continue
        for w in line.split():
            # "four-second" may be heard as two words; match on the parts
            parts = [p for p in w.split("-") if p]
            for i, p in enumerate(parts):
                shown = p + ("-" if i < len(parts) - 1 else "")
                tokens.append({"text": shown, "norm": norm(p)})
    return tokens, sections


def align(tokens, asr):
    """Return caption words [{text,start,end,ref}] and per-token times."""
    ref = [t["norm"] for t in tokens]
    hyp = [norm(w["word"]) for w in asr]
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    times = [None] * len(tokens)
    out = []
    ops = sm.get_opcodes()
    # ASR words before/after the scripted part are captioned straight from the ASR
    first_hyp_match = next((j1 for tag, i1, i2, j1, j2 in ops if tag == "equal"), 0)
    last_hyp_match = max((j2 for tag, i1, i2, j1, j2 in ops if tag == "equal"), default=0)
    for tag, i1, i2, j1, j2 in ops:
        ref_words = [i for i in range(i1, i2) if tokens[i]["norm"] != PLAY]
        has_play = any(tokens[i]["norm"] == PLAY for i in range(i1, i2))
        near_play = has_play or (i1 > 0 and tokens[i1 - 1]["norm"] == PLAY)
        if tag == "equal":
            for a, b in zip(range(i1, i2), range(j1, j2)):
                times[a] = (asr[b]["start"], asr[b]["end"])
                out.append({"text": tokens[a]["text"], "start": asr[b]["start"],
                            "end": asr[b]["end"], "ref": a})
            continue
        hyp_span = asr[j1:j2]
        outside = j2 <= first_hyp_match or j1 >= last_hyp_match
        if tag == "insert" or (tag == "replace" and not ref_words):
            # audio not in the script: app playback near a marker, or speech
            # outside the scripted part (caption it from the ASR)
            if outside or not near_play:
                out += [{"text": w["word"], "start": w["start"], "end": w["end"],
                         "ref": None} for w in hyp_span]
            continue
        if tag == "delete":
            continue  # script words with no audio (e.g. unspoken headings)
        if has_play:
            # script words either side of a playback marker: map them onto the
            # first / last heard words and leave the playback audio uncaptioned
            play_at = [i for i in range(i1, i2) if tokens[i]["norm"] == PLAY]
            pre = [i for i in ref_words if i < play_at[0]]
            post = [i for i in ref_words if i > play_at[-1]]
            groups = [(pre, hyp_span[:len(pre)]), (post, hyp_span[len(hyp_span) - len(post):])]
        else:
            groups = [(ref_words, hyp_span)]
        for idx, span in groups:
            if not idx or not span:
                continue
            # replace: spread the script words over the heard span
            t0, t1 = span[0]["start"], span[-1]["end"]
            total = sum(max(1, len(tokens[i]["text"])) for i in idx)
            cur = t0
            for i in idx:
                d = (t1 - t0) * max(1, len(tokens[i]["text"])) / total
                times[i] = (round(cur, 3), round(cur + d, 3))
                out.append({"text": tokens[i]["text"], "start": times[i][0],
                            "end": times[i][1], "ref": i})
                cur += d
    out.sort(key=lambda w: w["start"])
    return out, times


def section_times(sections, tokens, times):
    res = []
    for s in sections:
        t = next((times[i][0] for i in range(s["token"], len(tokens)) if times[i]), None)
        res.append({"title": s["title"], "time": t})
    return res


def find_phrase(phrase, tokens, times, near=None):
    target = [norm(p) for p in re.split(r"[\s-]+", phrase) if norm(p)]
    hits = []
    for i in range(len(tokens) - len(target) + 1):
        if [tokens[i + k]["norm"] for k in range(len(target))] == target:
            span = [times[i + k] for k in range(len(target)) if times[i + k]]
            if span:
                hits.append((span[0][0], span[-1][1]))
    if not hits:
        raise SystemExit(f"anchor phrase not found in aligned transcript: {phrase!r}")
    if near is not None:
        hits.sort(key=lambda h: abs(h[0] - near))
    return hits[0]


# -------------------------------------------------------------- captions --
def chunk_words(words, max_words=3, max_chars=22, max_gap=0.4):
    chunks, cur = [], []
    for w in words:
        if cur:
            text_len = sum(len(x["text"]) + 1 for x in cur) + len(w["text"])
            gap = w["start"] - cur[-1]["end"]
            if (len(cur) >= max_words or text_len > max_chars or gap > max_gap
                    or re.search(r"[.!?,:;]$", cur[-1]["text"])):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def write_ass(chunks, path, w, h, theme):
    k = h / 1080
    size = round(theme.get("caption_size", 64) * k)
    style = ",".join(str(x) for x in [
        "Cap", theme.get("caption_font", "Inter ExtraBold"), size,
        ass_color(theme["caption_text"]), ass_color(theme["caption_highlight"]),
        ass_color(theme["caption_outline"]), ass_color("#000000", 0x80),
        0, 0, 0, 0, 100, 100, 0, 0, 1, round(4.5 * k), round(2 * k), 2,
        60, 60, round(theme.get("caption_margin_v", 96) * k), 1])
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "ScaledBorderAndShadow: yes", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {style}", "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hi = ass_color(theme["caption_highlight"])
    base = ass_color(theme["caption_text"])
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else ch[-1]["end"] + 1
        chunk_end = min(ch[-1]["end"] + 0.35, nxt)
        for wi, wd in enumerate(ch):
            start = wd["start"]
            end = ch[wi + 1]["start"] if wi + 1 < len(ch) else chunk_end
            parts = []
            for j, x in enumerate(ch):
                t = x["text"].replace("{", "(").replace("}", ")")
                parts.append(f"{{\\c{hi}}}{t}{{\\c{base}}}" if j == wi else t)
            text = parts[0]
            for x, part in zip(ch, parts[1:]):  # no space after a split hyphen
                text += ("" if x["text"].endswith("-") else " ") + part
            if wi == 0:
                text = "{\\fscx86\\fscy86\\t(0,110,\\fscx100\\fscy100)}" + text
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{text}")
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- assets --
def title_card(title, index, path, h, theme):
    k = h / 1080
    bold = ImageFont.truetype(font_file("Inter:style=Bold"), round(44 * k))
    small = ImageFont.truetype(font_file("Inter:style=SemiBold"), round(20 * k))
    kicker = f"PART {index:02d}"
    tw = bold.getbbox(title)[2]
    pad_x, pad_y, bar = round(34 * k), round(22 * k), round(8 * k)
    box_w = tw + 2 * pad_x + bar
    box_h = round(112 * k)
    img = Image.new("RGBA", (box_w + 4, box_h + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, box_w, box_h), radius=round(18 * k),
                        fill=hex_rgb(theme["title_bg"]) + (235,))
    d.rounded_rectangle((0, 0, bar * 2, box_h), radius=round(18 * k),
                        fill=hex_rgb(theme["caption_highlight"]) + (255,))
    d.rectangle((bar, 0, bar * 2, box_h), fill=hex_rgb(theme["title_bg"]) + (235,))
    x = bar + pad_x
    d.text((x, pad_y), " ".join(kicker), font=small,
           fill=hex_rgb(theme["caption_highlight"]) + (255,))
    d.text((x, pad_y + round(28 * k)), title, font=bold,
           fill=hex_rgb(theme["title_text"]) + (255,))
    img.save(path)


def scene_background(path, w, h, theme):
    top, bottom = hex_rgb(theme["bg_top"]), hex_rgb(theme["bg_bottom"])
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / (h - 1)
        d.line((0, y, w, y), fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse((w * 0.3, h * 0.05, w * 0.7, h * 0.85), fill=70)
    from PIL import ImageFilter
    glow = glow.filter(ImageFilter.GaussianBlur(h * 0.12))
    img.paste((255, 255, 255), (0, 0), glow)
    img.save(path)


# ---------------------------------------------------------------- render --
def build_command(cfg, info, build, captions, titles, brolls, sfx_events, out, rng):
    w, h, fps = info["w"], info["h"], info["fps_str"]
    k = h / 1080
    inputs = ["-i", str(cfg["aroll"])]
    fc = []
    n = 1
    v = "0:v"
    for i, br in enumerate(brolls):
        d = br["duration"]
        inputs += ["-ss", f"{br['src_start']:.3f}", "-t", f"{d:.3f}", "-i", str(br["file"]),
                   "-loop", "1", "-framerate", fps, "-t", f"{d:.3f}", "-i", str(br["bg"]),
                   "-loop", "1", "-framerate", fps, "-t", f"{d:.3f}", "-i", str(br["mockup"])]
        src, bg, mk = n, n + 1, n + 2
        n += 3
        r = br["rect"]
        dy = (f"if(lt(t,0.45),{140 * k:.1f}*pow(1-t/0.45,3),"
              f"if(gt(t,{d - 0.4:.3f}),{140 * k:.1f}*pow((t-{d - 0.4:.3f})/0.4,3),0))")
        fc += [
            f"[{src}:v]fps={fps},scale={r['w']}:{r['h']}:force_original_aspect_ratio=increase,"
            f"crop={r['w']}:{r['h']},setsar=1[b{i}]",
            f"[{bg}:v][b{i}]overlay=x={r['x']}:y='{r['y']}+{dy}':eval=frame[bb{i}]",
            f"[bb{i}][{mk}:v]overlay=x=0:y='{dy}':eval=frame,format=yuva420p,"
            f"fade=t=in:st=0:d=0.3:alpha=1,fade=t=out:st={d - 0.3:.3f}:d=0.3:alpha=1,"
            f"setpts=PTS-STARTPTS+{br['start']:.3f}/TB[s{i}]",
            f"[{v}][s{i}]overlay=eof_action=pass[vb{i}]",
        ]
        v = f"vb{i}"
    for i, t in enumerate(titles):
        d = t["duration"]
        inputs += ["-loop", "1", "-framerate", fps, "-t", f"{d:.3f}", "-i", str(t["png"])]
        x0, y0 = round(72 * k), round(64 * k)
        fc += [
            f"[{n}:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1,"
            f"fade=t=out:st={d - 0.3:.3f}:d=0.3:alpha=1,"
            f"setpts=PTS-STARTPTS+{t['time']:.3f}/TB[t{i}]",
            f"[{v}][t{i}]overlay=x='{x0}-{70 * k:.1f}*pow(1-min(max(t-{t['time']:.3f},0)/0.4,1),3)'"
            f":y={y0}:eval=frame:eof_action=pass[vt{i}]",
        ]
        n += 1
        v = f"vt{i}"
    ass = str(captions).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    fc.append(f"[{v}]ass='{ass}',format=yuv420p[vout]")

    mix = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a0]"]
    labels = ["[a0]"]
    for i, ev in enumerate(sfx_events):
        inputs += ["-i", str(ev["file"])]
        ms = max(0, round(ev["time"] * 1000))
        gain = min(0.0, ev.get("gain_db", 0.0))  # never louder than the -14 dBFS master
        mix.append(f"[{n}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                   f"volume={gain}dB,adelay={ms}|{ms}[x{i}]")
        labels.append(f"[x{i}]")
        n += 1
    mix.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first"
               ":dropout_transition=0[aout]")
    fc += mix

    script = build / "filtergraph.txt"
    script.write_text(";\n".join(fc))
    cmd = ["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex_script", str(script),
           "-map", "[vout]", "-map", "[aout]"]
    if rng:
        cmd += ["-ss", f"{rng[0]:.3f}", "-to", f"{rng[1]:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-r", fps, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--range", default=None, help="render only START-END, e.g. 5:50-6:40")
    ap.add_argument("--dry-run", action="store_true", help="build assets, skip the render")
    a = ap.parse_args()

    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    root = cfg_path.parent
    for key in ("aroll", "transcript"):
        cfg[key] = (root / cfg[key]).resolve()
    build = (root / cfg.get("build_dir", "build")).resolve()
    build.mkdir(parents=True, exist_ok=True)
    theme = cfg["theme"]
    info = probe(cfg["aroll"])
    print(f"A-roll: {info['w']}x{info['h']} @ {info['fps']:.3f} fps, {info['duration']:.1f}s")

    raw = build / "words_raw.json"
    if not raw.exists():
        print("transcribing (cached in build/words_raw.json) ...")
        raw.write_text(json.dumps(transcribe(cfg["aroll"], cfg.get("whisper_model", "small.en")),
                                  indent=1))
    asr = json.loads(raw.read_text())
    tokens, sections = parse_transcript(cfg["transcript"])
    words, times = align(tokens, asr)
    (build / "words_aligned.json").write_text(json.dumps(words, indent=1))
    matched = sum(1 for t in times if t) / max(1, sum(1 for t in tokens if t["norm"] != PLAY))
    print(f"script words matched to audio: {matched:.0%}")

    # captions
    chunks = chunk_words(words, **cfg.get("caption_chunking", {}))
    captions = build / "captions.ass"
    write_ass(chunks, captions, info["w"], info["h"], theme)

    # SFX pack
    sfx_dir = build / "sfx"
    if not (sfx_dir / "whoosh_in.wav").exists():
        make_sfx.main(sfx_dir)
    sfx_cfg = cfg.get("sfx", {})
    sfx_events = []

    # section title cards
    titles = []
    for i, s in enumerate(section_times(sections, tokens, times), 1):
        if s["time"] is None:
            print(f"  ! section {s['title']!r}: no matching audio, skipped")
            continue
        png = build / f"title_{i:02d}.png"
        title_card(s["title"], i, png, info["h"], theme)
        t0 = max(0.0, s["time"] - 0.15)
        titles.append({"title": s["title"], "time": t0, "duration": 3.2, "png": png})
        if sfx_cfg.get("title"):
            sfx_events.append({"file": sfx_dir / sfx_cfg["title"], "time": t0,
                               "gain_db": sfx_cfg.get("title_gain_db", 0), "why": s["title"]})

    # B-roll scenes
    brolls = []
    bg = build / "scene_bg.png"
    scene_background(bg, info["w"], info["h"], theme)
    for i, br in enumerate(cfg.get("broll", [])):
        if br.get("skip"):
            continue
        f = (root / br["file"]).resolve()
        if br.get("at") is not None:
            start = tsec(br["at"])
        else:
            p0, _ = find_phrase(br["anchor"], tokens, times, tsec(br.get("near")))
            start = p0 - br.get("lead", 0.4)
        orient = br.get("orientation", "portrait")
        mk_png = build / f"iphone_stand_{orient}_{info['h']}.png"
        mk_json = mk_png.with_suffix(".json")
        if not mk_png.exists():
            img, rect = make_phone_mockup.build(orient, theme.get("phone_finish", "natural"),
                                                info["h"])
            img.save(mk_png)
            mk_json.write_text(json.dumps(rect))
        brolls.append({"file": f, "src_start": tsec(br.get("src_start", 0)), "start": start,
                       "duration": br["duration"], "bg": bg, "mockup": mk_png,
                       "rect": json.loads(mk_json.read_text()), "anchor": br.get("anchor")})
        if sfx_cfg.get("broll_in"):
            sfx_events.append({"file": sfx_dir / sfx_cfg["broll_in"], "time": start - 0.15,
                               "gain_db": sfx_cfg.get("broll_gain_db", 0), "why": "B-roll in"})
        if sfx_cfg.get("broll_out"):
            sfx_events.append({"file": sfx_dir / sfx_cfg["broll_out"],
                               "time": start + br["duration"] - 0.45,
                               "gain_db": sfx_cfg.get("broll_gain_db", 0), "why": "B-roll out"})

    # human-readable edit decision list
    def mmss(t):
        return f"{int(t // 60)}:{t % 60:05.2f}"
    edl = ["# Edit decisions", "", f"A-roll: {cfg['aroll'].name} "
           f"({info['w']}x{info['h']}, {mmss(info['duration'])})",
           f"Captions: {len(chunks)} caption groups, {len(words)} words "
           f"({matched:.0%} of script words matched to audio)", "", "## Section titles"]
    edl += [f"- {mmss(t['time'])} — {t['title']}" for t in titles]
    edl += ["", "## B-roll (iPhone on stand)"]
    edl += [f"- {mmss(b['start'])}–{mmss(b['start'] + b['duration'])} — {b['file'].name} "
            f"from {mmss(b['src_start'])} (anchor: {b['anchor']!r})" for b in brolls]
    edl += ["", "## SFX (all peak at or below -14 dBFS)"]
    edl += [f"- {mmss(e['time'])} — {e['file'].name} ({e['why']})" for e in sfx_events]
    (build / "edit_decisions.md").write_text("\n".join(edl) + "\n")
    print("\n".join(edl))

    rng = None
    if a.range:
        s0, s1 = a.range.split("-")
        rng = (tsec(s0), tsec(s1))
    out = Path(a.out).resolve() if a.out else build / "final.mp4"
    cmd = build_command(cfg, info, build, captions, titles, brolls, sfx_events, out, rng)
    (build / "render_cmd.sh").write_text(shlex.join(cmd) + "\n")
    if a.dry_run:
        print("dry run: command in build/render_cmd.sh")
        return
    subprocess.run(cmd, check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
