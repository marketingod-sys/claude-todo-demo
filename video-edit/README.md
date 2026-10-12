# YT 5 edit pipeline

Edits the "yt 5" walkthrough: burned-in word-highlight captions, section title
cards, B-roll shown inside an iPhone Pro Max on a desk stand, and synthesized
SFX. No background music is added.

| File | What it does |
| --- | --- |
| `edit.py` | Transcribes the A-roll, aligns it to `transcript.txt`, builds assets, renders with ffmpeg |
| `config.json` | Source files, B-roll placement, theme colours, SFX choices |
| `transcript.txt` | Script; `# ` lines are section titles, `[ ... ]` lines mark app playback (left uncaptioned) |
| `make_phone_mockup.py` | Draws the iPhone-on-stand overlay with a transparent screen |
| `make_sfx.py` | Synthesizes whoosh / pop / tap / chime, each peaking at -14 dBFS |

## Run

```sh
pip install faster-whisper            # speech-to-text for caption timing
# put the A-roll and B-roll files in media/ (names as in config.json)
python3 edit.py config.json --dry-run          # assets + build/edit_decisions.md
python3 edit.py config.json --range 6:00-6:30  # quick preview of one part
python3 edit.py config.json                    # full render -> build/final.mp4
```

The first run downloads the Whisper model from huggingface.co and caches the
word timings in `build/words_raw.json`.

## Still needed

- The A-roll itself (it is only on the editor's Mac).
- Network access to `drive.usercontent.google.com` (Drive downloads) and
  `huggingface.co` (speech model) from the cloud environment.
- Read access to the Figma file "Matis Revamp 002" for the real theme colours;
  the colours in `config.json` are placeholders.
- Placement for B-roll 2 (`ai-bible.mp4`).
