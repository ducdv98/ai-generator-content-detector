# AI-generated content detector (MVP)

This is a learning project about how AI-generated content is watermarked, taken
one medium at a time: text, images, video, then audio. All four steps are built.

## Step 1: text watermarks

Open http://localhost:8000/text. The page runs the two published sampling-time
watermarks with a key you can see:

- **Make a sample.** A small local model (SmolLM2-360M) writes a reply with no
  watermark, a green-list watermark (Kirchenbauer et al.) or a tournament
  watermark (SynthID-Text, the family Gemini and Claude use).
- **Paste text and look for a watermark.** Any text is tokenized and tested
  against the key. The page colours each token by its evidence and charts the
  score building up, where it sits against unmarked text, and (tournament) the
  signal per layer. A third check lists invisible Unicode characters.

It cannot detect the watermarks of Gemini, Claude or ChatGPT. Their keys are
private and none has a public text detector. Why, and how each scheme works, is
in [docs/text-watermarking.md](docs/text-watermarking.md).

```bash
python scripts/download_text_model.py    # once: about 390 MB into models/
pytest -q tests/test_text.py
```

Detection needs only the tokenizer; without the model weights the page still
analyzes pasted text and only generation is switched off.

```
app/text/prf.py         keyed pseudo-random function both schemes share
app/text/greenlist.py   green-list watermark: bias_logits / detect
app/text/tournament.py  tournament watermark: apply / detect
app/text/hidden.py      invisible-character scan
app/text/lm.py          the local model and the sampling loop (shows where the marks act)
app/text/analyze.py     runs all checks over a pasted text
web/text.html           the page
```

## Step 2: image watermarks

Open http://localhost:8000/image. Images are marked in two places at once, and
the page shows both:

- **Make a marked sample.** Adds any of three marks to the sample picture or
  your own: a file label (IPTC "AI-generated" tag), Stable Diffusion's public
  pixel watermark, and a keyed pixel watermark carrying an ID. Shows before,
  after, and the difference amplified until it can be seen.
- **Check an image.** Reads the file's labels (C2PA Content Credentials, IPTC,
  generator metadata) with a map of the file's blocks; decodes the public
  watermark bit by bit, with a map of where in the picture it survives; scores
  the keyed watermark against a key, per bit and per region.
- **What survives being shared.** Applies eleven everyday edits and re-runs all
  three checks on each.

It cannot read learned vendor watermarks such as Google's SynthID (used on
Gemini and OpenAI images); those need the vendors' own checkers. Details and
sources: [docs/image-watermarking.md](docs/image-watermarking.md).

```bash
python scripts/make_sample_image.py      # optional: redraw web/samples/landscape.png
pytest -q tests/test_image.py
```

The public-watermark code is a port of `invisible-watermark`'s "dwtDct" method
written from its source. It has not yet been run against that package; one test
does that cross-check and is skipped until `pip install invisible-watermark
PyWavelets` has been run.

```
app/image/anatomy.py    splits a JPEG / PNG / WebP into picture blocks and label blocks
app/image/public.py     Stable Diffusion's public pixel watermark: embed / votes / decode / detect
app/image/keyed.py      keyed spread-spectrum watermark: evidence per bit and per region
app/image/lab.py        make a sample, check an image, test edits
web/image.html          the page
```

## Step 3: video watermarks

Open http://localhost:8000/video. Needs `ffmpeg` on PATH to write video
(checking a clip works without it).

- **Make a marked clip.** Adds a file label (an IPTC tag in an MP4 box) and a
  keyed watermark in every frame to the built-in sample or your own short clip,
  then compresses it as H.264 at a quality you choose. Plays the result and
  shows one frame three ways: before, in the finished file, and what the mark
  added.
- **Check a clip.** Reads the MP4's boxes and labels; scores every frame on its
  own (with the codec's I/P/B frame types), shows the pooled score building as
  frames are averaged, the ID per bit, and where in the frame the mark survived.
- **What survives being shared.** Ten edits, each compressed again: re-encodes
  at three qualities, half size, half frame rate, trim, crop, mirror, screen
  recording, and one frame saved as a picture.

It cannot read vendor frame watermarks such as SynthID in Veo clips, and it
ignores the sound track. Details, measurements and sources:
[docs/video-watermarking.md](docs/video-watermarking.md).

```bash
pytest -q tests/test_video.py
```

```
app/video/codec.py      frames to H.264 and back (ffmpeg writes, OpenCV reads), frame types via ffprobe
app/video/container.py  MP4 boxes, the label box, label reading
app/video/frames.py     the same keyed mark in every frame; detection by averaging frames
app/video/sample.py     procedural 3-second sample clip
app/video/lab.py        make a clip, check a clip, test edits
web/video.html          the page
```

## Step 4: audio watermarks

Open http://localhost:8000/audio. Needs `ffmpeg` on PATH for anything other
than 16-bit WAV files and for the edit test.

- **Make a marked clip.** Adds a file label (an XMP chunk in the WAV) and a
  keyed sound watermark to the built-in 8-second tune or your own clip, at one
  of three strengths. Plays the clip before, after, and the mark on its own,
  each with its spectrogram.
- **Check a clip.** Reads the file's blocks and labels (WAV, MP3, M4A); scores
  the watermark for the whole clip, for each 1.5-second stretch, as more of the
  clip is heard, per ID bit, and per time and pitch range over the spectrogram.
- **What survives being shared.** Thirteen edits: MP3, AAC and Opus at several
  bitrates, phone-line quality, noise, volume, trimming, tempo, speed, pitch
  and a room echo.

It cannot read vendor marks such as AudioSeal or SynthID. Details, the design
that failed first, measurements and sources:
[docs/audio-watermarking.md](docs/audio-watermarking.md).

```bash
pytest -q tests/test_audio.py
```

```
app/audio/codec.py      any audio file to mono 44.1 kHz samples and back (ffmpeg), WAV writing, spectrograms
app/audio/container.py  WAV chunks, MP3 tags, the label chunk, label reading
app/audio/mark.py       keyed echo-pattern watermark: plan / embed / detect
app/audio/sample.py     synthesised 8-second tune
app/audio/lab.py        make a clip, check a clip, test edits
web/audio.html          the page
```

## Images (earlier detector page)

A small web service that checks an image the way social platforms do, in three
layers, and returns an evidence report instead of a bare yes/no.

| Layer | What it checks | Status |
|---|---|---|
| 1. Provenance | C2PA Content Credentials, IPTC `DigitalSourceType`, Stable Diffusion / ComfyUI / InvokeAI PNG chunks, EXIF `Software`, camera EXIF | Working, tested |
| 2. Watermarks | Stable Diffusion's public DWT-DCT watermark (SD 1.x, 2.x, SDXL) and this service's own keyed watermark | Working, tested |
| 3. Classifier | Any pretrained real-vs-AI image classifier you plug in | Plumbing tested with a dummy model only; **no real model chosen or evaluated yet** |

A second tab, **Sign and verify**, embeds an invisible watermark in an image and
shows which edits it survives.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-extra.txt                # optional, see the file
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Run the tests with `pytest -q`.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `AIGC_WM_KEY` | Secret key for the sign-and-verify watermark. **Set this before deploying**; the default is a public development key. |
| `AIGC_CLASSIFIER_ONNX` | Directory with `model.onnx`, `config.json`, `preprocessor_config.json` (made by `scripts/export_onnx.py`). Needs only onnxruntime. |
| `AIGC_CLASSIFIER_MODEL` | Alternative: a Hugging Face model id or local directory (needs torch + transformers). |
| `AIGC_AI_THRESHOLD`, `AIGC_REAL_THRESHOLD` | Probability cut-offs for "likely AI" / "likely not AI" (default 0.75 / 0.25). |
| `AIGC_MAX_UPLOAD_MB` | Upload limit (default 20). |

## Enabling the classifier (layer 3)

1. On the GPU laptop, pick an open real-vs-AI image-classification model on
   Hugging Face. Check its licence, and measure it on your own set of real and
   generated images before trusting it; published accuracy rarely transfers to
   new generators.
2. `python scripts/export_onnx.py <model-id> models/detector`
3. Copy `models/detector` to the server and set `AIGC_CLASSIFIER_ONNX=models/detector`.

The model's labels are mapped to "AI" or "real" by name (`artificial`, `fake`,
`ai`, `synthetic` ... vs `human`, `real` ...). Unrecognised labels are reported
rather than guessed.

## How the verdict is decided

Strongest evidence wins, in this order:

1. **Declared**: an intact C2PA manifest stating an AI source.
2. **High**: an unsigned AI metadata tag, generator metadata, or a known watermark.
3. **Probabilistic**: the classifier's estimate.
4. Otherwise "No evidence of AI generation", which is not the same as "real".

## The sign-and-verify watermark

`app/signing.py` is a spread-spectrum watermark in the 8x8 block-DCT domain,
written with NumPy and OpenCV only. It carries a 32-bit ID plus a 16-bit
preamble used to decide presence. Measure it with:

```bash
python scripts/robustness.py [your images...]
```

Measured on six sample photos (scikit-learn / scikit-image), default strength:

- Image quality: 41.4 dB PSNR.
- Detected with the ID recovered in 6/6 images after: JPEG quality 90, 70, 50
  and 30; resizing to 50% and 200%; resize 75% + JPEG 70; Gaussian blur; added
  noise; brightness/contrast change.
- **Not detected (0/6) after cropping 10% off each edge.** Cropping, rotation
  and other geometric edits break the block alignment.
- No false alarms: 1,200 unsigned image/wrong-key pairs, highest score 3.3
  against a threshold of 6.

Six images is a smoke test, not a benchmark.

## Known limits

- **C2PA signer trust is not checked.** The service verifies that a manifest is
  intact, but has no trust list configured, so a self-signed manifest is
  reported the same way as one from a known vendor (with a note).
- The Stable Diffusion watermark is fragile; it often decodes only partially
  even on untouched images and is lost on resize. Detection uses a
  chance-probability test (under 1 in 100,000) rather than an exact match.
- Metadata checks see only what survived the upload. Screenshots and most
  messaging apps strip everything layer 1 looks at.
- Video (frame sampling) and audio are not implemented. Text has its own page, see step 1.
- The image service was developed and tested on x86-64 Linux with Python 3.13; the text page on Windows with Python 3.10. Not yet run on ARM.
- No authentication or rate limiting. Put it behind a reverse proxy before
  exposing it publicly.

## Layout

```
app/main.py                 API and static page
app/report.py               evidence model and verdict logic
app/detectors/provenance.py layer 1
app/detectors/watermark.py  layer 2
app/detectors/classifier.py layer 3
app/signing.py              keyed watermark (embed / detect)
app/attacks.py              edits used in the robustness table
web/index.html              front end
scripts/                    robustness measurement, ONNX export
tests/                      pytest suite and C2PA fixtures
```

Test fixtures in `tests/fixtures` are derived from the c2pa-python project's
public test files and signed with its public test certificate.
