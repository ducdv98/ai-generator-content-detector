# How AI video gets marked (state of play, November 2026)

Research notes for step three of this project. The code is in `app/video/`,
and the page at `/video` shows it working. Earlier steps:
[text](text-watermarking.md), [images](image-watermarking.md).

## What changes when the picture moves

A video is a stack of images plus a sound track inside a container file, so
every idea from the image step carries over. Three things are new.

1. **Compression is far harsher.** A still image is compressed once, mildly.
   Video is compressed by a factor of a hundred or more, and again on every
   upload. H.264 and its successors store a few frames whole (I-frames) and the
   rest as "this block moved from there, plus a correction" (P- and B-frames),
   and they round the corrections off hard. Faint fine detail that does not
   move with the picture is the first thing rounded away, and that describes an
   invisible watermark exactly.
2. **There is a time axis to attack.** Frames can be dropped, the clip trimmed,
   the speed or frame rate changed. A mark that depends on frame order or
   timing has to survive all of that.
3. **There are many frames.** That is a cost (every frame must be marked, fast)
   and an opportunity (the detector can combine frames).

## The channels, simple to complex

| Channel | What it is | Survives a re-upload? |
|---|---|---|
| **Visible badge** | A logo burned into the picture, sometimes moving around. For people, not detectors. | Yes, until cropped or painted out. |
| **File label** | C2PA Content Credentials or an IPTC tag in a box of the MP4, beside the frames. `app/video/container.py` | No. Re-encoding writes a new file from the frames alone. |
| **Frame watermark** | An invisible pixel watermark in the frames. `app/video/frames.py` | Depends on how hard the compression is. |
| **Sound-track watermark** | An inaudible mark in the audio (for example Google's SynthID for audio, Meta's AudioSeal). | Not covered by this project. |

### File labels in an MP4

An MP4 is a row of boxes: `ftyp` (file type), `moov` (the index: where each
frame is, timing, tool names), `mdat` (the compressed frames and sound) and
optional `uuid` extension boxes. C2PA manifests and XMP metadata each live in a
`uuid` box. The page shows the boxes to scale; the label added here is 345
bytes beside tens of kilobytes of frames. The `moov` box also carries a tool
tag (the sample says `Lavf…`, ffmpeg's muxer), which names the encoder, not
what made the pictures.

### Frame watermarks: the design choice that matters

The scheme here is the keyed spread-spectrum image watermark from step two,
applied to every frame with the **same** pattern. Repeating one pattern means:

- **No timing to recover.** Any frames that survive carry the whole mark.
  Halving the frame rate or keeping only the last third of the clip changed
  little in the measurements below, and one saved frame is still detectable.
- **Frames can be pooled.** The detector averages the frames' DCT values and
  runs the image detector once on the average. What differs between frames
  (motion, grain, the codec's rounding errors) cancels; the pattern stays.

Pooling must be done by averaging first. Adding up per-frame scores looks
equivalent and is wrong: neighbouring frames are nearly the same picture, so
their scores are not independent, and a still clip would turn one frame's luck
into a confident "detection". `tests/test_video.py` checks both directions:
thirty copies of one frame score exactly what one frame does, and a weak mark
under heavy per-frame noise scores well above any single frame once pooled.

The honest lesson is that the gain from pooling is modest on real footage. In
the text step every token was a fresh piece of evidence; here 72 frames pooled
scored only 1.13 times what the first frame scored alone.

### What vendors deploy

- **Google (Veo).** SynthID is applied to the frames; the SynthID-Image paper
  counts "images and video frames" together, over ten billion marked. Checked
  with Google's own tools.
- **Meta (Video Seal, open research).** A trained embedder and extractor. Its
  training simulates video codecs so the mark survives them. To save computing
  time it marks one frame in every k and copies that pattern onto the frames
  between ("temporal propagation"); the paper notes that leaving frames
  unmarked instead would hurt robustness to temporal edits and compression.
  Whether Meta runs it in production is not confirmed by what I read.
- **OpenAI (Sora).** Reported to carry C2PA credentials plus a visible badge.
  A summary of OpenAI's help page lists Sora as discontinued; check before
  relying on either statement.
- **Research direction.** Marks built into the video diffusion model itself
  (for example VideoShield), which can also point to where a clip was altered.

## So can an uploaded video be checked?

| Question | Answer |
|---|---|
| Does the file carry a label saying it is AI-made? | Yes, if the file is the original. Almost never after it has been through a platform. |
| Does it carry a keyed frame watermark made here? | Yes, with the key, unless compression was heavy or the frame was cropped or mirrored. |
| Does it carry SynthID or another vendor's frame watermark? | Not offline; the decoder is not published. Google runs a public checker for its own output. |
| Is the sound track marked? | Not checked here. |
| Is it AI-made at all? | Not a watermark question. No mark found means "no evidence", never "real". |

## Measurements

The built-in sample clip (72 frames, 480 × 320, 3 seconds), key `demo-key`,
watermark strength 9 (step two used 5 for stills).

How much survives one compression, by ffmpeg quality setting (CRF, lower is
better quality):

| CRF | File size | Pooled score | Typical single frame |
|---|---|---|---|
| 20 (light) | 214 kB | 113 | 106 |
| 28 (typical) | 19 kB | 40.5 | 34.0 |
| 35 (heavy) | 7 kB | 4.8, not detected | 3.4 |
| 42 | 6 kB | 0.2 | 0.2 |

The detection line is 6. A wrong key scored 0.0 and the unmarked clip −0.4.
At CRF 28 the codec stored 1 I-frame, 18 P-frames and 53 B-frames, with average
single-frame scores of 35.8, 32.8 and 34.2: the score follows the codec's
rhythm, not the picture.

The clip saved at CRF 28, then edited and compressed a second time:

| Edit | Label | Frame watermark (pooled score) |
|---|---|---|
| Re-encode, light / typical | lost | survives (34.6 / 25.7) |
| Re-encode, heavy | lost | lost (1.7) |
| Scale to half size | lost | survives (10.2) |
| Half the frame rate | lost | survives (36.9) |
| Trim to the last third | lost | survives (30.4) |
| Crop 10% off each edge | lost | lost (0.1) |
| Mirror left to right | lost | lost (−0.4) |
| Screen recording (75% size, slight blur) | lost | lost (2.5) |
| One frame saved as a picture | lost | survives (32.8) |

One procedural clip, one codec, one machine: a demonstration, not a benchmark.

## Limits

- **Strength against quality.** The mark here costs 36 dB PSNR (the still-image
  mark in step two cost 41 dB) and still dies at heavy compression. Closing
  that gap is what the learned schemes are trained for.
- **Geometry.** A hand-built pattern cannot re-align itself after a crop or a
  mirror. Learned extractors are trained against such edits.
- **A fixed pattern can be averaged out.** Averaging frames from many different
  marked clips cancels the pictures and leaves the pattern, which can then be
  subtracted. Content-dependent marks resist this better.
- **Everything from the image step still applies:** regeneration attacks,
  forgery of public marks, and false positives of learned detectors.

## What was not verified

- Vendor practice above comes from search-result summaries and secondary
  articles, apart from the Google DeepMind page read during step one.
- Writing video needs ffmpeg, which this machine has; the page says so when it
  is missing. Only H.264 was tested. HEVC, VP9 and AV1 were not.

## Sources

From search-result summaries only:
- Meta, *Video Seal: Open and Efficient Video Watermarking*: https://arxiv.org/abs/2412.09492
- Meta engineering, *Video invisible watermarking at scale*: https://engineering.fb.com/2025/11/04/video-engineering/video-invisible-watermarking-at-scale/
- Meta Seal: https://facebookresearch.github.io/meta-seal
- Google DeepMind, *SynthID-Image: Image watermarking at internet scale*: https://arxiv.org/abs/2510.09263
- Vendor comparisons (secondary): https://www.layer3labs.io/comparisons/which-ai-models-watermark-their-output, https://spotwatermark.com/blog/which-ai-models-have-watermarks
- Platform disclosure rules (secondary): https://www.clipspeed.ai/blog/ai-generated-video-disclosure-rules.html

Read directly in step one: Google DeepMind, SynthID: https://deepmind.google/models/synthid/

From memory, not re-read: how H.264 stores I-, P- and B-frames; the MP4 box
layout and the XMP and C2PA `uuid` identifiers; AudioSeal; that Sora's badge
moves.
