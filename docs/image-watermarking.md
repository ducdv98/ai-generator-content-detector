# How AI images get marked (state of play, November 2026)

Research notes for step two of this project. The code is in `app/image/`, and
the page at `/image` shows it working. Step one (text) is in
[text-watermarking.md](text-watermarking.md).

## The big difference from text

Text has one place to hide a mark: the choice of words. An image file has two,
and vendors now use both at once.

```
image file = [ notes about the picture ]  +  [ the picture itself ]
                  ▲                               ▲
          labels / metadata                 pixel watermark
     exact, signed, easy to read       survives copying, needs a detector
     gone after a screenshot           can be weakened by editing
```

The EU Code of Practice on marking AI content (June 2026) says no single
technique is enough and asks for at least two layers. In practice that is a
signed label plus an invisible pixel watermark.

## Layer 1: labels stored beside the pixels

| Label | What it is | Trust |
|---|---|---|
| **C2PA Content Credentials** | A signed manifest: who made the file, with which tool, what was done to it. Stored in JUMBF blocks (JPEG APP11, PNG `caBX`). | Cryptographically signed, so tampering shows. Says nothing once removed. |
| **IPTC DigitalSourceType** | One XMP tag, e.g. `trainedAlgorithmicMedia`. | Unsigned. Anyone can add or delete it. |
| **Generator metadata** | Stable Diffusion WebUI `parameters`, ComfyUI `workflow` PNG chunks, EXIF `Software`. | Unsigned, incidental. |

`app/image/anatomy.py` splits a file into its blocks so the page can show where
these live and how small they are next to the picture data. The lesson the
"what survives" table makes concrete: every edit that re-encodes the picture
(upload to most platforms, paste into a chat, screenshot) drops all of them.

## Layer 2: marks inside the pixels

### a. Fixed public pattern: Stable Diffusion's watermark (`app/image/public.py`)

The open-source Stable Diffusion scripts and the diffusers SDXL pipeline call
the `invisible-watermark` package, method "dwtDct":

1. Take the U colour channel, one level of Haar wavelet, keep the low-pass band.
2. In each 4×4 block of that band (8×8 pixels), move the largest value onto a
   grid of step 36: remainder 9 for a 0 bit, 27 for a 1 bit.
3. Block k carries bit `k mod N` of a fixed message: `StableDiffusionV1`
   (136 bits), `SDV2` (32 bits) or a 48-bit constant for SDXL.

No key, published algorithm, published message. Anyone can read it, strip it or
stamp it on a photograph. Reading the upstream source turned up two things the
name hides: there is no DCT step (that code is commented out), and the inverse
transform swaps two detail bands, which slightly scrambles colour detail. The
port keeps both so it behaves like the real thing.

Measured here on the sample picture: 129 of 136 bits read back from an
untouched image (smooth areas do not hold the mark, because nudging one value
changes which value is largest). It was lost after JPEG at quality 90, any
resize and a 10% crop, and survived blur, mild noise and a brightness change.

### b. Keyed pattern: spread spectrum (`app/signing.py`, `app/image/keyed.py`)

The classic idea (Cox et al., 1997) behind everything that follows:

1. A secret key generates a +1/−1 pattern over mid-frequency DCT values of the
   brightness channel.
2. Each message bit is spread over about a thousand of those values and added
   faintly.
3. Detection multiplies the image's values by the key's pattern and sums. Right
   key: thousands of tiny agreements add up to a large score. Any other key: the
   sum is noise around zero.

This is the image twin of the text watermarks from step one: a key, evidence
too small to see in any one place, and a z-score that grows with the amount of
material. Measured here: score 81 on the sample (threshold 6), about −0.5 with
a wrong key, still above 50 after JPEG quality 30 and after resizing, and lost
after a 10% crop, because the hand-made pattern has no way to re-align itself.

### c. Learned watermarks: what vendors deploy

An encoder network writes the mark and a decoder network reads it. They are
trained together while the marked image is put through crops, rotations,
resizing, filters and compression, so the pair learns a pattern that survives
them. The secret is the decoder's weights (and often a key or payload on top).

- **SynthID-Image (Google DeepMind).** Post-hoc: applied to the finished image,
  independent of the generator. The October 2025 paper reports over ten billion
  images and video frames marked, and describes a variant offered to partners.
- **TrustMark (Adobe, open source).** Carries an identifier that lets a
  stripped Content Credential be found again ("durable" credentials).
- **Stable Signature, Watermark Anything, Meta Seal (Meta, open research).**
  Stable Signature fine-tunes the generator's own decoder so every image it
  produces is born marked; no separate stamping step exists to skip.
- **In-generation research schemes** (Tree-Ring, Gaussian Shading) hide the
  mark in the diffusion model's starting noise instead of the output pixels.

## Who does what

| Vendor | File label | Pixel watermark | Can the public check? |
|---|---|---|---|
| Google (Gemini, Imagen) | C2PA | SynthID | Yes, Google's own tools: the Gemini app and the SynthID Detector, now described as open to everyone. |
| OpenAI (ChatGPT, API images) | C2PA | SynthID, announced May 2026 in partnership with Google | Yes, OpenAI's verify page checks for both signals. |
| Adobe Firefly | C2PA | "Durable" credentials watermark | Adobe's Content Authenticity tools. |
| Meta AI | IPTC / C2PA labels, visible badge | Invisible watermark (2024 policy; current system not confirmed) | Not found. |
| Amazon (Titan, Nova Canvas) | C2PA | Invisible watermark | AWS detection tooling for its own models (per secondary sources). |
| Stable Diffusion, self-hosted | Often generator metadata | Public dwtDct mark, if the script's default is left on | Yes, by anyone: this project reads it. |
| Midjourney | Reported as none | Reported as none | Nothing to check. |

Rows for Adobe, Meta, Amazon and Midjourney rest on secondary articles; treat
them as leads, not facts.

## So can an uploaded image be checked?

| Question | Answer |
|---|---|
| Does the file carry a label saying it is AI-made? | Yes, exactly, if the file is the original. This project reads C2PA, IPTC and generator metadata. |
| Does it carry Stable Diffusion's public watermark? | Yes, offline, by anyone. But most images online have been resized or recompressed, which destroys it. |
| Does it carry a keyed watermark made here? | Yes, with the key. |
| Does it carry SynthID or another vendor's learned watermark? | Not offline. The decoder is not published. Unlike text, though, Google and OpenAI do run public checkers for their own images. |
| Is it AI-made at all? | Not a watermark question. No mark found means "no evidence", never "real". |

## Limits

- **Removal.** Regenerating an image through a diffusion model, or attacks such
  as UnMarker, have been shown to erase invisible watermarks, learned ones
  included, at some cost in quality.
- **Forgery.** A public mark can be stamped onto a real photo by anyone. Keyed
  and learned marks resist this only while the key or encoder stays private.
- **Labels prove origin only when signed and intact.** C2PA signer trust lists
  are not checked in this project (see the README).
- **False positives exist** for learned detectors; vendors publish little about
  rates.

## What was not verified

- The port of the public watermark follows the upstream source line by line,
  but it has not been run against the `invisible-watermark` package or a real
  Stable Diffusion output on this machine (installing the package was not
  permitted in this session). `tests/test_image.py` contains that cross-check
  and runs it automatically once the package is installed.
- Sample measurements are from one procedural test picture, not a benchmark.

## Sources

Read directly:
- invisible-watermark source (`imwatermark/maxDct.py`, `watermark.py`): https://github.com/ShieldMnt/invisible-watermark
- Google DeepMind, SynthID: https://deepmind.google/models/synthid/

From search-result summaries only:
- Google DeepMind, *SynthID-Image: Image watermarking at internet scale* (Oct 2025): https://arxiv.org/abs/2510.09263
- OpenAI, *Advancing content provenance*: https://openai.com/index/advancing-content-provenance/
- OpenAI, *Verify OpenAI-generated content*: https://openai.com/research/verify/
- OpenAI help centre, *Provenance signals (Content Credentials, SynthID)*: https://help.openai.com/en/articles/8912793
- Meta Seal: https://facebookresearch.github.io/meta-seal
- Vendor comparison (secondary): https://www.sentisight.ai/hidden-watermarks-in-generative-ai-models/

From memory, not re-read: Cox et al. 1997 (spread spectrum); Stable Signature;
Tree-Ring; Gaussian Shading; TrustMark; UnMarker; regeneration attacks.
