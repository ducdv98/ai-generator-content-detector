# How AI audio gets marked (state of play, November 2026)

Research notes for step four of this project. The code is in `app/audio/`,
and the page at `/audio` shows it working. Earlier steps:
[text](text-watermarking.md), [images](image-watermarking.md),
[video](video-watermarking.md).

## What is different about sound

1. **The ear, not the eye, sets the rules.** A loud sound hides quieter sounds
   near it in pitch (frequency masking) and just after it in time (forward
   masking). Every audio watermark, and every lossy audio codec, is built
   around those two facts. That puts them in direct competition: MP3 and Opus
   save bits by throwing away what the ear will not miss, which is exactly
   where a watermark hides.
2. **There is no fixed grid.** An image has pixel positions and a video has
   frames. A clip of sound can start anywhere, so a mark must either be found
   by searching for its alignment or be built so that alignment does not
   matter.
3. **Time and pitch can be stretched.** Speeding a recording up by a few
   percent is inaudible to most listeners and moves every frequency.

## The channels

| Channel | What it is | Survives a re-upload? |
|---|---|---|
| **File label** | C2PA Content Credentials, XMP or ID3 tags in a block beside the sound. `app/audio/container.py` | No. Converting or re-encoding writes a new file from the sound alone. |
| **Sound watermark** | A change to the waveform itself, hidden under the sound. `app/audio/mark.py` | Depends on bitrate, noise and pitch changes. |

## The scheme built here

**Writing.** The key assigns +1 or −1 to each of 550 pitch bands, 21.5 Hz wide,
from 150 Hz to 12 kHz. A filter makes each band a fraction of a decibel louder
or quieter to match. In time, a fine ripple across pitch is the same thing as a
set of faint echoes, and the filter is built minimum-phase so that all of them
fall in the roughly 50 ms after each sound, where forward masking hides them.
This is "echo hiding", one of the oldest audio watermarks (Gruhl, Lu and
Bender, 1996), with a keyed pattern instead of a single echo. The message is
16 preamble bits (is this key's mark here?) plus a 16-bit ID, each bit spread
over about 17 bands.

Because the mark is the sound re-coloured, it is loud only where the music is
loud and silent where the music is silent. That is the masking idea in its
simplest form.

**Reading.** Measure each band's loudness in each 186 ms frame, subtract the
median of its neighbours (the instrument's own tone), average over time, and
correlate with the key's pattern. The music's fine structure changes from note
to note and averages away; the pattern is the same at every instant and stays.
The score is calibrated: for any key that did not write the mark it has mean 0
and spread 1, whatever the music.

**No alignment needed.** The pattern depends on pitch only, never on time. Cut
the start off, keep any stretch, change the tempo: the bands are where they
were. It also lets every 1.5-second stretch be scored on its own, giving a
timeline that would show a spliced-in section as a dip.

### What did not work first, and why

The first version added a fixed ripple of ±1 dB to every band. On the sample
tune it scored 3.2 against a detection line of 6, although the same mark on
plain noise scored 16.0, close to the most the scheme can reach. The music's
own harmonics were the problem: a note sitting in a band stands far above its
neighbours, which swamps a 1 dB ripple, and averaging does not help when the
same notes keep returning. Three changes took the score from 3.2 to 9.0. The
third changes the test clip, so the two figures are not a like-for-like
comparison:

- **The detector ignores bands occupied by a pure tone.** A band more than
  5 dB away from its neighbours in a given frame says nothing about the mark
  and is skipped for that frame. The mark is read from everything else: noise,
  reverberation, the gaps between harmonics. In the sample 68% of band readings
  were usable.
- **The writer listens first** ("informed embedding"). It measures where each
  band already sits, leaves alone the ones the music has on the right side, and
  pushes only the others, never by more than a limit. On its own this raised
  the score only from 3.2 to 4.0.
- **The sample tune got a room tone.** The first tune was nothing but pure
  harmonics over digital silence, which no real recording is. A faint noise
  floor, 45 dB down, gives the gaps between harmonics something for the mark to
  colour. This makes the sample more realistic and also easier to mark.

Deployed systems go much further down the same road: an encoder network shapes
the mark to the sound moment by moment.

## What vendors deploy

- **Meta, AudioSeal (open).** A trained generator writes the mark and a trained
  detector finds it, with a 16-bit message. One paper reports it is the scheme
  applied to audio generation in Meta's public AI products.
- **Google, SynthID for audio.** Described as encoding a pattern into the
  audio's spectrogram at generation time. OpenAI's content-provenance results
  for audio are reported to include a SynthID entry.
- **Others.** Resemble AI describes its own watermark (PerTh). Suno is reported
  to have announced a watermarking and fingerprinting system in August 2026.
- **Robustness research.** AudioMarkBench (NeurIPS 2024) tested three
  state-of-the-art audio watermarks against 15 kinds of change and found them
  vulnerable to removal; other work singles out speed changes as especially
  effective. The table below shows the same weakness in miniature.

All of this comes from search-result summaries, not pages I read in full.

## So can a pasted clip be checked?

| Question | Answer |
|---|---|
| Does the file carry a label saying it is AI-made? | Yes, if the file is the original. |
| Does it carry a keyed sound watermark made here? | Yes, with the key, unless it was compressed hard, had noise added, or was pitch-shifted. |
| Does it carry AudioSeal, SynthID or another vendor's mark? | Not here. AudioSeal's detector is open source but is a neural network this project does not run; SynthID's is not published. |
| Is it AI-made at all? | Not a watermark question. No mark found means "no evidence", never "human". |

## Measurements

The built-in 8-second tune (plucked chords, bass, light percussion, faint room
tone), key `demo-key`. Detection line 6; the most this scheme can score is 16.7.

| Strength | Bands moved | Average change | Mark level below the music | Score | ID read |
|---|---|---|---|---|---|
| Subtle | 83% | 0.50 dB | 21.4 dB | 4.4, not detected | – |
| Normal | 96% | 0.99 dB | 15.0 dB | 9.0 | BEEF, correct |
| Strong | 99% | 1.50 dB | 10.2 dB | 11.1 | BEEF, correct |

The unmarked tune scored between −2 and 2 for 200 different keys (mean 0.02,
spread 1.05). With the normal mark, the score using only the first 0.65 s was
4.1 and rose to 9.0 over the full 8 s; 7 of 13 separate 1.5-second stretches
passed the line on their own. Sound really does add fresh evidence as it goes
on, unlike video frames, so a longer clip could carry a quieter mark.

The normal-strength clip, edited and saved as a new file:

| Edit | Label | Sound watermark (score) |
|---|---|---|
| MP3, 128 kbps | lost | survives (6.2), ID correct |
| MP3, 64 kbps | lost | lost (1.1) |
| AAC, 96 kbps | lost | lost (4.0) |
| Opus, 24 kbps | lost | lost (1.2) |
| Phone line, 8 kHz sample rate | lost | lost (1.1) |
| Background noise, 20 dB below the sound | lost | lost (2.4) |
| Volume down to a quarter | lost | survives (9.0) |
| Trim to the last third | lost | survives (6.8) |
| First 0.37 s cut off | lost | survives (8.9) |
| Tempo up 5%, same pitch | lost | survives (8.0) |
| Speed up 5%, pitch rises | lost | lost (−1.6) |
| Pitch up 1%, same tempo | lost | lost (−1.0) |
| Echo of a small room | lost | survives (8.5) |

One synthesised tune, one machine: a demonstration, not a benchmark.

## Limits

- **I could not listen to it.** Whether the normal mark is audible is for your
  ears; the page plays the clip before and after, and the mark on its own. At
  15 dB below the music it is a much heavier mark than a product would use,
  and the subtle setting is not detectable in 8 seconds.
- **Pitch.** A 1% pitch change removes it. A detector could search over small
  pitch offsets; this one does not.
- **Low bitrates and noise** replace the quiet detail between the strong tones,
  which is where the mark lives.
- **A fixed pattern can be estimated** by anyone with many marked clips, and
  then removed or forged, as with the video mark.
- **Mono only.** Clips are mixed to one channel before marking or checking.

## What was not verified

- Vendor practice and the robustness literature come from search summaries.
- Only the sample tune was measured. Speech, which has no long steady notes,
  should be easier for this detector, but that was not tested.

## Sources

From search-result summaries only:
- AudioMarkBench (NeurIPS 2024): https://arxiv.org/abs/2406.06979
- SoK: Watermarking for AI-Generated Content: https://arxiv.org/abs/2411.18479
- Watermarking Without Standards Is Not AI Governance: https://arxiv.org/abs/2505.23814
- Meta, Seamless Interaction (uses AudioSeal): https://arxiv.org/abs/2506.22554
- OpenAI, Verify OpenAI-generated content: https://openai.com/research/verify/
- Resemble AI on voice watermarking (vendor page): https://www.resemble.ai/resources/best-voice-watermarking-tools-in-2026
- AI music watermarks explained (secondary): https://blog.wavedisco.com/blog/ai-music-watermarks-explained

From memory, not re-read: Gruhl, Lu and Bender, *Echo hiding* (1996); frequency
and forward masking; minimum-phase filter design by cepstrum folding; informed
embedding (Cox; Malvar and Florêncio's improved spread spectrum); the WAV, ID3
and XMP chunk layouts.
