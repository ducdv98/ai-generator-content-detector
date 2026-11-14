# How LLM text gets watermarked (state of play, November 2026)

Research notes for step one of this project. The code that goes with them is in
`app/text/`, and the page at `/text` shows it working.

## The one idea

A language model does not write a word. It scores every token in its vocabulary,
turns the scores into probabilities, and a random draw picks one. Every text
watermark in production today lives in that last step:

```
prompt ─► model ─► scores for ~50,000 tokens ─► [watermark] ─► random draw ─► next token
                                                     ▲
                                  secret key + the few tokens just written
```

A keyed pseudo-random function turns (key, recent tokens, candidate token) into
a number. The generator uses that number to tilt the draw. Nothing is added to
the text: no hidden characters, no metadata. The mark is a statistical lean in
which of several acceptable words was chosen, far too small to see in one word
and unmistakable over a few hundred.

To detect it you recompute the same pseudo-random numbers for the tokens in
front of you and ask whether they are luckier than chance. That needs the
**key** and the **tokenizer**, and nothing else: not the model, not the prompt.

## The three families

### 1. Green-list (Kirchenbauer et al., ICML 2023), `app/text/greenlist.py`

- Hash key + previous token, and use it to split the vocabulary into a green
  share γ (0.25 here) and a red rest. The split changes after every token.
- Add δ (2.0 here) to every green token's score, then sample as usual.
- Detect: count green tokens among T scored tokens.
  `z = (greens − γT) / sqrt(γ(1−γ)T)`. z ≥ 4 is about a 1 in 30,000 false alarm.
- Trade-off: it changes the model's probabilities, so quality drops a little as
  δ rises. Simple, robust, and the baseline everything else is compared with.

### 2. Gumbel / exponential sampling (Aaronson, 2022; not implemented here)

- The key gives every candidate token a pseudo-random number r in (0, 1). Pick
  the token maximising `r ^ (1 / p)`, where p is the model's probability.
- For a random key this picks each token with exactly probability p, so the
  text distribution is unchanged, but the choice is now fixed by the key.
- Detect: chosen tokens have r close to 1; sum `−ln(1 − r)` and compare with T.
- This is the scheme OpenAI built internally in 2022-23 and did not ship then.

### 3. Tournament sampling (SynthID-Text, Nature 2024), `app/text/tournament.py`

- Key + previous 4 tokens give each candidate 30 random bits (g-values).
- Draw 2^30 candidates from the model's own distribution and run a knockout:
  round k keeps, from each pair, the one with the higher k-th bit.
- Each round has a closed form, `p(x) ← p(x) · (1 + g(x) − Σ p(y) g(y))`, so
  the real cost is 30 cheap updates, not 2^30 draws.
- Non-distortionary like Gumbel (probabilities are unchanged on average), with
  a dial (number of rounds) for how much signal each token carries.
- Detect: average the g-values of the tokens; 0.5 is chance. Google also uses a
  weighted mean and a trained Bayesian scorer.

### Others worth knowing

- **Provably undetectable / robust distortion-free** schemes (Christ et al.
  2023; Kuditipudi et al. 2023): stronger guarantees, not known to be deployed.
- **Semantic watermarks** (SemStamp and similar) mark sentence meanings rather
  than tokens to survive paraphrase; research stage.
- **Hidden characters** (zero-width spaces, look-alike spaces, Unicode tags):
  trivial to add after generation, trivial to strip. `app/text/hidden.py`
  scans for them. No large vendor uses them as its watermark.

## Who deploys what

| Vendor | Scheme | Where | Public text detector |
|---|---|---|---|
| Google | SynthID-Text (tournament sampling) | Gemini app and web, since 2024 | No. The SynthID Detector and Gemini's "verify" flow are documented for image, video and audio. The algorithm is open source (in Hugging Face Transformers) but Gemini's keys are not. |
| Anthropic | "A version of the SynthID-Text approach" | Claude models launched from 2 August 2026, worldwide, on by default; older models being added | No. A detection API is in private preview for regulators, researchers, media and similar. |
| OpenAI | "textGrain": a key biases word choice through several adjusted probability sets whose average matches the original | ChatGPT and Codex text for EU users (announced 5 October 2026); opt-in for API customers | No. Detector access is case-by-case for approved researchers and expert organisations. |

All three say the same things about their marks: nothing visible is added, no
user or conversation is identified, short or heavily edited text may not be
detectable, and a missing mark proves nothing about authorship.

The trigger is **EU AI Act Article 50(2)**: providers must mark generated
content in a machine-readable, detectable way. The Commission's Code of Practice
on marking and labelling was published on 10 June 2026 and says no single
technique is enough, so it expects layers (metadata plus watermark). Article 50
applies from 2 August 2026; some sources give systems already on the market
until 2 December 2026 for the marking duty. Check the Official Journal before
relying on either date.

## So can a pasted paragraph be checked?

| Question | Answer |
|---|---|
| Does this text carry a watermark made with a key I hold? | Yes. Reliable from roughly 100-200 tokens. This is what `/text` does. |
| Does this text carry Gemini's / Claude's / ChatGPT's watermark? | Not by an outsider. Without the key the lean is indistinguishable from chance, by design. Only the vendor's own detector can answer, and none is public for text. |
| Does this text contain invisible characters? | Yes, with no key. But it is weak evidence of anything. |
| Was this text written by AI at all? | Not a watermark question. That is the job of statistical classifiers, which guess from style and have real false-positive rates. |

Research has shown keys can leak: with many thousands of watermarked samples
from one model, an attacker can estimate a green-list split well enough to
forge or scrub the mark ("watermark stealing", Jovanović et al. 2024). That is
an attack needing bulk access to the model, not a way to check one paragraph.

## Limits that apply to every scheme

- **Length.** Evidence adds up token by token. A sentence cannot be checked.
- **Low-entropy text.** Where only one answer is right (code, maths, quoted
  facts, lists of names) there is no choice to lean on, so there is little mark.
- **Editing.** Each changed token removes its own evidence and that of the
  tokens seeded by it. Light edits weaken the score; a paraphrase or a
  round-trip through translation usually erases it. OpenAI reports detection
  falling from about 92% to 66% after replacing 10% of words, on 400 tokens.
- **Tokenizer.** The detector must split the text exactly as the generator
  did. Inserted invisible characters change the split, which is one reason
  they lower the score (try it on the page).
- **Meaning of a hit.** It shows the model with that key produced most of
  these tokens. It does not say who prompted it, or whether a person then
  approved, edited or merely translated their own writing with it.

## What the measurements here show

With the 360M-parameter demo model and about 200 tokens, both schemes score
z ≈ 7 to 10 with the right key and z within ±3 with a wrong key or on unmarked
text. On 1,500 unmarked token sequences both detectors' scores had mean 0 and
standard deviation 1, and none reached the z = 4 line (`tests/test_text.py`
keeps a smaller version of this check). These are a small model's numbers on a
handful of prompts, not a benchmark.

## Sources

Primary pages read for these notes:
- Kirchenbauer et al., *A Watermark for Large Language Models*: https://arxiv.org/abs/2301.10226
- Dathathri et al., *Scalable watermarking for identifying large language model outputs*, Nature 634 (2024): https://www.nature.com/articles/s41586-024-08025-4
- Anthropic, *How Claude's text watermarking works* (14 Aug 2026, updated 1 Sep): https://www.anthropic.com/news/claude-text-watermark
- Anthropic help centre, *How Claude marks AI-generated content*: https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content
- Claude Platform docs, *Text watermark*: https://platform.claude.com/docs/en/build-with-claude/text-watermark
- Google DeepMind, SynthID: https://deepmind.google/models/synthid/
- Nemecek, Chaudhary, Ayday, *Watermarks Without Verification* (Sep 2026, abstract only): https://arxiv.org/abs/2609.09604

Known only through search-result summaries (the pages refused automated
reading), so treat the details as second-hand:
- OpenAI, *Our approach to EU text provenance rules*: https://openai.com/index/eu-text-provenance/
- OpenAI help centre, *Provenance signals in OpenAI-generated content*: https://help.openai.com/en/articles/8912793-provenance-signals-in-openai-generated-content
- EU Code of Practice coverage, e.g. Jones Day: https://www.jonesday.com/en/insights/2026/06/european-commission-publishes-final-code-of-practice-on-marking-and-labelling-aigenerated-content

From memory, not re-read for these notes: Aaronson's 2022 talk on the Gumbel
scheme; Christ et al.; Kuditipudi et al.; Jovanović et al.
