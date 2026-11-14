"""Text watermarking, step one of the learning project.

    prf.py       keyed pseudo-random function shared by both schemes
    greenlist.py green/red-list watermark (Kirchenbauer et al., 2023)
    tournament.py tournament-sampling watermark (SynthID-Text, Dathathri et al., 2024)
    hidden.py    scan for invisible Unicode characters
    lm.py        the small local language model the watermarks are applied to
    analyze.py   run every detector over a pasted text
"""
