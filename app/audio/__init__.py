"""Audio watermarking, step four of the learning project.

    codec.py      any audio file to samples and back (ffmpeg), WAV writing, spectrogram pictures
    container.py  what a WAV / MP3 / M4A file is made of, and the label stored beside the sound
    mark.py       a keyed watermark made of faint echoes; detection with a score over time
    sample.py     a short synthesised tune to experiment on
    lab.py        make a marked clip, check a clip, test which edits each mark survives
"""
