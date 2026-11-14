"""Video watermarking, step three of the learning project.

    codec.py      turn frames into an H.264 file and back (ffmpeg writes, OpenCV reads)
    container.py  what an MP4 is made of, and the label stored beside the frames
    frames.py     a keyed watermark in every frame, detected by pooling frames
    sample.py     a small procedural clip to experiment on
    lab.py        make a marked clip, check a clip, test which edits each mark survives
"""
