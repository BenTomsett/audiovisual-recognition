# scale_videos.py
# Scales all videos down by half
# Author: Ben Tomsett


import ffmpeg
from pathlib import Path
from tqdm import tqdm

all_videos = list(Path('../videos').glob('**/*.mp4'))

for video in tqdm(all_videos, desc='Scaling videos'):
    folder = video.parts[-2]
    filename = video.stem

    # equivalent ffmpeg command
    # ffmpeg -i {input} -filter:v scale=810:-1 -c:a copy {output}

    input = video.absolute().as_posix()
    output = Path(f'../videos_scaled/{folder}/{filename}.mp4').absolute().as_posix()
    Path.mkdir(Path(f'../videos_scaled/{folder}'), exist_ok=True, parents=True)

    ffmpeg.input(input).output(output, vf='scale=810:-1', acodec='copy', loglevel='quiet').run()

