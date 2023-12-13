import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from feature_extraction import create_mel_filters, generate_mfccs


def generate_mfccs_from_file(path, mfcc_dir, mfcc_params):
    print(path)

    audio, sample_rate = sf.read(path)

    filters = create_mel_filters(40, 512, sample_rate)
    data = generate_mfccs(audio, sample_rate, filters, mfcc_params['n_ceps'])

    output_dir = Path(f'{mfcc_dir}/testing')

    if int(path.stem.split('_')[1]) <= 20:
        output_dir = Path(f'{mfcc_dir}/training')

    mfcc_path = Path(f'{output_dir}/{path.stem}.npy')
    np.save(mfcc_path, data, allow_pickle=True)

    label = path.stem.split('-')[0]

    return data, label


audio_dir = Path('../audio/')
mfcc_dir = Path('../audio_features/mfccs')

mfcc_params = {
    'n_ceps': 13
}

audio_files = sorted(audio_dir.glob('*.wav'))
generated_data = [generate_mfccs_from_file(file, mfcc_dir, mfcc_params) for file in audio_files]


