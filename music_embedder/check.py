
import os
from datasets import Dataset, Audio

# https://huggingface.co/docs/datasets/en/audio_dataset

d = './data_mp3/'
listed = [os.path.abspath('{0}{1}'.format(d, x)) for x in os.listdir(d)]


audio_dataset = Dataset.from_dict({"audio": listed}).cast_column("audio", Audio(sampling_rate=16000))
# audio_sampling_rate = 44100
audio_sampling_rate = 16000
audio_dataset[0]["audio"]

from transformers import pipeline

processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")

pipeline(feature_extractor=..., model=...)
