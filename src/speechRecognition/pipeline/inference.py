import os
import audeer
import audonnx
import audiofile
import librosa
import audresample
from pydub import AudioSegment
import whisperx
#from utils.common import recognize_emotions

def recognize_emotions(audio):

        url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
        cache_root = audeer.mkdir('cache')
        model_root = audeer.mkdir('model')

        archive_path = audeer.download_url(url, cache_root, verbose=True)
        audeer.extract_archive(archive_path, model_root)
        model = audonnx.load(model_root)

        sound = AudioSegment.from_mp3(audio)
        sound.export("emotions.mp3", format="mp3")
        audio = "emotions.mp3"

        wav, fs = audiofile.read(audio)
        if fs != 16000:
            wav = audresample.resample(wav, fs, 16000)

        duration = librosa.get_duration(filename=audio)

        os.remove("emotions.mp3")

        for i in range(wav.shape[0] // int(fs * duration)):
                pred = model(
                    wav[int((0 + i) * fs * duration):  int((i+1) * fs * duration)], fs)
                emotion = f"Arousal, dominance, valence #{i}: {pred['logits']}"
                return emotion
        if wav.shape[0] % int(fs * duration) != 0:
            pred = model(wav[-(wav.shape[0] % (fs*duration)):], fs)
            emotion = f"Arousal, dominance, valence #{i+1}: {pred['logits']}"
            return emotion


class PredictionPipeline:
    def __init__(self, audio):
        self.audio=audio
    
    def transcribe(self, model: str, token: str):
        device = "cpu" 
        audio_file = self.audio
        batch_size = 16
        compute_type = "int8"

        model = whisperx.load_model(model, device, compute_type=compute_type)

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


        diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)

        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        Audio = AudioSegment.from_mp3(self.audio)

        for i in result['segments']:
            t1 = i['start'] * 1000
            t2 = i['end'] * 1000
            newAudio = Audio[t1:t2]
            newAudio.export('newAudio.mp3', format="mp3")
            i['emotion'] = recognize_emotions('newAudio.mp3')
            os.remove('newAudio.mp3')

        output = {}

        ind = 0
        for i in result['segments']:

            output[ind] = {"start": i['start'],
                           "end": i['end'],
                           "speaker": i['speaker'],
                           "emotions": i["emotion"],
                           "text": i['text']}

            ind += 1
        return output
