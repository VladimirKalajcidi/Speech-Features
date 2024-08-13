from speechRecognition.pipeline.prediction import PredictionPipeline
import argparse
import json
 
parser = argparse.ArgumentParser(description="app")
parser.add_argument("-i", "--input", help="input file", required=True)
parser.add_argument("-o", "--output", help="output file", required=True)
args = parser.parse_args()


pipe = PredictionPipeline(args.input)
text = pipe.transcribe()
emotion = pipe.recognize_emotions()

data = {"text": text,
        "emotions": emotion}
with open(args.output, 'w') as fp:
    json.dump(data, fp, ensure_ascii=False,)
