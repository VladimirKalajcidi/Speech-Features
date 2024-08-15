from speechRecognition.pipeline.inference import PredictionPipeline
import argparse
import json
 
parser = argparse.ArgumentParser(description="app")
parser.add_argument("-i", "--input", help="input file", required=True)
parser.add_argument("-o", "--output", help="output file", required=True)
parser.add_argument("-m", "--model", help="model path", default="openai/whisper-base")
parser.add_argument("-t", "--token", help="huggingface token", required=True)
args = parser.parse_args()


pipe = PredictionPipeline(args.input)
output = pipe.transcribe(args.model, args.token)
    
with open(args.output, 'w') as fp:
        json.dump(output, fp, ensure_ascii=False,)
