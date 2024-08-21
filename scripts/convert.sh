#!/bin/bash
if test -d $2; 
then echo "Model file already exists"
else ct2-transformers-converter --model $1 --output_dir $2
fi

python src/speechRecognition/utils/push_to_hub.py -m $2 -d $3