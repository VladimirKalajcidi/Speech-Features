# SPEECH FEATURES

## Setup
Install Docker.

```sh
$ git clone https://github.com/VladimirKalajcidi/endtoend.git
$ cd endtoend
$ docker build -t whisper .
```

## Usage
### Extracting Speech Features 
Execute following command in `endtoend` directory.

```sh
$ docker run -it -d -v $(pwd):/app/ --net host --name whisper whisper
$ docker exec -it whisper bash
root@hostname:/workspace# python app.py -i audio.mp3 -o output.json -m openai/whisper-small 
```

## arguments:
    - i: input .mp3 file
    - o: ouput json file
    - m: model path, defalut = "openai/whisper-base"