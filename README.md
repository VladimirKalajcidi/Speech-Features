# SPEECH FEATURES

## Setup
Install Docker.

```sh
$ git clone https://github.com/VladimirKalajcidi/endtoend.git
$ cd endtoend
$ docker build -t stt .
```

## Usage
### Extracting Speech Features 
Execute following command in `Speech-Features` directory.

```sh
$ docker run -it -d -v $(pwd):/app/ --net host --name stt stt
$ docker exec -it stt bash
root@hostname:/workspace# ./installation.sh
root@hostname:/workspace# python app.py -i audio.mp3 -o output.json -m openai/whisper-small 
```
Arguments for `app.py`:
```sh
    - i: input .mp3 file
    - o: ouput .json file
    - m: model path, defalut = "openai/whisper-base", finetuned = "artifacts/training/model"
```

## Model training
Execute following command in `Speech-Features` directory.

```sh
$ docker run -it -d -v $(pwd):/app/ --net host --name stt stt
$ docker exec -it stt bash
root@hostname:/workspace# ./installation.sh
root@hostname:/workspace# dvc repro
```