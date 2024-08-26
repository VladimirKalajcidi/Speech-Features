# SPEECH FEATURES

## Setup
Install Docker.

```sh
$ git clone https://github.com/VladimirKalajcidi/Speech-Features.git
$ cd endtoend
$ docker build -t stt .
```

## Usage
### Extracting Speech Features 
Execute following command in `Speech-Features` directory.

```sh
$ docker run -it -d -v $(pwd):/app/ --net host --name stt stt
$ docker exec -it stt bash
root@hostname:/workspace# ./scripts/installation.sh
root@hostname:/workspace# python app.py -i audio.mp3 -o output.json -m base -t token
```
Arguments for `app.py`:
```sh
    - i: input .mp3 file
    - o: ouput .json file
    - m: model path, defalut = "base", custom = "username/repo"
    - t: huggingface token ("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")
```

## Model training
Execute following command in `Speech-Features` directory.

```sh
$ docker run -it -d -v $(pwd):/app/ --net host --name stt stt
$ docker exec -it stt bash
root@hostname:/workspace# ./scripts/installation.sh
root@hostname:/workspace# dvc repro
```

## Converting trained openAI model to CT2 model
Execute following command in `Speech-Features` directory.

```sh
$ docker run -it -d -v $(pwd):/app/ --net host --name stt stt
$ docker exec -it stt bash
root@hostname:/workspace# ./scripts/installation.sh
root@hostname:/workspace# ./scripts/convert.sh artifacts/training/model whisper-ct2 username/repo
```
Arguments for `convert.sh`:
```sh
    1: path to trained model
    2: path to converted model in local repository
    3: path to HuggingFace repository
```