#!/bin/bash

apt-get -y update
apt-get -y upgrade
apt-get install -y ffmpeg
apt-get install -y sox

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt