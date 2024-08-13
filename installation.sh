#!/bin/bash

apt-get -y update
apt-get -y upgrade
apt-get install -y ffmpeg
pip install -r requirements.txt