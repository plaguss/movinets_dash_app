#!/bin/bash
cd /tmp
python3 -m flask run --host=0.0.0.0 --port=80 &> /dev/null &