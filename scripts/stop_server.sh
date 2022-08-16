#!/bin/bash
ps -A | grep python3 | awk '{print $1}' | xargs kill -9 || :