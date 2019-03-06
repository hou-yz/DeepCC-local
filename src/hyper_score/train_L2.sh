#!/usr/bin/env bash
 
task(){
	CUDA_VISIBLE_DEVICES=4 python main.py --train --window "$1" --triplet;
}


for thing in 75 150 300 600 1200 2400 4800 9600 19200 'Inf'; do
	task "$thing" & 
done
