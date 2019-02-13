#!/usr/bin/env bash
 
task(){
	CUDA_VISIBLE_DEVICES=0 python main_gt.py --train --window "$1";
}


for thing in 75 150 300 600 1200 2400 4800 9600 19200 'Inf'; do
	task "$thing" & 
done
