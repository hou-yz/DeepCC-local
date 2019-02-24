#!/usr/bin/env bash
 
task(){
	CUDA_VISIBLE_DEVICES=1 python main.py --train --window "$1" --mot --combine-trainval;
}


for thing in 15 30 75 'Inf'; do
	task "$thing" & 
done
