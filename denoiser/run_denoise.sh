#!/bin/bash
# echo "running...."
# prex="speaker_4"
# for i in {0..1270}
#     do 
#         name="$i"
#         # echo "make dir $name"
#         # mkdir ./denoiser/$name
#         python -m denoiser.enhance --dns64 --noisy_dir=./data/$name/ --out_dir=./result
#         # echo "copy txt"
#         # cp ./a/$name/*.txt ./result/$name/

#     done
python -m denoiser.enhance --dns64 --noisy_dir='/mnt/c/Users/phudh/Desktop/src/dialog_system/denoiser/input' --out_dir='/mnt/c/Users/phudh/Desktop/src/dialog_system/denoiser/result'