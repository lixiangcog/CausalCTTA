#!/bin/bash

#Please modify the following roots to yours.
dataset_root=./Fundus
model_root=./models/
path_save_log=./logs/

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=RIM_ONE_r3

#Optimizer
optimizer=Adam
lr=0.05

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5

# Only need to care about the prompt,AdaBN and clip settings

use_prompt=true # Set to true to use the prompt, false otherwise
use_vida=false    # Set to true to use the VIDA modules, false otherwise
use_dropout=false # Set to true to use MC-Dropout, false otherwise
use_AdaBN=true  # Set to true to use AdaBN, false otherwise
use_trans_input=false # Set to true to use data augmentation, false otherwise
clip=false # Set to true to use the CLIP-based loss, false otherwise

if [ "$use_prompt" = true ]; then
    use_prompt_flag="--use_prompt"
else
    use_prompt_flag=""
fi

if [ "$use_AdaBN" = true ]; then
    use_AdaBN_flag="--use_AdaBN"
else
    use_AdaBN_flag=""
fi

if [ "$clip" = true ]; then
    clip_flag="--clip"
else
    clip_flag=""
fi

#Command
cd example
CUDA_VISIBLE_DEVICES=1 python c2tta.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
 $use_prompt_flag \
 $use_AdaBN_flag \
 $clip_flag \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n
