CUDA_VISIBLE_DEVICES="4" python train.py --fold 4 --model_type "vit_large_patch32_384" --height 384 --width 384 --seed 666 --batch_size 32 --accumulation_steps 2
