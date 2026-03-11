CKPT_PATH=checkpoints/lr1e-5_bs2_ep200_eot_urdf-params/epoch_200.pth # TODO: change to your checkpoint path

python scripts/inference.py \
    --image_path assets/display.jpg \
    --model_path $CKPT_PATH \
    --in_the_wild \
    --output_dir outputs/display

python scripts/inference.py \
    --image_path assets/laptop.jpg \
    --model_path $CKPT_PATH \
    --in_the_wild \
    --output_dir outputs/laptop

python scripts/inference.py \
    --image_path assets/faucet.jpg \
    --model_path $CKPT_PATH \
    --in_the_wild \
    --output_dir outputs/faucet
