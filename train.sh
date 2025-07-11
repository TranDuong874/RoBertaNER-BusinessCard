# Run with single GPU
python train.py training_config.yaml 

# python -m torch.distributed.run --nproc_per_node=2 train.py training_config.yaml