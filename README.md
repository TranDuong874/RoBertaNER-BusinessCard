# RoBertaNER-BusinessCard

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/TranDuong874/RoBertaNER-BusinessCard.git
   cd RoBertaNER-BusinessCard
   ```

2. **Install dependencies**
   ```bash
    !pip install -U transformers datasets seqeval evaluate accelerate bitsandbytes sklearn-crfsuite -q
   ```

3. **Run the training script**
   ```bash
   python train.py training_config.yaml
   ```

---

- For multi-GPU training, use:
  ```bash
  python -m torch.distributed.run --nproc_per_node=NUM_GPUS train.py training_config.yaml
  ```
  Replace `NUM_GPUS` with the number of GPUs you want to use.
