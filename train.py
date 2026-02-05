from lightning.pytorch.callbacks import Callback
import os
import hydra
from src.Motif_DiffMol.model import Motif_DiffMol
from src.Motif_DiffMol.utils.utils_data import get_dataloader, get_last_checkpoint

@hydra.main(version_base=None,
            config_path="configs/",
            config_name="base",
            )
def train(config):
    model = Motif_DiffMol(config)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f'model #parameters: {sum(p.numel() for p in model.parameters())}, '
              f'#trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    ckpt_path = get_last_checkpoint(config.callbacks.step_ckpt.dirpath)
    current_step = 0
    if ckpt_path:
        try:
            current_step = int(ckpt_path.split("step-")[-1].split(".ckpt")[0])
        except Exception:
            current_step = 0

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    train_dataloader = get_dataloader(
        config,
        global_step=current_step,
        rank=rank,
        world_size=world_size
    )

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=hydra.utils.instantiate(config.callback),
        enable_progress_bar=True)

    if rank == 0:
        print(f"Dataloader prepared: Rank {rank}/{world_size}, Skipping {current_step} steps.")

    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    train()

