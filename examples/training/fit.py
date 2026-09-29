"""Run a Lightning fit, keep the best validation epoch and return the learning curve."""

from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from matgl.graph.data import MGLDataLoader


def loaders(train, val, test, collate_fn, batch_size: int):
    return MGLDataLoader(
        train_data=train, val_data=val, test_data=test,
        collate_fn=collate_fn, batch_size=batch_size, num_workers=0,
    )


def fit(module, train_loader, val_loader, *, epochs: int, outdir: Path, name: str, accelerator: str, seed: int):
    L.seed_everything(seed, workers=True)
    logger = CSVLogger(outdir / "logs", name=name, version=0)
    best = ModelCheckpoint(dirpath=outdir / "checkpoints" / name, monitor="val_Total_Loss", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1, logger=logger, callbacks=[best],
        precision="64-true" if torch.get_default_dtype() == torch.float64 else "32-true",
        inference_mode=False, enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1,
    )
    trainer.fit(module, train_loader, val_loader)
    state = torch.load(best.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state)
    return curve(Path(logger.log_dir) / "metrics.csv")


def curve(metrics_csv: Path) -> dict[str, list[float]]:
    frame = pd.read_csv(metrics_csv).groupby("epoch").first()
    keep = [c for c in frame.columns if c.endswith("MAE") and (c.startswith("train_") or c.startswith("val_"))]
    return {c: frame[c].dropna().round(6).tolist() for c in keep}
