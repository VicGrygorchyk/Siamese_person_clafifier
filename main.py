import os

from torch import cuda
from torch import compile as torch_compile
from mlflow import start_run
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint, EarlyStopping

from siamese.trainer_mng import ModelTrainingWrapper

EPOCH = 10
if epochs := os.getenv('EPOCH'):
    EPOCH = int(epochs)
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")


if __name__ == "__main__":
    cuda.empty_cache()
    # dataset

    with start_run(description=f"Run {EPOCH} epochs, BCElogits"):
        model_wrapped = ModelTrainingWrapper()
        # model_wrapped = torch_compile(model_wrapped)

        trainer = pl.Trainer(
            min_epochs=1,
            max_epochs=EPOCH,
            accumulate_grad_batches=4,
            log_every_n_steps=10,
            callbacks=[
                StochasticWeightAveraging(swa_lrs=1e-2),
                ModelCheckpoint(dirpath=SAVE_MODEL_PATH, save_top_k=3, monitor="eval_loss"),
                # EarlyStopping(monitor='eval_loss', patience=10)
            ],
            default_root_dir=SAVE_MODEL_PATH
        )

        tuner = Tuner(trainer)
        if eval(os.getenv("FIND_BATCH_SIZE")):
            tuner.scale_batch_size(model_wrapped, mode='binsearch')
        if eval(os.getenv("FIND_LR_RATE")):
            tuner.lr_find(model_wrapped, num_training=50)

        trainer.fit(model=model_wrapped)
        trainer.test(model_wrapped)
