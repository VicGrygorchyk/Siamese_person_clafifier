import os

from torch import cuda
from torch import compile as torch_compile
from mlflow import start_run
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint

from siamese.model import SiameseNN
from siamese.dataset import PersonsImages
from siamese.trainer_mng import ModelTrainingWrapper

EPOCH = 100
START_BATCH_SIZE = 2

DATASET_PATH = os.getenv("DATASET_PATH")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")


if __name__ == "__main__":
    cuda.empty_cache()
    # dataset
    dataset = PersonsImages(DATASET_PATH)

    with start_run(description=f"Run with epoch ${EPOCH} and BCE loss, small dataset"):
        model = SiameseNN()
        model_wrapped = ModelTrainingWrapper(
            model,
            START_BATCH_SIZE,
            dataset,
            save_chpt_path=SAVE_MODEL_PATH
        )
        # model_wrapped = torch_compile(model_wrapped)

        trainer = pl.Trainer(
            min_epochs=1,
            max_epochs=EPOCH,
            accumulate_grad_batches=64,
            callbacks=[
                StochasticWeightAveraging(swa_lrs=1e-2),
                ModelCheckpoint(dirpath=SAVE_MODEL_PATH, save_top_k=2, monitor="eval_loss")
            ],
            default_root_dir=SAVE_MODEL_PATH
        )

        tuner = Tuner(trainer)
        tuner.scale_batch_size(model_wrapped, mode='binsearch')
        # tuner.lr_find(model_wrapped)

        trainer.fit(model=model_wrapped)
        trainer.test(model_wrapped)
