import glob
from pathlib import Path
import os


def main(path_to_file):
    for folder in glob.glob(f"{path_to_file}/*"):
        for img in glob.glob(f"{folder}/*"):

            if " " in img.split('/')[-1]:
                try:
                    new_path_label = img.replace(' ', '_')
                    os.rename(img, new_path_label)

                except FileNotFoundError as exc:
                    # folder have been already renamed
                    # FIXME: make iterator which removes already renamed examples
                    print(exc)
                    pass


if __name__ == "__main__":
    labels_path = os.getenv('LABELED_CSL_DS_TRAIN')
    main(labels_path)
