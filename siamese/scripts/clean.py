import os
import glob

from siamese.preprocess import image_helper


all_imgs_path = glob.glob('/home/mudro/Downloads/refined/*')

target_imgs_path = glob.glob('/home/mudro/PycharmProjects/siamese/xxx/*')

# load targets
target_imgs = [image_helper.load_image(path_) for path_ in target_imgs_path]

deleted = 0

for img_path in all_imgs_path:
    img = image_helper.load_image(img_path)

    for target in target_imgs:
        diff = image_helper.get_image_difference(img, target)
        if diff <= 0.09:
            try:
                os.remove(img_path)
                deleted += 1
                continue
            except Exception as exc:
                print(exc)
                pass

print(f"DELETED {deleted}")
