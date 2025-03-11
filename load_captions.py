import os

def load_captions(images_dir, captions_dir):
    captions = {}
    for image_file in os.listdir(images_dir):
        image_id = image_file.split(".")[0]
        image_caption = os.path.join(captions_dir, image_id + ".txt")
        with open(image_caption) as f:
            caption = f.readlines()[0].strip()
            if image_id not in captions:
                captions[image_id] = caption
    return captions



