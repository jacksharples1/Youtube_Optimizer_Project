from google.cloud import storage
from tqdm import tqdm
import numpy as np
import cv2

def pull_images(bucket_name, dataset, videos_to_pick):
    bucket = storage.Client().get_bucket(bucket_name)
    images = []
    ids = []
    for blob in tqdm(bucket.list_blobs(prefix = dataset)):
        long_filename = blob.name
        filename = long_filename.strip(f'{dataset}/')
        last_underscore = filename.rfind('_')
        video_id = filename[0:last_underscore]
        if filename in videos_to_pick:
            array = np.array(cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), -1))
            if array.shape == (180, 320, 3):
                images.append(array)
                ids.append(video_id)
    return images, ids
