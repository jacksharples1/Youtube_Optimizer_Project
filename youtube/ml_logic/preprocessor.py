import pandas as pd
from youtube.ml_logic.processnlp import (drop_duplicate_ids,strip_english,preprocessing,drop_y_nas)
from youtube.ml_logic.pull import pull_images
from youtube.ml_logic.params import (DATASET,BUCKET_NAME)

def preprocess_features():

    df = pd.read_csv(f"./raw_data/{DATASET}.csv")

    print(f'\nLength of original df = {len(df)}')

    df = drop_duplicate_ids(df)
    print(f'\nLength without duplicate ids = {len(df)}')

    print('\nFinding English titles...')

    df = strip_english(df)
    print(f'\nLength with only english = {len(df)}')

    df['title'] = df['title'].apply(lambda x: preprocessing(x))

    df = drop_y_nas(df)

    assert df.views.isna().sum() == 0

    df = df.sort_values('id')

    df["get"] =  + df["id"] +'_'+ df["views"].astype(int).astype(str)

    videos_to_pick = set(df['get'])

    print('\nPulling images...')
    images, ids = pull_images(BUCKET_NAME,DATASET,videos_to_pick)

    assert len(ids) == len(images)

    df= df[df['id'].isin(ids)]

    assert list(df.id) == ids

    print('\n✅ Images and NLP data is matched.')
    return df, images

