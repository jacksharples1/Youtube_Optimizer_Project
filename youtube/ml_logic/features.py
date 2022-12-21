import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
import isodate

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """
    def duration_to_seconds(x):
        if x == 'P0D':
            return None
        elif type(x) == float:
            return None
        else:
            try:
                return isodate.parse_duration(x).total_seconds()
            except:
                print(x)

    try:
        df['duration'] = df['duration'].apply(duration_to_seconds)
    except:
        pass

    def publish_to_date(x):
        x = x.replace('Z',"")
        x = x.replace('T'," ")
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    df['published'] = df['published'].apply(publish_to_date)

    def hours(x):
        y=x.hour
        return y

    def days(x):
        y=x.weekday()
        return y

    def months(x):
        y=x.month
        return y

    hours = df['published'].apply(hours)
    days = df['published'].apply(days)
    months = df['published'].apply(months)

    df_new_joined = pd.concat([hours,days,months], axis=1)
    df_new_joined.columns = ['hours','days','months']

    hours_in_day = 24
    days_in_week = 7
    months_in_year = 12

    df_new_joined['sin_hours'] = np.sin(2*np.pi*(df_new_joined['hours']-1)/hours_in_day)
    df_new_joined['cos_hours'] = np.cos(2*np.pi*(df_new_joined['hours']-1)/hours_in_day)
    df_new_joined['sin_day_of_week'] = np.sin(2*np.pi*(df_new_joined['days']-1)/days_in_week)
    df_new_joined['cos_day_of_week'] = np.cos(2*np.pi*(df_new_joined['days']-1)/days_in_week)
    df_new_joined['sin_months'] = np.sin(2*np.pi*(df_new_joined['months']-1)/months_in_year)
    df_new_joined['cos_months'] = np.cos(2*np.pi*(df_new_joined['months']-1)/months_in_year)

    df_new_joined.drop(columns=['hours'], inplace=True)
    df_new_joined.drop(columns=['days'], inplace=True)
    df_new_joined.drop(columns=['months'], inplace=True)

    df = df.join(df_new_joined)

    from sklearn.impute import SimpleImputer
    imputer_mean = SimpleImputer(strategy="mean")
    try:
        df['duration'] = imputer_mean.fit_transform(df[['duration']])
    except:
        pass

    imputer_mean = SimpleImputer(strategy="mean")
    try:
        df['duration'] = imputer_mean.fit_transform(df[['channel_subscriberCount']])
    except:
        pass

    imputer_mode = SimpleImputer(strategy="most_frequent")
    try:
        df['category_id'] = imputer_mode.fit_transform(df[['category_id']])
    except:
        pass
    return df
