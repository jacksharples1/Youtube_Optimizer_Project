from langdetect import detect
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import string # "string" module is already installed with Python


'''drop duplicates'''
def drop_duplicate_ids(df):
    df = df.drop_duplicates(subset=['id'])
    return df

'''drop y nans'''
def drop_y_nas(df):
    df = df.dropna(subset=['views'])
    return df

'''detect with try-expect'''
def detect_except(x):
    try:
        lang = detect(x)
    except:
        lang = 'Other'

    return lang

'''first remove of non-english rows'''
def strip_english(df):
    '''basic strip'''
    df['detect'] = df['title'].apply(detect_except)

    df = df[df['detect'] == 'en']
    return df

'''reduce dataframe to title and views'''
def reduce_features(df):
    df = df.loc[:, ['title', 'views']]
    return df

'''completed dataframe preprocessing'''
def df_preprocess(df):
    df = drop_duplicate_ids(df)
    df = strip_english(df)
    df = reduce_features(df)
    return df

'''removes emojis'''
def emoji_strip(text):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

'''lemmatizing'''
def lemmatize(text):
    # Lemmatizing the verbs
    verb_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v") # v --> verbs
        for word in text
    ]

# 2 - Lemmatizing the nouns
    noun_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "n") # n --> nouns
        for word in text
    ]
    return ' '.join(text)

'''completed preprocessing. use with .apply() to title column'''
def preprocessing(sentence):
#     Lower
    sentence = sentence.lower()

    '''punctuation strip'''
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    '''strip numbers'''
    sentence = ''.join(char for char in sentence if not char.isdigit())

    '''emoji strip'''
    sentence = emoji_strip(sentence)

    '''remove chinese characters'''
    sentence=re.sub(u'(?<=[^0-9])[^\u4e00-\u9fff0-9a-zA-Z]+(?=[^0-9])',' ',sentence)

    '''tokenizer'''
    tokenized = word_tokenize(sentence)

    '''lemmatize'''
    clean_reviews = lemmatize(tokenized)

    return clean_reviews
