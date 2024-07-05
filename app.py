# import required dependencies
import streamlit as st
import googleapiclient.discovery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
import re
from textblob import TextBlob
from PIL import Image
from pytube import YouTube
import os
from dotenv import load_dotenv

# Streamlit page configuration
st.set_page_config(page_title = 'YouTube Sentiment Analyzer',
                   page_icon = ':double_vertical_bar:',
                   layout = 'centered',
                   initial_sidebar_state = 'auto')


st.title('YouTube Sentiment Analyzer')
st.sidebar.title('User Input')
#key = st.sidebar.text_input('Enter Your Developer Key: ')
url = st.sidebar.text_input('Enter YouTube Video URL')
submit = st.sidebar.button('Analyze')

load_dotenv()
key = os.getenv('DEVELOPER_KEY')

# function to collect the comments and return in a dataframe
def extract_comments(videoId, key):

    api_service_name = 'youtube'
    api_version = 'v3'
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=key)

    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
        part="snippet",
        videoId=videoId,
        maxResults=100,
        pageToken=next_page_token
    )
        
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])
        
        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
    return df

# function to obtain YouTube video metadata
def get_video_metadata(url):
    video = YouTube(url)

    title = video.title
    publish_date = video.publish_date
    views = video.views
    length = video.length
    author = video.author

    metadata = [title, publish_date, views, length, author]

    return metadata

# function to remove emoji [obtained from stackoverflow: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python]
def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

# function to perform text processing
def text_processing(text):
    pattern = r'[^\w\s]'
    stopwords_list = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    pure_text = remove_emojis(text)                                                    # remove emojis in the text
    cleaned_text = re.sub(pattern,'', pure_text)                                       # remove any non character, non-space
    lemmatized_text = lemmatizer.lemmatize(cleaned_text)                               # lemmatization
    tokens = word_tokenize(lemmatized_text)                                            # tokenization
    fil_tokens = [token for token in tokens if token.lower() not in stopwords_list]    # stopwords removal
    processed_text = ' '.join(fil_tokens)                                              # concatenate all token into string of text
    return processed_text

# function to compute the polarity of the text
def polarity(text):
    polarity = TextBlob(text).sentiment.polarity
    return polarity
    
# function to categorize the sentiment type 
def sentiment_analysis(score):
    if score == 0:
        return 'Neutral'
    elif score > 0:
        return 'Positive'
    else:
        return 'Negative'

# function to compute the subjectivity of the text
def subjectivity(text):
    subjectivity = TextBlob(text).sentiment.subjectivity
    return subjectivity

# function to plot word cloud
def generate_wordcloud(data):
    combined_text = ' '.join(data['processed_text'])
    
    wordcloud = WordCloud(width = 500, height = 500,
                            background_color ='white',
                            colormap = 'magma_r',
                            min_font_size = 15).generate(combined_text)

    fig = plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=2)
    return fig

# function to plot bar chart --- polarity
def sentiment_count(data):
    count = data['sentiment'].value_counts()
    count_df = pd.DataFrame(count).reset_index()
    count_df.columns = ['Sentiment','Count']

    fig = px.bar(count_df, x='Sentiment', y='Count')
    fig.update_traces(marker_color='#FFA500')
    return fig

# function to plot histogram --- polarity score
def polarity_dist(data):
    fig = px.histogram(data, x='polarity_score', template='plotly_dark')
    fig.update_layout(
        xaxis = dict(
            title = 'Polarity Score'
        ),
        yaxis = dict(
            title = 'Frequency'
        )
    )
    fig.update_traces(marker_color='red')
    return fig

# function to plot histogram --- subjectivity score
def subjectivity_dist(data):
    fig = px.histogram(data, x='subjectivity_score', template='plotly_dark')
    fig.update_layout(
        xaxis = dict(
            title = 'Subjectivity Score'
        ),
        yaxis = dict(
            title = 'Frequency'
        )
    )
    fig.update_traces(marker_color='#9932CC')
    return fig



subjectivity_desc = '''
Subjectivity lies between 0 to 1. 
Subjectivity quantifies the amount of personal opinion and factual information contained in the text. 
The higher subjectivity means that the text contains personal opinion rather than factual information.
'''

polarity_desc = '''
Polarity scores can range from -1 to 1, with -1 indicating highly negative sentiment and +1 indicating highly positive sentiment. 
A score of 0 represents neutral sentiment.
'''

sentiment_desc = '''
Based on the polarity scores, categorize each comment into one of the three sentiment categories:
- Positive: Comments with a polarity score greater than 0.
- Neutral: Comments with a polarity score of exactly 0.
- Negative: Comments with a polarity score less than 0.
'''

wordcloud_desc = '''
Wordcloud is a visual representation of word frequency and value. 
It provides instant insight into the most important/frequent terms in text data.
'''

if submit:
    if url:
        videoId = url.split('v=')[1].split('&')[0]
        data = extract_comments(videoId, key)
        metadata = get_video_metadata(url)
        comments_num = data.shape[0]

        data_copy = data.copy()
        data_copy['processed_text'] = data_copy['text'].apply(lambda x: text_processing(x))
        data_copy['polarity_score'] = data_copy['text'].apply(lambda x: polarity(x))
        data_copy['sentiment'] = data_copy['polarity_score'].apply(lambda x: sentiment_analysis(x))
        data_copy['subjectivity_score'] = data_copy['text'].apply(lambda x: subjectivity(x))

        container = st.container(border=True)
        container.markdown(f'### {metadata[0]}')
        container.write(f'Publish on: {metadata[1]}')
        container.image(f"http://img.youtube.com/vi/{videoId}/0.jpg", use_column_width=True)
        container.write(f':timer_clock: Length of video: {round(metadata[3]/60)} minutes')
        container.write(f':male-technologist: Author: {metadata[4]}')
        container.write(f':film_projector: Number of Views: {metadata[2]}')
        container.markdown(f'#### Total Number of Comments: {comments_num}')

        container1 = st.container(border=True)
        hist = polarity_dist(data_copy)
        container1.markdown('<h3 style="color:red;"> Polarity Score </h3>', unsafe_allow_html=True)
        container1.markdown(polarity_desc, unsafe_allow_html=True)
        container1.plotly_chart(hist)

        container2 = st.container(border=True)
        bar = sentiment_count(data_copy)
        container2.markdown('<h3 style="color:#FFA500;"> Positive, Neutral & Negative Sentiments </h3>', unsafe_allow_html=True)
        container2.markdown(sentiment_desc, unsafe_allow_html=True)
        container2.plotly_chart(bar)

        container3 = st.container(border=True)
        hist = subjectivity_dist(data_copy)
        container3.markdown('<h3 style="color:#9932CC;"> Subjectivity Score </h3>', unsafe_allow_html=True)
        container3.markdown(subjectivity_desc, unsafe_allow_html=True)
        container3.plotly_chart(hist)

        container4 = st.container(border=True)
        cloud = generate_wordcloud(data_copy)
        container4.markdown('<h3 style="color:navy;"> Keywords </h3>', unsafe_allow_html=True)
        container4.markdown(wordcloud_desc, unsafe_allow_html=True)
        container4.pyplot(cloud)


    else:
        st.warning('Please enter an YouTube video URL.')
