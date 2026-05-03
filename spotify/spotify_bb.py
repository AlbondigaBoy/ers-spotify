import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
import requests
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk


@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def artist_id_in_list(artist_ids_str: str, target_id: str) -> bool:
    try:
        return target_id in ast.literal_eval(artist_ids_str)
    except:
        return False

@st.cache_data
def get_lyrics_df(df):
    def get_lyrics(artist, title):
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return r.json().get("lyrics", "")
            return ""
        except:
            return ""

    df = df.copy()
    df['lyrics'] = df['name'].apply(lambda x: get_lyrics('Bad Bunny', x))
    return df

def get_sentiment(text):
    if not text:
        return 0
    return TextBlob(text).sentiment.polarity


def main():

    st.set_page_config(page_title="Bad Bunny Analysis", layout="wide")
    sns.set_theme(style='whitegrid', palette='tab10')

    st.title("🎧 Bad Bunny Discography Analysis")

    nltk.download('stopwords')

    DATA_FILE = 'tracks_features.csv'
    df = load_data(DATA_FILE)


    TARGET_ARTIST_ID = '4q3ewBCX7sLwd24euuV69X'

    artist_df = df[df['artist_ids'].apply(
        lambda x: artist_id_in_list(str(x), TARGET_ARTIST_ID)
    )].copy()

    artist_df['short_album_name'] = artist_df['album'].str.split('(').str[0].str.strip()

    STUDIO_ALBUMS = [
        'X 100PRE',
        'YHLQMDLG',
        'EL ÚLTIMO TOUR DEL MUNDO'
    ]

    artist_df = artist_df[artist_df['short_album_name'].isin(STUDIO_ALBUMS)]
    artist_df = artist_df.drop_duplicates(subset=['short_album_name', 'name'])

    ALBUM_ORDER = (
        artist_df
        .drop_duplicates('short_album_name')
        .sort_values('year')['short_album_name']
        .tolist()
    )


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Canciones por álbum")
        track_counts = artist_df.groupby('short_album_name')['name'].count().loc[ALBUM_ORDER]

        fig, ax = plt.subplots(figsize=(5, 3))
        track_counts.plot(kind='barh', ax=ax)
        ax.set_xlabel("Nº canciones")
        st.pyplot(fig)

    with col2:
        st.subheader("Acousticness vs Valence")

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(
            data=artist_df,
            x='valence',
            y='acousticness',
            hue='short_album_name',
            ax=ax
        )
        ax.legend(fontsize=7)
        st.pyplot(fig)

    st.subheader("Audio Features por Álbum")

    RADAR_FEATURES = [
        'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'valence'
    ]

    album_means = artist_df.groupby('short_album_name')[RADAR_FEATURES].mean().loc[ALBUM_ORDER]

    album_norm = (album_means - album_means.min()) / (
        album_means.max() - album_means.min() + 1e-9
    )

    N = len(RADAR_FEATURES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = sns.color_palette('tab10', n_colors=len(album_norm))

    for (album, row), color in zip(album_norm.iterrows(), colors):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=album)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_FEATURES)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)

    if st.button("Cargar letras"):
        artist_df = get_lyrics_df(artist_df)

    if 'lyrics' in artist_df.columns:

        artist_df['sentiment'] = artist_df['lyrics'].apply(get_sentiment)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Sentiment por álbum")

            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(
                data=artist_df,
                x='sentiment',
                y='short_album_name',
                order=ALBUM_ORDER,
                ax=ax
            )
            st.pyplot(fig)

        with col4:
            st.subheader("Valence vs Sentiment")

            fig, ax = plt.subplots(figsize=(5, 3))
            sns.scatterplot(
                data=artist_df,
                x='valence',
                y='sentiment',
                hue='short_album_name',
                ax=ax
            )
            ax.legend(fontsize=7)
            st.pyplot(fig)

        st.subheader("WordCloud")

        stopwords_es = set(stopwords.words('spanish'))
        stopwords_en = set(stopwords.words('english'))

        custom_stopwords = {
            'yeah', 'eh', 'oh', 'pa\'', 'na', 'nah',
            'na\'', 'to\'', 'te', 'yo', 'ey', 'yeh',
            'si', 'prr', 'uh', 'wuh', 'lo\'', 'ah'
        }

        all_stopwords = stopwords_es.union(stopwords_en).union(custom_stopwords)

        text = " ".join(artist_df['lyrics'].dropna())

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=all_stopwords
        ).generate(text)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud)
        ax.axis('off')

        st.pyplot(fig)

    st.subheader("Datos (canciones)")
    st.dataframe(artist_df[['name', 'short_album_name', 'year']])


if __name__ == "__main__":
    main()