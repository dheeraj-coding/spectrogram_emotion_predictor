import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import SpotifyException
import requests
from tqdm import tqdm

load_dotenv(os.path.join(os.getcwd(), '.env'))

SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
DATA_COLUMNS = ['spotify_id', 'valence', 'arousal', 'dominance']


def download_song(spotify_id, url):
    r = requests.get(url, allow_redirects=True)
    open(f'./dataset/songs/{spotify_id}.mp3', 'wb').write(r.content)


def main():
    # Initialize Spotify API
    client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                          client_secret=SPOTIFY_CLIENT_SECRET)
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    spotify.max_retries = 10
    spotify.backoff_factor = 0.4
    spotify.retries = 10

    df = pd.read_csv('muse_v3.csv').astype(
        {'valence_tags': np.float32, 'arousal_tags': np.float32, 'dominance_tags': np.float32})
    df = df.dropna(subset=['spotify_id'])

    output_df = pd.DataFrame(columns=DATA_COLUMNS)

    avail_markets = spotify.available_markets()['markets']
    df.reset_index()
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        track = spotify.track(row['spotify_id'], market=avail_markets)
        url = track.get('preview_url')
        if url is not None:
            song_info = {'spotify_id': row['spotify_id'], 'valence': row['valence_tags'],
                         'arousal': row['arousal_tags'], 'dominance': row['dominance_tags']}
            df_row = pd.DataFrame(song_info, index=[0])
            output_df = pd.concat([output_df, df_row], ignore_index=True)
            download_song(row['spotify_id'], url)
    output_df.to_csv('./dataset/muse.csv', sep=',', index=False)
    print("Total datapoints: ", output_df.shape[0])


if __name__ == "__main__":
    main()
