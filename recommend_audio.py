from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

TRACK_IDS = [
    "15taea6JZLAfG5OX0pzfG6",  # Make Believe
    "07q0QVgO56EorrSGHC48y3",  # I Was Made For Lovin' You
    "0DnA2DyUwy0W4xMYtRxocL",  # Who Can It Be Now?
]


def load_data(folder):
    tracks = pd.read_csv(
        folder / "tracks.csv",
        usecols=["track_id", "track_name", "track_artist"],
        dtype={"track_id": str},
        keep_default_na=False,
    )
    audio = pd.read_csv(folder / "audio_features.csv", dtype={"track_id": str})
    for table in (tracks, audio):
        if table.track_id.isna().any() or table.track_id.eq("").any():
            raise ValueError("Есть пустые track_id")
        if not table.track_id.is_unique:
            raise ValueError("Есть повторяющиеся track_id")
    if set(tracks.track_id) != set(audio.track_id):
        raise ValueError("Наборы track_id в двух файлах не совпадают")
    tracks = tracks.merge(audio, on="track_id", validate="one_to_one")
    tracks = tracks.sort_values("track_id").set_index("track_id")
    if not np.isfinite(tracks[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("В аудиопризнаках есть пропуски или бесконечные значения")

    artist = (
        tracks.track_artist.str.casefold()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    title = (
        tracks.track_name.str.casefold()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    tracks["song"] = artist + " | " + title
    return tracks


def find_similar(tracks, scaled_features, track_id, k=5):
    position = tracks.index.get_loc(track_id)
    distances = np.linalg.norm(scaled_features - scaled_features[position], axis=1)
    result = tracks[["track_artist", "track_name", "song"]].copy()
    result["distance"] = distances
    result = result[result.song != tracks.loc[track_id, "song"]]
    result = result.sort_values("distance", kind="stable")
    result = result.drop_duplicates("song").head(k)
    return result.drop(columns="song")


def main():
    folder = Path(__file__).resolve().parent / "data"
    tracks = load_data(folder)
    scaled_features = StandardScaler().fit_transform(tracks[FEATURES])
    for track_id in TRACK_IDS:
        track = tracks.loc[track_id]
        print(f"\n{track.track_artist} — {track.track_name}")
        similar = find_similar(tracks, scaled_features, track_id)
        for rank, row in enumerate(similar.itertuples(), start=1):
            print(f"{rank}. {row.track_artist} — {row.track_name} ({row.distance:.3f})")


if __name__ == "__main__":
    main()
