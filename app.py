import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ----------------------
# LOAD AND PREP DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bollywood_cleaned.csv")
    df.columns = df.columns.str.strip().str.lower()
    df['song name'] = df['song name'].fillna("").astype(str)
    df['lyrics'] = df['lyrics'].fillna("").astype(str)
    return df

df = load_data()

# ----------------------
# TF-IDF VECTORIZATION
# ----------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['lyrics'])
cosine_sim = cosine_similarity(tfidf_matrix)

# ----------------------
# RECOMMENDATION FUNCTION
# ----------------------
def recommend(song_name, top_n=5):
    matches = get_close_matches(song_name, df['song name'], n=1, cutoff=0.6)
    if not matches:
        return None, []

    matched_song = matches[0]
    idx = df[df['song name'] == matched_song].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return matched_song, df.iloc[indices]

# ----------------------
# STREAMLIT UI
# ----------------------
st.title("🎵 Bollywood Music Recommender")
st.write("Discover similar Bollywood songs based on lyrics!")

# ----------------------
# SIDEBAR FILTER OPTIONS
# ----------------------
st.sidebar.header("🔍 Filter Options")

def get_unique_values(columns):
    values = []
    for col in columns:
        if col in df.columns:
            values.extend(df[col].dropna().astype(str).unique())
    return sorted(set(values))

# Sidebar dropdowns
selected_year = st.sidebar.selectbox("📅 Year", ["All"] + sorted(df['year'].dropna().astype(str).unique()))
selected_singer_filter = st.sidebar.selectbox("🎤 Singer", ["All"] + get_unique_values(['singer_1', 'singer_2', 'singer_3', 'singer_4', 'singer_5', 'singer_6', 'singer_7']))
selected_music_director = st.sidebar.selectbox("🎧 Music Director", ["All"] + get_unique_values(['music_director_1', 'music_director_2', 'music_director_3']))
selected_lyricist = st.sidebar.selectbox("✍️ Lyricist", ["All"] + get_unique_values(['lyricists_1', 'lyricists_2', 'lyricists_3', 'lyricists_4', 'lyricists_5', 'lyricists_6']))
selected_cast = st.sidebar.selectbox("🎬 Cast", ["All"] + get_unique_values([
    'cast_1', 'casts_2', 'casts_3', 'casts_4', 'casts_5', 'casts_6',
    'casts_7', 'casts_8', 'casts_9', 'casts_10', 'casts_11', 'casts_12',
    'casts_13', 'casts_14'
]))

# Apply initial filters to dropdown
filtered_df = df.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['year'].astype(str) == selected_year]
if selected_singer_filter != "All":
    filtered_df = filtered_df[filtered_df[['singer_1', 'singer_2', 'singer_3', 'singer_4', 'singer_5', 'singer_6', 'singer_7']]
                              .astype(str).apply(lambda row: selected_singer_filter in row.values, axis=1)]

song_list = sorted(filtered_df['song name'].dropna().unique())
selected_song = st.selectbox("🎵 Type or select a song", song_list)

# ----------------------
# Trigger Recommendation
# ----------------------
if st.button("🔍 Recommend"):
    if selected_song.strip() == "":
        st.warning("Please select a song.")
    else:
        matched_song, results = recommend(selected_song)
        if matched_song is None:
            st.error(f"❌ Song '{selected_song}' not found.")
        else:
            # Apply advanced filters to recommendations
            if selected_singer_filter != "All":
                results = results[
                    results[['singer_1', 'singer_2', 'singer_3', 'singer_4', 'singer_5', 'singer_6', 'singer_7']]
                    .astype(str).apply(lambda row: selected_singer_filter in row.values, axis=1)
                ]

            if selected_music_director != "All":
                results = results[
                    results[['music_director_1', 'music_director_2', 'music_director_3']]
                    .astype(str).apply(lambda row: selected_music_director in row.values, axis=1)
                ]

            if selected_lyricist != "All":
                results = results[
                    results[['lyricists_1', 'lyricists_2', 'lyricists_3', 'lyricists_4', 'lyricists_5', 'lyricists_6']]
                    .astype(str).apply(lambda row: selected_lyricist in row.values, axis=1)
                ]

            if selected_cast != "All":
                results = results[
                    results[['cast_1', 'casts_2', 'casts_3', 'casts_4', 'casts_5', 'casts_6',
                             'casts_7', 'casts_8', 'casts_9', 'casts_10', 'casts_11', 'casts_12',
                             'casts_13', 'casts_14']]
                    .astype(str).apply(lambda row: selected_cast in row.values, axis=1)
                ]

            st.success(f"✅ Songs similar to: **{matched_song}**")
            if results.empty:
                st.warning("No matching recommendations with applied filters.")
            else:
                for _, row in results.iterrows():
                    st.markdown(f"### 🎶 {row['song name']}")
                    if 'image_url' in row and pd.notna(row['image_url']):
                        st.image(row['image_url'], width=300)

                    if 'singer_1' in row and pd.notna(row['singer_1']):
                        st.caption(f"👤 Singer: {row['singer_1']}")
                    if 'music_director_1' in row and pd.notna(row['music_director_1']):
                        st.caption(f"🎧 Music Director: {row['music_director_1']}")
                    if 'year' in row and pd.notna(row['year']):
                        st.caption(f"📅 Year: {int(row['year'])}")
                    st.write(f"📝 *Lyrics Preview:* {row['lyrics'][:300]}...")
                    st.divider()
