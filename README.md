# SonicLens

This folder contains the deployed SonicLens web app.

The website is a static, browser-based interface for exploring song similarity through precomputed audio embeddings and interactive song tools. It is organized around a few main experiences:

## What The Website Includes

### Artist and library browsing

The top bar lets you switch between artists in the dataset, then move across the main views for clusters, similarity search, comparison, heatmaps, and uploads.

### Cluster view

This view shows how songs group together based on their stem-level embeddings. It is meant for quickly seeing nearby tracks and broader structure in the library.

### Similarity search

Choose a song and inspect its nearest neighbors. The UI is built for finding songs that sound close to a selected track and for comparing the strongest matches.

### Multi-song comparison

The compare tab lets you select 2 to 5 songs and view a pairwise similarity matrix. Individual cells can be opened for more detailed stem-level breakdowns.

### Heatmap view

The heatmap provides a compact visual summary of similarity across songs, making it easier to spot clusters and outliers at a glance.

### Upload pipeline

The upload tab walks through adding a song to the library. The flow supports either a YouTube URL or an audio file, then trimming, stem separation with Demucs, and embedding generation before the song is added to the library.

### Local sample data

The deployed frontend reads from the `data/` folder, which contains artist metadata, song JSON files, and embedding vectors for the included artists.

## Main Files

- `index.html` - page structure and all major app panels
- `styles.css` - visual styling for the deployed UI
- `app.js` - client-side behavior and data loading
- `data/` - sample data and embeddings used by the UI
