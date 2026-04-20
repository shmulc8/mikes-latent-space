# Mike's Latent Space

Interactive 3D map of [Mike Erlihson's 236 paper reviews](https://github.com/merlihson/scientific-resources/tree/main/mike-paper-reviews-all), projected into a latent space.

**Live:** https://shmulc8.github.io/mikes-latent-space/

## How it works

1. Fetch the 236 English-language review markdowns from `merlihson/scientific-resources`.
2. Embed each review with `sentence-transformers/all-MiniLM-L6-v2` (384-d).
3. Reduce to 3D with UMAP (cosine metric).
4. KMeans (k=10) clusters the 384-d embeddings.
5. TF-IDF on cluster members generates short keyword labels.
6. Render with Three.js: glowing points, persistent nearest-neighbor web, click-to-read.

## Layout

```
docs/         static site (GitHub Pages serves from here)
pipeline/     embed.py — fetches, embeds, projects, writes docs/data.json
```

## Rebuild the data

```bash
# from repo root
mkdir -p reviews
gh api repos/merlihson/scientific-resources/contents/mike-paper-reviews-all/split-english-reviews-md \
  --jq '.[] | .download_url' \
  | xargs -n1 -P20 curl -s -O --output-dir reviews

cd pipeline
uv run --python 3.12 \
  --with sentence-transformers --with umap-learn --with scikit-learn --with numpy \
  embed.py
```
