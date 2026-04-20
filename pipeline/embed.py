"""Embed reviews locally, reduce to 2D with UMAP, cluster with KMeans.

Run with: uv run --with sentence-transformers --with umap-learn --with scikit-learn --with numpy embed.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEWS_DIR = REPO_ROOT / "reviews"
OUT = REPO_ROOT / "docs" / "data.json"


def parse_review(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    # Strip leading blank lines
    text = raw.strip()
    # First non-empty line — usually the header/title
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header = lines[0] if lines else path.stem

    # Heuristic: pull paper title from header like
    # "Omri and Mike's Daily Paper: 27.06.2024 Agent-as-a-Judge: Evaluate ..."
    title = re.sub(r"^#+\s*", "", header)
    m = re.search(r"\d{2}\.\d{2}\.\d{4}\s+(.*)", title)
    if m:
        title = m.group(1).strip()
    # Date from filename
    m2 = re.search(r"(\d{4})_(\d{2})_(\d{2})", path.stem)
    date = f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}" if m2 else ""

    body = "\n".join(lines[1:]).strip()
    # Short preview for the card
    preview = body[:400].replace("\n", " ")
    return {
        "file": path.name,
        "title": title[:200],
        "date": date,
        "preview": preview,
        "body": body,
        "n_chars": len(body),
    }


def main():
    files = sorted(REVIEWS_DIR.glob("Review_*.md"))
    print(f"Parsing {len(files)} reviews...")
    reviews = [parse_review(p) for p in files]

    # Embedding text = title + body (capped)
    texts = [(r["title"] + "\n\n" + r["body"])[:8000] for r in reviews]

    print("Loading model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding...")
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    print(f"Embeddings shape: {embs.shape}")

    print("UMAP 3D...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.2,
        n_components=3,
        metric="cosine",
        random_state=42,
    )
    xyz = reducer.fit_transform(embs)

    # Center around 0 and scale so the cloud roughly fits in a unit sphere
    xyz = xyz - xyz.mean(axis=0)
    radius = np.linalg.norm(xyz, axis=1).max()
    xyz = xyz / (radius + 1e-9)

    print("Clustering (KMeans)...")
    n_clusters = 10
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(embs)

    # Per-cluster keyword labels via TF-IDF
    print("Labeling clusters with TF-IDF keywords...")
    docs_by_cluster = [[] for _ in range(n_clusters)]
    for c, r in zip(clusters, reviews):
        docs_by_cluster[c].append(r["title"] + " " + r["body"])

    extra_stop = [
        "model", "models", "paper", "authors", "approach", "method",
        "results", "work", "figure", "table", "section", "using", "use",
        "uses", "used", "new", "propose", "proposed", "different",
        "also", "shows", "show", "like", "based", "way", "ways",
        "way", "simply", "instead", "trained", "training",
    ]
    import sklearn.feature_extraction.text as _fe
    stop = list(_fe.ENGLISH_STOP_WORDS) + extra_stop
    vec = TfidfVectorizer(
        max_features=4000,
        stop_words=stop,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.75,  # drop near-universal terms
    )
    all_docs = [" ".join(d) for d in docs_by_cluster]
    tfidf = vec.fit_transform(all_docs)
    terms = np.array(vec.get_feature_names_out())
    cluster_labels = []
    for i in range(n_clusters):
        row = tfidf[i].toarray().ravel()
        top = row.argsort()[::-1][:5]
        # Prefer bigrams over repeated unigrams
        picked = []
        seen_tokens = set()
        for idx in top:
            term = terms[idx]
            toks = term.split()
            if any(t in seen_tokens for t in toks):
                continue
            picked.append(term)
            seen_tokens.update(toks)
            if len(picked) == 3:
                break
        cluster_labels.append(", ".join(picked) if picked else terms[top[0]])
        print(f"  Cluster {i} ({(clusters == i).sum()} reviews): {cluster_labels[i]}")

    # Build JSON for the viz
    points = []
    for i, r in enumerate(reviews):
        points.append(
            {
                "id": i,
                "x": float(xyz[i, 0]),
                "y": float(xyz[i, 1]),
                "z": float(xyz[i, 2]),
                "cluster": int(clusters[i]),
                "title": r["title"],
                "date": r["date"],
                "preview": r["preview"],
                "body": r["body"],
                "file": r["file"],
            }
        )

    data = {
        "points": points,
        "clusters": [
            {"id": i, "label": cluster_labels[i], "count": int((clusters == i).sum())}
            for i in range(n_clusters)
        ],
    }

    OUT.write_text(json.dumps(data, ensure_ascii=False))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
