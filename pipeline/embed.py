"""Embed reviews locally, reduce to 3D with UMAP, cluster with KMeans,
label clusters with Gemma 4 E2B via Ollama.

Run with:
    uv run --python 3.12 \
      --with sentence-transformers --with umap-learn --with scikit-learn --with numpy \
      embed.py

Requires Ollama running locally with gemma4:e2b pulled.
"""
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"


def gemma_label_cluster(bullets: str, n_members: int) -> str:
    """Ask Gemma for a 3-6 word human label for a cluster of paper reviews.

    Gemma 4 is a thinking model — MUST use /api/chat with "think": false,
    otherwise all tokens go to hidden reasoning and the response is empty.
    """
    system = (
        "You label clusters of AI / ML paper-review summaries. You will be "
        "shown a sample of reviews in one cluster. Return ONLY a 3-to-6 word "
        "label capturing the shared research theme — concrete domain or "
        "method, not fluff. Nearly every review is about large language "
        "models, so NEVER start the label with 'LLM' or 'Large Language "
        "Model' — lead with the specific technique, sub-area, or domain. "
        "No quotes, no trailing period."
    )
    user = (
        f"Cluster of {n_members} paper reviews. Sample titles + leads:\n\n"
        f"{bullets}\n\n"
        "Label this cluster in 3-6 words:"
    )
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": False,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 32},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("message") or {}).get("content", "").strip()
    text = text.strip().strip('"\'').strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,;:")
    text = text.splitlines()[0] if text else "cluster"
    return text[:80] if text else "cluster"

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

    # LLM cluster labels — Gemma 4 E2B via Ollama, /api/chat with think:false
    print(f"Labeling clusters with {OLLAMA_MODEL} via Ollama...")
    cluster_labels = []
    for i in range(n_clusters):
        idxs = np.where(clusters == i)[0]
        centroid = embs[idxs].mean(axis=0)
        dists = np.linalg.norm(embs[idxs] - centroid, axis=1)
        order = idxs[np.argsort(dists)]
        sample_idxs = order[: min(25, len(order))]
        # For each sampled review: title + first ~350 chars of body
        bullets = "\n".join(
            f"- {reviews[j]['title']}: {reviews[j]['body'][:350].replace(chr(10), ' ')}"
            for j in sample_idxs
        )
        label = gemma_label_cluster(bullets, len(idxs))
        cluster_labels.append(label)
        print(f"  Cluster {i} ({len(idxs)} reviews): {label}")

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
