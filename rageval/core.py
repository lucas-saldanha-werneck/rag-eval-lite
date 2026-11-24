import json
import math
import re
from collections import Counter

STOP = {"a", "an", "the", "of", "to", "in", "is", "are", "do", "does", "how", "i", "my",
        "can", "for", "and", "or", "on", "it", "be", "you", "your", "what", "when", "with"}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.ids = [c["id"] for c in chunks]
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        self.k1, self.b = k1, b
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=3):
        q = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scored.append((s, self.ids[i]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for s, cid in scored[:k] if s > 0]


def retrieval_metrics(index, questions, k=3):
    hits, rr, misses = 0, 0.0, []
    for q in questions:
        got = index.search(q["question"], k)
        if q["gold"] in got:
            hits += 1
            rr += 1.0 / (got.index(q["gold"]) + 1)
        else:
            misses.append({"question": q["question"], "gold": q["gold"], "got": got})
    n = max(len(questions), 1)
    return {"n": len(questions), "k": k, f"recall@{k}": hits / n, "mrr": rr / n, "misses": misses}


CITE = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def check_grounded(answer, chunks, min_overlap=0.5):
    """Each sentence must cite an existing chunk and share enough content words with it."""
    by_id = {c["id"]: set(tokenize(c["text"])) for c in chunks}
    problems = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    for s in sentences:
        cites = CITE.findall(s)
        words = set(tokenize(CITE.sub("", s)))
        if not cites:
            problems.append({"sentence": s, "issue": "no citation"})
            continue
        unknown = [c for c in cites if c not in by_id]
        if unknown:
            problems.append({"sentence": s, "issue": f"unknown chunk {unknown[0]}"})
            continue
        support = set().union(*(by_id[c] for c in cites))
        overlap = len(words & support) / max(len(words), 1)
        if overlap < min_overlap:
            problems.append({"sentence": s, "issue": f"weak support ({overlap:.0%})"})
    return {"sentences": len(sentences), "problems": problems, "ok": not problems}


def run(corpus_path, questions_path, k=3):
    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(questions_path) as f:
        qs = json.load(f)
    idx = BM25(corpus["chunks"])
    ret = retrieval_metrics(idx, qs["questions"], k)
    grounded = []
    for a in qs.get("answers", []):
        r = check_grounded(a["answer"], corpus["chunks"])
        grounded.append({"id": a["id"], **r})
    return {"retrieval": ret, "grounding": grounded}


def render(res):
    r = res["retrieval"]
    out = [f"Retrieval: recall@{r['k']} = {r[f'recall@{r['k']}']:.2f}, MRR = {r['mrr']:.2f} ({r['n']} questions)"]
    for m in r["misses"]:
        out.append(f"  MISS: {m['question']!r} wanted {m['gold']}, got {m['got']}")
    for g in res["grounding"]:
        out.append(f"Answer {g['id']}: {'OK' if g['ok'] else 'PROBLEMS'}")
        for p in g["problems"]:
            out.append(f"  - {p['issue']}: {p['sentence']}")
    return "\n".join(out)
