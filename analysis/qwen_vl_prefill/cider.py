"""
cider.py -- CIDEr-D (standard image-captioning metric), self-contained (pycocoevalcap not installed).
Follows the coco-caption cider_scorer: n-grams n=1..4, tf-idf with document frequency over the whole
reference corpus (each image = one document), per-order cosine similarity with min-count clipping and
a gaussian length penalty (sigma=6), averaged over refs and orders, x10.
"""
import math
import re
from collections import Counter, defaultdict


def _tok(s):
    return re.sub(r"[^\w\s]", " ", str(s).lower()).split()


def _ngrams(tokens, n):
    c = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            c[tuple(tokens[i:i + k])] += 1
    return c


def compute_cider(candidates, references, n=4, sigma=6.0):
    """candidates: {id: str}; references: {id: [str, ...]}. Returns (mean_cider, {id: cider})."""
    ids = list(candidates.keys())
    cand_ng = {i: _ngrams(_tok(candidates[i]), n) for i in ids}
    refs_ng = {i: [_ngrams(_tok(r), n) for r in references[i]] for i in ids}

    df = defaultdict(float)
    for i in ids:
        seen = set()
        for ref in refs_ng[i]:
            seen.update(ref.keys())
        for ng in seen:
            df[ng] += 1.0
    N = max(len(ids), 1)
    logN = math.log(N)

    def vec(counter):
        v = [defaultdict(float) for _ in range(n)]
        norm = [0.0] * n
        length = 0
        for ng, cnt in counter.items():
            o = len(ng) - 1
            idf = logN - math.log(max(df.get(ng, 0.0), 1.0))
            tfidf = cnt * idf
            v[o][ng] = tfidf
            norm[o] += tfidf * tfidf
            if o == 0:
                length += cnt
        return v, [math.sqrt(x) for x in norm], length

    def sim(vc, nc, lc, vr, nr, lr):
        delta = lc - lr
        out = [0.0] * n
        for o in range(n):
            for ng, tc in vc[o].items():
                tr = vr[o].get(ng, 0.0)
                if tr:
                    out[o] += min(tc, tr) * tr
            if nc[o] and nr[o]:
                out[o] /= (nc[o] * nr[o])
            out[o] *= math.exp(-(delta ** 2) / (2 * sigma * sigma))
        return out

    per = {}
    for i in ids:
        vc, nc, lc = vec(cand_ng[i])
        acc = [0.0] * n
        refs = refs_ng[i]
        for ref in refs:
            vr, nr, lr = vec(ref)
            s = sim(vc, nc, lc, vr, nr, lr)
            for o in range(n):
                acc[o] += s[o]
        m = max(len(refs), 1)
        per[i] = 10.0 * sum(acc[o] / m for o in range(n)) / n
    mean = sum(per.values()) / max(len(per), 1)
    return mean, per
