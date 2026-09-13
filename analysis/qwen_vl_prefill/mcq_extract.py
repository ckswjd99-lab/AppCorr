r"""MCQ answer extraction that reads the ANSWER, not the first capital letter in the string.

Why this exists (measured 2026-09-13 on B200-8, GLM-5.3-Flash stock TP=2 proof):
GLM-5.3 answered a V*Bench row in prose and never emitted a letter --

    "The user is asking about the material of the glove visible in the image. Looking at the
     image, there's a person wearing a blue glove ..."

and `VStarSpec.score(text, gold="A")` returned **ok=1**: the first `[ABCD]` match is the 'A' in
"ASKING", index 12. The model never answered and the scorer awarded the point.

It is not one spec, and the four letter-scorers fail on DIFFERENT prose -- which one fires
depends on incidental wording, so none of them is safe. Measured on the live file, gold="A"
(gold="a" for MMVP, whose choices are a/b):

    text                                                  cvbench  rwqa  vstar  mmvp  this
    the GLM-5.3 output above                                    0     0      1     1  none
    "Looking at the image, I can see a person wearing ..."      0     0      1     1  none
    "a person is wearing a blue glove in this photo"            1     1      1     1  none
    "Based on the image, the object appears to be a rubber"     1     1      0     1  none
    "About the glove: the material looks like rubber."          0     1      1     0  none

The mechanisms:

    VStarSpec / SOUDrivingSpec  re.search(r"[ABCD]", pred.upper())  -- any capital A-D, inside
                                words: the 'A' of "ASKING", the 'B' of "Based" (which scores a
                                WRONG letter, not a miss)
    MMVPSpec                    r"\(([ab])\)" or \b([ab])\b        -- the English article "a"
    CVBenchSpec                 r"\(([A-Da-d])\)" or \b([A-Za-z])\b  -- the same article, but only
    RealWorldQASpec (MCQ mode)  \b([A-Za-z])\b, then first [A-Za-z]   when no other standalone
                                letter precedes it; "there's" and "I" shadow it and yield 'S'/'I',
                                which score 0 by luck rather than by correctness

That last point is the trap inside the trap: cvbench and rwqa returning 0 on the GLM-5.3 output
is a coincidence of a contraction appearing before the first article, not evidence that they are
sound. Change the wording slightly and they award the point too.

The bias matters more than the rate: 'a'/"asking"/"about"/"appears" are everywhere in English
prose, so gold-A rows collect free points while B/C/D rows mostly score 0. In aggregate it does
not look like noise, it looks like a model that is good at the A options.

Contract:
    extract_choice(text, choices="ABCD") -> letter (upper) or None
`None` means NO ANSWER FOUND and the caller should score 0 and record `no_answer=True`, rather
than fall back to a lucky match.

A letter inside a word never counts. A bare standalone letter in running prose never counts --
only a letter that IS the answer: the whole string, the whole last line, a sentinel, an explicit
"answer is X", or a parenthesised "(X)".
"""

import re

__all__ = ["extract_choice", "score_mcq"]

_SENTINEL = re.compile(r"<\|begin_of_box\|>\s*\(?\s*([A-Za-z])\s*\)?\s*<\|end_of_box\|>")
_PUNCT = r"[.．,:)\]}*_\s]*"


def _only_letter(s: str, choices: str):
    """The whole string is the answer: 'A', '(B)', 'B.', ' C ) ', '**D**'."""
    m = re.match(r"^[*_#\s]{0,4}\(?\s*([A-Za-z])\s*\)?" + _PUNCT + r"$", s.strip())
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    return None


def extract_choice(text, choices: str = "ABCD"):
    """Return the chosen letter (uppercase) or None if the text does not contain an answer."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    choices = choices.upper()

    # 1. sentinel-wrapped letter (GLM emits these; our clean_text strips them, so this is for
    #    raw/unclean text and costs nothing when they are already gone)
    m = _SENTINEL.search(t)
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()

    # 2. the whole answer is the letter
    r = _only_letter(t, choices)
    if r:
        return r

    # 3. "A. 1", "B. 2 cars", "C) green" -- the letter, a delimiter, then the option text.
    #    The delimiter is what keeps prose out: "A person is wearing ..." has no delimiter after
    #    the 'A', and "About the glove:" has a letter immediately followed by more letters.
    #    (Found by the regression gate: these rows were being called no-answer.)
    for probe in (t, t.splitlines()[-1].strip() if t.splitlines() else ""):
        m = re.match(r"^[*_#\s]{0,4}\(?\s*([A-Za-z])\s*\)?\s*[.)\]:\-\u2013]\s+\S", probe)
        if m and m.group(1).upper() in choices:
            return m.group(1).upper()

    # 4. an explicit answer statement
    # `[\s*_:]*` after the marker: models write "Correct answer: **(D) blue**" and the markdown
    # emphasis sat between the colon and the letter, so this rule missed a clear answer.
    # (Found on real 4B V*Bench output at n=160, not in the synthetic cases.)
    pat = (r"(?:answer|option|choice|correct\s+(?:answer|option|choice))"
           r"\s*(?:is|:|=)?[\s*_]*\(?\s*([A-Za-z])\s*\)?(?![A-Za-z])")
    best = None
    for m in re.finditer(pat, t, re.I):
        if m.group(1).upper() in choices:
            best = m.group(1).upper()      # last such statement wins ("... so the answer is C")
    if best:
        return best

    # 5. a standalone letter that is the whole final line
    for line in reversed([l for l in t.splitlines() if l.strip()]):
        r = _only_letter(line, choices)
        if r:
            return r
        break                                # only the last non-empty line

    # 6. a parenthesised letter, e.g. "... so (B) is right". Requires the parens: this is the
    #    weakest rule, and without the parens it would readmit the article.
    hits = [g.upper() for g in re.findall(r"\(\s*([A-Za-z])\s*\)", t) if g.upper() in choices]
    if len(set(hits)) == 1:
        return hits[0]                       # unambiguous; several distinct ones = restated options

    return None


def score_mcq(pred_text, gold, choices: str = "ABCD"):
    """(ok, val, no_answer). ok=0 and no_answer=True when no answer could be extracted."""
    pred = extract_choice(pred_text, choices)
    if pred is None:
        return 0, 0.0, True
    ok = int(pred == str(gold).strip().upper())
    return ok, float(ok), False
