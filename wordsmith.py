#!/usr/bin/env python3
"""
Wordsmith - spelling Bee solver focused on reaching Genius.

Optional but strongly recommended:  pip install wordfreq
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

try:
    import nltk
    from nltk.corpus import words as nltk_words
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

ENABLE1_URL = "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"

ENGLISH_LETTER_FREQUENCIES = {
    'e': 11.16, 'a': 8.50, 'r': 7.58, 'i': 7.54, 'o': 7.16,
    't': 6.95, 'n': 6.65, 's': 5.74, 'l': 5.49, 'c': 4.54,
    'u': 3.63, 'd': 3.38, 'p': 3.17, 'm': 3.01, 'h': 3.00,
    'g': 2.47, 'b': 2.07, 'f': 1.81, 'y': 1.78, 'w': 1.28,
    'k': 1.10, 'v': 1.01, 'x': 0.29, 'z': 0.27, 'j': 0.20, 'q': 0.20,
}

BEE_RANKS = [
    ("Beginner", 0.00), ("Good Start", 0.02), ("Moving Up", 0.05),
    ("Good", 0.08), ("Solid", 0.15), ("Nice", 0.25), ("Great", 0.40),
    ("Amazing", 0.50), ("Genius", 0.70), ("Queen Bee", 1.00),
]
GENIUS_FRACTION = 0.70

# Confidence tiers (estimated probability the NYT accepts the word).
SAFE_THRESHOLD = 0.75
LIKELY_THRESHOLD = 0.45
DEFAULT_CONFIDENCE = 0.60  # used when wordfreq is not installed

# Suffixes used to group words into families (longest first).
FAMILY_SUFFIXES = ["ments", "ment", "ness", "able", "ions", "ings", "ion",
                   "ing", "ers", "er", "ed", "es", "ly", "s", "y", "d"]


# --------------------------------------------------------------------------
# Word list loading
# --------------------------------------------------------------------------
def _clean(lines):
    out = set()
    for line in lines:
        w = line.strip().lower()
        if w.isascii() and w.isalpha():
            out.add(w)
    return out


def load_word_list(filepath, refresh=False):
    """Local file -> online enable1.txt (cached) -> NLTK corpus."""
    if filepath and os.path.exists(filepath) and not refresh:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                words = _clean(f)
            if words:
                return words
            print(f"[WARNING] '{filepath}' contained no usable words.", file=sys.stderr)
        except OSError as e:
            print(f"[WARNING] Could not read '{filepath}': {e}", file=sys.stderr)

    print(f"Downloading word list from {ENABLE1_URL} ...", file=sys.stderr)
    try:
        req = urllib.request.Request(ENABLE1_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        words = _clean(text.splitlines())
        if words:
            if filepath:
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write("\n".join(sorted(words)))
                    print(f"Cached word list to '{filepath}'.", file=sys.stderr)
                except OSError as e:
                    print(f"[WARNING] Could not cache word list: {e}", file=sys.stderr)
            return words
    except Exception as e:
        print(f"[WARNING] Failed to download enable1.txt: {e}", file=sys.stderr)

    if NLTK_AVAILABLE:
        print("Falling back to NLTK words corpus...", file=sys.stderr)
        try:
            try:
                nltk_words.words()
            except LookupError:
                nltk.download("words", quiet=True)
            words = _clean(nltk_words.words())
            if words:
                return words
        except Exception as e:
            print(f"[WARNING] NLTK fallback failed: {e}", file=sys.stderr)

    raise ValueError(
        "Could not load any word list. Check your internet connection, "
        "install nltk (`pip install nltk`), or provide a valid file with -f."
    )


def load_word_file(path, label, required=True):
    """Load a one-word-per-line file. Missing optional files give an empty set."""
    if not path:
        return set()
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"{label} file not found: {path}")
        return set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean(f)


def save_word_file(path, words):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)) + ("\n" if words else ""))


# --------------------------------------------------------------------------
# Validation, scoring, matching
# --------------------------------------------------------------------------
def validate_inputs(optional_letters, mandatory_letter):
    opt = optional_letters.strip().lower()
    mand = mandatory_letter.strip().lower()

    if not opt.isalpha() or not mand.isalpha() or not (opt + mand).isascii():
        raise ValueError("Letters must contain only alphabetic characters (A-Z).")
    if len(mand) != 1:
        raise ValueError("The mandatory letter must be a single character.")
    if len(set(opt)) != len(opt):
        raise ValueError("Optional letters must all be different.")
    if len(opt) != 6:
        raise ValueError(f"Exactly 6 optional letters are required (got {len(opt)}).")
    if mand in opt:
        raise ValueError("The mandatory letter must not appear in the optional letters.")
    return opt, mand


def calculate_score(word, is_pangram):
    """4-letter word = 1 point; longer = 1 point per letter; pangram +7."""
    score = 1 if len(word) == 4 else len(word)
    return score + 7 if is_pangram else score


def compile_pattern(pattern):
    if not pattern:
        return None
    escaped = re.escape(pattern.lower()).replace(r"\?", "[a-z]").replace(r"\*", "[a-z]*")
    return re.compile(f"^{escaped}$")


def find_matching_words(optional_letters, mandatory_letter, word_list, filters):
    opt, mand = validate_inputs(optional_letters, mandatory_letter)
    allowed = set(opt + mand)

    min_len = max(4, filters.get("min_length") or 4)
    max_len = filters.get("max_length")
    starts = (filters.get("starts_with") or "").lower()
    ends = (filters.get("ends_with") or "").lower()
    contains = (filters.get("contains") or "").lower()
    regex = compile_pattern(filters.get("pattern"))
    blocked = filters.get("blocked") or set()
    pangrams_only = filters.get("pangrams_only", False)

    if max_len is not None and max_len < min_len:
        raise ValueError("--max-len cannot be smaller than --min-len.")

    results = []
    for word in word_list:
        n = len(word)
        if n < min_len or (max_len and n > max_len):
            continue
        if word in blocked:
            continue
        letters = set(word)
        if mand not in letters or not letters <= allowed:
            continue
        if starts and not word.startswith(starts):
            continue
        if ends and not word.endswith(ends):
            continue
        if contains and contains not in word:
            continue
        if regex and not regex.match(word):
            continue

        is_pangram = allowed <= letters
        if pangrams_only and not is_pangram:
            continue
        results.append({
            "word": word,
            "is_pangram": is_pangram,
            "score": calculate_score(word, is_pangram),
        })
    return results


# --------------------------------------------------------------------------
# Confidence scoring (NYT acceptance likelihood)
# --------------------------------------------------------------------------
def tier_for(conf):
    if conf >= SAFE_THRESHOLD:
        return "Safe"
    if conf >= LIKELY_THRESHOLD:
        return "Likely"
    return "Risky"


def acceptance_confidence(word, allow):
    """
    Estimated probability (0-1) that the NYT accepts the word.
    Uses Zipf word frequency through a logistic curve:
      zipf 4 -> ~0.95, 3 -> ~0.82, 2 -> 0.50, 1 -> ~0.18, 0 -> ~0.05
    Allowlisted words are treated as certain.
    """
    if word in allow:
        return 1.0
    if not WORDFREQ_AVAILABLE:
        return DEFAULT_CONFIDENCE
    zipf = zipf_frequency(word, "en")
    return 1.0 / (1.0 + math.exp(-1.5 * (zipf - 2.0)))


def enrich_words(words, allow):
    """Add confidence, tier and expected points to each word dict."""
    for w in words:
        w["confidence"] = round(acceptance_confidence(w["word"], allow), 3)
        w["tier"] = tier_for(w["confidence"])
        w["expected"] = round(w["score"] * w["confidence"], 2)
    return words


# --------------------------------------------------------------------------
# Analysis: ranks, Genius planner, difficulty
# --------------------------------------------------------------------------
def rank_letters(letters):
    unique = sorted(set(letters.lower()))
    return sorted(unique, key=lambda l: (-ENGLISH_LETTER_FREQUENCIES.get(l, 0), l))


def letter_usage(words, letters):
    counts = Counter()
    for item in words:
        for ch in set(item["word"]):
            counts[ch] += 1
    return {l: counts.get(l, 0) for l in sorted(set(letters.lower()))}


def bee_rank_targets(total_points):
    return [(name, round(total_points * frac)) for name, frac in BEE_RANKS]


def plan_genius(all_words, found):
    """
    Build the shortest high-confidence path to Genius.
    Words are taken in order of confidence, then points, until the target is met.
    """
    total = sum(w["score"] for w in all_words)
    target = round(total * GENIUS_FRACTION)
    have = sum(w["score"] for w in all_words if w["word"] in found)
    remaining = [w for w in all_words if w["word"] not in found]

    tier_points = {"Safe": 0, "Likely": 0, "Risky": 0}
    for w in remaining:
        tier_points[w["tier"]] += w["score"]

    ordered = sorted(remaining, key=lambda w: (-w["confidence"], -w["score"], w["word"]))
    steps, running = [], have
    for w in ordered:
        if running >= target:
            break
        running += w["score"]
        steps.append({**w, "cumulative": running})

    return {
        "total": total, "target": target, "have": have,
        "need": max(0, target - have), "steps": steps,
        "reachable": running >= target,
        "tier_points": tier_points,
        "safe_reaches_genius": have + tier_points["Safe"] >= target,
        "safe_plus_likely_reaches": have + tier_points["Safe"] + tier_points["Likely"] >= target,
        "expected_points": round(sum(s["expected"] for s in steps), 1),
    }


def estimate_difficulty(all_words, plan):
    """Rate the puzzle from how much of Genius the safe words cover."""
    if not all_words:
        return "n/a", ["No words found."]
    target = plan["target"] or 1
    safe_cover = (plan["have"] + plan["tier_points"]["Safe"]) / target
    pangrams = [w for w in all_words if w["is_pangram"]]
    notes = [f"Safe words cover {min(safe_cover, 9.99):.0%} of the Genius target."]
    if pangrams:
        best = max(p["confidence"] for p in pangrams)
        notes.append(f"Most accessible pangram is a {tier_for(best)} word "
                     f"(~{best:.0%} confidence).")
    if safe_cover >= 1.0:
        rating = "Easy"
    elif safe_cover >= 0.80:
        rating = "Medium"
    else:
        rating = "Hard"
    return rating, notes


# --------------------------------------------------------------------------
# Word families, affixes
# --------------------------------------------------------------------------
def build_families(all_words):
    """Group words under a root that is also a valid puzzle word."""
    wordset = {w["word"] for w in all_words}
    memo = {}

    def parent(word):
        best = None
        for suf in FAMILY_SUFFIXES:
            if not word.endswith(suf) or len(word) - len(suf) < 3:
                continue
            stem = word[:-len(suf)]
            cands = [stem, stem + "e"]
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])
            for c in cands:
                if c != word and len(c) >= 4 and c in wordset:
                    if best is None or len(c) < len(best):
                        best = c
        return best

    def root(word):
        if word in memo:
            return memo[word]
        p = parent(word)
        memo[word] = word if p is None else root(p)
        return memo[word]

    families = defaultdict(list)
    for w in all_words:
        families[root(w["word"])].append(w)
    return {r: ms for r, ms in families.items() if len(ms) >= 2}


def affix_stats(words, kind, lengths=(2, 3, 4), min_count=3):
    stats = defaultdict(lambda: [0, 0])
    for w in words:
        word = w["word"]
        for n in lengths:
            if len(word) - n < 3:
                continue
            frag = word[-n:] if kind == "suffix" else word[:n]
            stats[frag][0] += 1
            stats[frag][1] += w["score"]
    rows = [(f, c, p) for f, (c, p) in stats.items() if c >= min_count]
    return sorted(rows, key=lambda r: (-r[2], r[0]))


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------
def mask_word(word):
    """Spoiler-light hint: first and last letter plus length."""
    return f"{word[0].upper()}{'_' * (len(word) - 2)}{word[-1].upper()} ({len(word)} letters)"


def print_summary(words, all_letters, found, pangram_hint):
    total = sum(w["score"] for w in words)
    pangrams = [w for w in words if w["is_pangram"]]
    print(f"Total words   : {len(words)}")
    print(f"Total points  : {total}")
    if pangram_hint:
        shown = ", ".join(mask_word(p["word"]) for p in pangrams) or "none"
        print(f"Pangrams      : {len(pangrams)} - {shown}")
    else:
        names = ", ".join(f"{p['word']} ({p['tier']})" for p in pangrams)
        print(f"Pangrams      : {len(pangrams)}" + (f" - {names}" if pangrams else ""))

    by_len = Counter(len(w["word"]) for w in words)
    print("Words by length : " + ", ".join(f"{k}:{by_len[k]}" for k in sorted(by_len)))
    print("Words per letter: " + ", ".join(
        f"{l.upper()}:{c}" for l, c in letter_usage(words, all_letters).items()))

    tiers = Counter(w["tier"] for w in words)
    print(f"Confidence tiers: Safe {tiers['Safe']}, Likely {tiers['Likely']}, Risky {tiers['Risky']}")
    print("-" * 56)
    print("Rank targets (points needed):")
    for name, pts in bee_rank_targets(total):
        print(f"  {name:<11} {pts}")


def print_difficulty(rating, notes):
    print(f"Difficulty estimate: {rating}")
    for n in notes:
        print(f"  - {n}")


def print_plan(plan, detail):
    tp = plan["tier_points"]
    print(f"Genius target : {plan['target']} pts (you have {plan['have']}, need {plan['need']})")
    print(f"Points still available -> Safe: {tp['Safe']}, Likely: {tp['Likely']}, Risky: {tp['Risky']}")
    if plan["need"] == 0:
        print("You have already reached Genius!")
        return
    if plan["safe_reaches_genius"]:
        print("Verdict: Safe words alone reach Genius - no need to gamble on obscure words.")
    elif plan["safe_plus_likely_reaches"]:
        print("Verdict: Safe + Likely words reach Genius. Avoid Risky words.")
    elif plan["reachable"]:
        print("Verdict: you will need to include some Risky words to reach Genius.")
    else:
        print("Verdict: Genius is not reachable with the current word list.")
    print(f"Path length   : {len(plan['steps'])} word(s), ~{plan['expected_points']} expected pts")
    if detail:
        print(f"\n{'#':<4} {'Word':<16} {'Pts':<5} {'Conf':<6} {'Tier':<7} Running total")
        for i, s in enumerate(plan["steps"], 1):
            print(f"{i:<4} {s['word']:<16} {s['score']:<5} {s['confidence']:<6.2f} "
                  f"{s['tier']:<7} {s['cumulative']}")


def print_hints(words):
    grid = defaultdict(Counter)
    lengths = set()
    for w in words:
        grid[w["word"][0]][len(w["word"])] += 1
        lengths.add(len(w["word"]))
    lengths = sorted(lengths)

    print("Hint grid (first letter x length):")
    print("     " + "".join(f"{n:>4}" for n in lengths) + "    Σ")
    for letter in sorted(grid):
        row = "".join(f"{grid[letter].get(n, 0) or '-':>4}" for n in lengths)
        print(f"  {letter.upper()}  {row}  {sum(grid[letter].values()):>3}")

    pairs = Counter(w["word"][:2] for w in words)
    print("\nTwo-letter starts: " + ", ".join(f"{p.upper()}-{c}" for p, c in sorted(pairs.items())))


def print_families(families, limit):
    if not families:
        print("No word families found.")
        return
    ranked = sorted(families.items(),
                    key=lambda kv: (-sum(m["score"] for m in kv[1]), kv[0]))
    print(f"Word families (top {min(limit, len(ranked))} by total points):")
    for root, members in ranked[:limit]:
        pts = sum(m["score"] for m in members)
        names = ", ".join(m["word"] for m in sorted(members, key=lambda m: len(m["word"])))
        print(f"  {root.upper():<12} {len(members)} words, {pts:>3} pts: {names}")


def print_affixes(words, limit=8):
    for kind in ("suffix", "prefix"):
        rows = affix_stats(words, kind)[:limit]
        if not rows:
            continue
        print(f"Top {kind}es by points:")
        print("  " + ", ".join(
            f"{('-' + f) if kind == 'suffix' else (f + '-')} ({c}w/{p}p)" for f, c, p in rows))


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_results(words, path):
    ext = os.path.splitext(path)[1].lower()
    fields = ["word", "score", "confidence", "tier", "expected", "is_pangram"]
    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{k: w[k] for k in fields} for w in words], f, indent=2)
    elif ext == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for w in words:
                writer.writerow({k: w[k] for k in fields})
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(w["word"] for w in words) + "\n")
    print(f"Exported {len(words)} word(s) to '{path}'.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(
        prog="wordsmith",
        description="Wordsmith: solve, score and plan NYT Spelling Bee puzzles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("optional", nargs="?", help="the 6 outer letters (e.g., 'abcefg')")
    p.add_argument("mandatory", nargs="?", help="the 1 center letter (e.g., 'd')")
    p.add_argument("-f", "--file", default="words.txt",
                   help="word list file (downloaded and cached here if missing)")
    p.add_argument("--refresh", action="store_true", help="re-download the word list")

    g = p.add_argument_group("Word filters")
    g.add_argument("--min-len", type=int, default=4, help="minimum word length (min 4)")
    g.add_argument("--max-len", type=int, default=None, help="maximum word length")
    g.add_argument("--starts", default="", help="words starting with this prefix")
    g.add_argument("--ends", default="", help="words ending with this suffix")
    g.add_argument("--contains", default="", help="words containing this substring")
    g.add_argument("--pattern", default=None,
                   help="wildcard pattern: ? = one letter, * = any letters (e.g., 'a?c*')")
    g.add_argument("--pangrams", action="store_true", help="show only pangrams")
    g.add_argument("--min-conf", type=float, default=0.0,
                   help="hide words below this confidence (0-1; 0.45 = Likely+, 0.75 = Safe only)")
    g.add_argument("--found", metavar="FILE", help="file of words already found (hidden)")

    k = p.add_argument_group("Learning lists (persistent)")
    k.add_argument("--blocklist", default="blocklist.txt",
                   help="words the NYT rejects (removed everywhere)")
    k.add_argument("--allowlist", default="allowlist.txt",
                   help="words the NYT accepts (treated as 100%% confident)")
    k.add_argument("--block", nargs="+", metavar="WORD", help="add words to the blocklist")
    k.add_argument("--allow", nargs="+", metavar="WORD", help="add words to the allowlist")

    o = p.add_argument_group("Analysis & output")
    o.add_argument("--plan", action="store_true", help="show the full step-by-step Genius path")
    o.add_argument("--families", action="store_true", help="show word families (shared roots)")
    o.add_argument("--affixes", action="store_true", help="show the most valuable prefixes/suffixes")
    o.add_argument("--hints", action="store_true", help="show NYT-style hint grid and exit")
    o.add_argument("--pangram-hint", action="store_true",
                   help="show pangrams masked (first/last letter + length) instead of spelled out")
    o.add_argument("--sort", choices=["expected", "score", "alpha"], default="expected",
                   help="word table order (expected = score x confidence)")
    o.add_argument("--top", type=int, default=None, help="show only the top N words")
    o.add_argument("--no-summary", action="store_true", help="hide stats, difficulty and plan summary")
    o.add_argument("--export", metavar="FILE", help="save results to .csv, .json or .txt")
    return p.parse_args()


def prompt_for_letters(args):
    print("Wordsmith interactive mode (Ctrl+C to quit)")
    args.optional = input("Enter the 6 outer letters: ").strip()
    args.mandatory = input("Enter the 1 center letter: ").strip()
    return args


def update_learning_lists(args, allow, block):
    """Apply --allow / --block and persist them. Each list removes from the other."""
    if args.block:
        new = _clean(args.block)
        block |= new
        allow -= new
        print(f"Added {len(new)} word(s) to blocklist.", file=sys.stderr)
    if args.allow:
        new = _clean(args.allow)
        allow |= new
        block -= new
        print(f"Added {len(new)} word(s) to allowlist.", file=sys.stderr)
    if args.block or args.allow:
        save_word_file(args.blocklist, block)
        save_word_file(args.allowlist, allow)


def main():
    try:
        args = parse_arguments()
        if not args.optional or not args.mandatory:
            args = prompt_for_letters(args)

        opt, mand = validate_inputs(args.optional, args.mandatory)
        all_letters = opt + mand
        if "s" in all_letters:
            print("[NOTE] Official NYT puzzles never contain the letter S.", file=sys.stderr)
        if not WORDFREQ_AVAILABLE:
            print("[NOTE] 'wordfreq' not installed - confidence scores are flat. "
                  "Run `pip install wordfreq` for real acceptance estimates.", file=sys.stderr)

        print("=" * 56)
        print(" WORDSMITH - SPELLING BEE SOLVER")
        print("=" * 56)
        print(f"Outer Letters  : {' '.join(opt.upper())}")
        print(f"Center Letter  : {mand.upper()}")
        print("-" * 56)

        word_list = load_word_list(args.file, refresh=args.refresh)
        allow = load_word_file(args.allowlist, "Allowlist", required=False)
        block = load_word_file(args.blocklist, "Blocklist", required=False)
        update_learning_lists(args, allow, block)
        found = load_word_file(args.found, "Found-words")

        word_list = set(word_list) | allow  # allowlisted words always participate

        filters = {
            "min_length": args.min_len, "max_length": args.max_len,
            "starts_with": args.starts, "ends_with": args.ends,
            "contains": args.contains, "pattern": args.pattern,
            "pangrams_only": args.pangrams, "blocked": block,
        }

        all_matches = enrich_words(
            find_matching_words(opt, mand, word_list, filters), allow)
        remaining = [w for w in all_matches if w["word"] not in found]

        plan = plan_genius(all_matches, found)

        if not args.no_summary:
            print_summary(all_matches, all_letters, found, args.pangram_hint)
            print("-" * 56)
            rating, notes = estimate_difficulty(all_matches, plan)
            print_difficulty(rating, notes)
            print("-" * 56)
            print_plan(plan, detail=args.plan)
            if found:
                print(f"Found so far  : {plan['have']} pts "
                      f"({plan['have'] / (plan['total'] or 1):.0%}) - "
                      f"{len(remaining)} word(s) remaining")
            print("-" * 56)
        elif args.plan:
            print_plan(plan, detail=True)
            print("-" * 56)

        if args.families:
            print_families(build_families(all_matches), limit=15)
            print("-" * 56)
        if args.affixes:
            print_affixes(remaining)
            print("-" * 56)

        if args.hints:
            print_hints(remaining)
            print("=" * 56)
            return

        shown = [w for w in remaining if w["confidence"] >= args.min_conf]
        if args.sort == "expected":
            shown.sort(key=lambda w: (-w["expected"], -w["score"], w["word"]))
        elif args.sort == "score":
            shown.sort(key=lambda w: (-w["score"], -w["confidence"], w["word"]))
        else:
            shown.sort(key=lambda w: w["word"])
        if args.top:
            shown = shown[:args.top]

        print(f"Showing {len(shown)} of {len(remaining)} remaining word(s) "
              f"(sorted by {args.sort}):")
        print(f"{'Rank':<6}{'Word':<16}{'Pts':<5}{'Conf':<6}{'Tier':<8}{'Exp':<6}Status")
        print("-" * 56)
        for rank, w in enumerate(shown, 1):
            status = "PANGRAM" if w["is_pangram"] else ""
            print(f"{rank:<6}{w['word']:<16}{w['score']:<5}{w['confidence']:<6.2f}"
                  f"{w['tier']:<8}{w['expected']:<6.1f}{status}")
        print("=" * 56)

        if args.export:
            export_results(shown, args.export)

    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"\n[ERROR]: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
