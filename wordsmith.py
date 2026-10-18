#!/usr/bin/env python3
"""
Wordsmith - Spelling Bee solver, scorer and hint generator.

Examples:
    python wordsmith.py abcefg d
    python wordsmith.py abcefg d --pangrams
    python wordsmith.py abcefg d --starts de --min-len 6 --top 10
    python wordsmith.py abcefg d --hints
    python wordsmith.py abcefg d --found found.txt --export results.csv
    python wordsmith.py                       # interactive mode
"""

import argparse
import csv
import json
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

ENABLE1_URL = "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"

# Approximate English letter frequencies (%), used for the general ranking.
ENGLISH_LETTER_FREQUENCIES = {
    'e': 11.16, 'a': 8.50, 'r': 7.58, 'i': 7.54, 'o': 7.16,
    't': 6.95, 'n': 6.65, 's': 5.74, 'l': 5.49, 'c': 4.54,
    'u': 3.63, 'd': 3.38, 'p': 3.17, 'm': 3.01, 'h': 3.00,
    'g': 2.47, 'b': 2.07, 'f': 1.81, 'y': 1.78, 'w': 1.28,
    'k': 1.10, 'v': 1.01, 'x': 0.29, 'z': 0.27, 'j': 0.20, 'q': 0.20,
}

# NYT Spelling Bee rank thresholds as a fraction of total possible points.
BEE_RANKS = [
    ("Beginner", 0.00), ("Good Start", 0.02), ("Moving Up", 0.05),
    ("Good", 0.08), ("Solid", 0.15), ("Nice", 0.25), ("Great", 0.40), 
    ("Amazing", 0.50), ("Genius", 0.70), ("Queen Bee", 1.00),
]


# --------------------------------------------------------------------------
# Word list loading
# --------------------------------------------------------------------------
def _clean(lines):
    """Lowercase, strip and keep only purely alphabetic words (deduplicated)."""
    out = set()
    for line in lines:
        w = line.strip().lower()
        if w.isascii() and w.isalpha():
            out.add(w)
    return out


def load_word_list(filepath, refresh=False):
    """
    Load words from, in order:
      1. A local file (if it exists and --refresh is not set).
      2. The online enable1.txt list (cached to `filepath` for next time).
      3. The NLTK words corpus.
    """
    # 1. Local file
    if filepath and os.path.exists(filepath) and not refresh:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                words = _clean(f)
            if words:
                return words
            print(f"[WARNING] '{filepath}' contained no usable words.", file=sys.stderr)
        except OSError as e:
            print(f"[WARNING] Could not read '{filepath}': {e}", file=sys.stderr)

    # 2. Download enable1.txt
    print(f"Downloading word list from {ENABLE1_URL} ...", file=sys.stderr)
    try:
        req = urllib.request.Request(ENABLE1_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        words = _clean(text.splitlines())
        if words:
            if filepath:  # cache for future runs
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write("\n".join(sorted(words)))
                    print(f"Cached word list to '{filepath}'.", file=sys.stderr)
                except OSError as e:
                    print(f"[WARNING] Could not cache word list: {e}", file=sys.stderr)
            return words
    except Exception as e:
        print(f"[WARNING] Failed to download enable1.txt: {e}", file=sys.stderr)

    # 3. NLTK fallback
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


def load_word_file(path, label):
    """Load a simple one-word-per-line file (e.g., found words / exclusions)."""
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean(f)


# --------------------------------------------------------------------------
# Validation, scoring, matching
# --------------------------------------------------------------------------
def validate_inputs(optional_letters, mandatory_letter):
    """Validate and normalise the puzzle letters."""
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
    """Turn a '?' / '*' wildcard pattern into a compiled regex (letters only)."""
    if not pattern:
        return None
    escaped = re.escape(pattern.lower()).replace(r"\?", "[a-z]").replace(r"\*", "[a-z]*")
    return re.compile(f"^{escaped}$")


def find_matching_words(optional_letters, mandatory_letter, word_list, filters):
    """Find every valid puzzle word and apply the CLI filters."""
    opt, mand = validate_inputs(optional_letters, mandatory_letter)
    allowed = set(opt + mand)

    min_len = max(4, filters.get("min_length") or 4)
    max_len = filters.get("max_length")
    starts = (filters.get("starts_with") or "").lower()
    ends = (filters.get("ends_with") or "").lower()
    contains = (filters.get("contains") or "").lower()
    regex = compile_pattern(filters.get("pattern"))
    exclude = filters.get("exclude") or set()
    pangrams_only = filters.get("pangrams_only", False)

    if max_len is not None and max_len < min_len:
        raise ValueError("--max-len cannot be smaller than --min-len.")

    results = []
    for word in word_list:
        n = len(word)
        if n < min_len or (max_len and n > max_len):
            continue
        if word in exclude:
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
# Analysis helpers
# --------------------------------------------------------------------------
def rank_letters(letters):
    """Rank letters by general English frequency (stable, deterministic ties)."""
    unique = sorted(set(letters.lower()))
    return sorted(unique, key=lambda l: (-ENGLISH_LETTER_FREQUENCIES.get(l, 0), l))


def letter_usage(words, letters):
    """How many solution words use each puzzle letter."""
    counts = Counter()
    for item in words:
        for ch in set(item["word"]):
            counts[ch] += 1
    return {l: counts.get(l, 0) for l in sorted(set(letters.lower()))}


def bee_rank_targets(total_points):
    """Points needed for each NYT Spelling Bee rank."""
    return [(name, round(total_points * frac)) for name, frac in BEE_RANKS]


def print_summary(words, all_letters):
    total = sum(w["score"] for w in words)
    pangrams = [w for w in words if w["is_pangram"]]
    print(f"Total words   : {len(words)}")
    print(f"Total points  : {total}")
    print(f"Pangrams      : {len(pangrams)}"
          + (f" ({', '.join(p['word'] for p in pangrams)})" if pangrams else ""))

    by_len = Counter(len(w["word"]) for w in words)
    print("Words by length: " + ", ".join(f"{k}:{by_len[k]}" for k in sorted(by_len)))

    print("Words per letter: " + ", ".join(
        f"{l.upper()}:{c}" for l, c in letter_usage(words, all_letters).items()))
    print("-" * 50)
    print("Rank targets (points needed):")
    for name, pts in bee_rank_targets(total):
        print(f"  {name:<11} {pts}")


def print_hints(words):
    """NYT-style hint grid (first letter x word length) + two-letter list."""
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


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_results(words, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(words, f, indent=2)
    elif ext == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["word", "score", "is_pangram"])
            writer.writeheader()
            for w in words:
                writer.writerow({k: w[k] for k in ("word", "score", "is_pangram")})
    else:  # plain text
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(w["word"] for w in words) + "\n")
    print(f"Exported {len(words)} word(s) to '{path}'.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(
        prog="wordsmith",
        description="Wordsmith: solve, filter and score NYT Spelling Bee puzzles.",
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
    g.add_argument("--found", metavar="FILE", help="file of words already found (hidden)")
    g.add_argument("--exclude", metavar="FILE", help="file of words to ignore (e.g., obscure ones)")

    o = p.add_argument_group("Output")
    o.add_argument("--top", type=int, default=None, help="show only the top N words")
    o.add_argument("--hints", action="store_true", help="show NYT-style hint grid only")
    o.add_argument("--no-summary", action="store_true", help="hide stats summary")
    o.add_argument("--export", metavar="FILE",
                   help="save results to .csv, .json or .txt (by extension)")
    return p.parse_args()


def prompt_for_letters(args):
    """Interactive mode when letters aren't given on the command line."""
    print("Wordsmith interactive mode (Ctrl+C to quit)")
    args.optional = input("Enter the 6 outer letters: ").strip()
    args.mandatory = input("Enter the 1 center letter: ").strip()
    return args


def main():
    try:
        args = parse_arguments()
        if not args.optional or not args.mandatory:
            args = prompt_for_letters(args)

        # Validate first so bad input fails before any download.
        opt, mand = validate_inputs(args.optional, args.mandatory)
        all_letters = opt + mand
        if "s" in all_letters:
            print("[NOTE] Official NYT puzzles never contain the letter S.", file=sys.stderr)

        print("=" * 50)
        print(" WORDSMITH - SPELLING BEE SOLVER")
        print("=" * 50)
        print(f"Outer Letters  : {' '.join(opt.upper())}")
        print(f"Center Letter  : {mand.upper()}")
        print(f"Word List File : {args.file}")
        print("-" * 50)

        print("Letters ranked by English frequency:")
        for rank, l in enumerate(rank_letters(all_letters), 1):
            print(f" {rank}. {l.upper()} ({ENGLISH_LETTER_FREQUENCIES.get(l, 0.0)}%)")
        print("-" * 50)

        word_list = load_word_list(args.file, refresh=args.refresh)
        found = load_word_file(args.found, "Found-words")
        exclude = load_word_file(args.exclude, "Exclude")

        filters = {
            "min_length": args.min_len,
            "max_length": args.max_len,
            "starts_with": args.starts,
            "ends_with": args.ends,
            "contains": args.contains,
            "pattern": args.pattern,
            "pangrams_only": args.pangrams,
            "exclude": exclude,
        }

        all_matches = find_matching_words(opt, mand, word_list, filters)
        # Words already found still count toward totals but are hidden from the list.
        remaining = [w for w in all_matches if w["word"] not in found]
        remaining.sort(key=lambda x: (-x["score"], x["word"]))

        if not args.no_summary:
            print_summary(all_matches, all_letters)
            if found:
                got = sum(w["score"] for w in all_matches if w["word"] in found)
                total = sum(w["score"] for w in all_matches) or 1
                print(f"Found so far  : {got} pts ({got / total:.0%}) - "
                      f"{len(remaining)} word(s) remaining")
            print("-" * 50)

        if args.hints:
            print_hints(remaining)
            print("=" * 50)
            return

        shown = remaining[:args.top] if args.top else remaining
        print(f"Showing {len(shown)} of {len(remaining)} word(s):")
        print(f"{'Rank':<6} {'Word':<18} {'Score':<8} Status")
        print("-" * 50)
        for rank, item in enumerate(shown, 1):
            status = "PANGRAM" if item["is_pangram"] else ""
            print(f"{rank:<6} {item['word']:<18} {item['score']:<8} {status}")
        print("=" * 50)

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
