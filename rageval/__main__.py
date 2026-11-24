import argparse
import json
import sys

from .core import render, run


def main(argv=None):
    p = argparse.ArgumentParser(prog="rageval", description="Offline RAG eval (BM25 + grounding check)")
    p.add_argument("corpus")
    p.add_argument("questions")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    res = run(a.corpus, a.questions, a.k)
    print(json.dumps(res, indent=2) if a.json else render(res))
    bad = res["retrieval"]["misses"] or any(not g["ok"] for g in res["grounding"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
