# rag-eval-lite

Independent demo, not affiliated with GRAN.

A small Python tool that checks a RAG setup without any API call. It does two things.

1. Retrieval check. It builds a BM25 index over a list of text chunks, runs a list of questions with a known correct chunk, and prints recall@k and MRR. Questions that miss are listed.
2. Citation check. For each generated answer, every sentence must cite an existing chunk, like `[c2]`, and must share enough content words with the cited chunk. Sentences with no citation, a made-up chunk id, or weak support are flagged.

It is about 100 lines of standard-library Python. There are no dependencies.

The data in `data/` is invented example data about a fictional study app. It is not real course content.

## Run

```
python -m rageval data/example_corpus.json data/example_questions.json
python -m rageval data/example_corpus.json data/example_questions.json -k 1 --json
python -m unittest discover -s tests -v
```

Exit code is 1 when there is any miss or flagged answer, so it can gate a CI job.

## Example output

```
Retrieval: recall@3 = 0.83, MRR = 0.83 (6 questions)
  MISS: 'When can I talk to a human?' wanted c6, got []
Answer a1: OK
Answer a2: PROBLEMS
  - no citation: Refunds are paid within 7 days.
Answer a3: PROBLEMS
  - unknown chunk c9: Quizzes show the correct answer after each question [c9].
```

The miss is a real limit of lexical search: the question says "talk to a human" and the chunk says "support answers messages". They share no word. An embedding retriever would likely catch it. This tool gives you the number to compare against.

## What was tested

- Tokenizer, BM25 ranking, empty result, recall and MRR, and the list of misses (unit tests).
- The citation check: valid answer, missing citation, unknown chunk id, weak support.
- The example run gives the same numbers every time.
- All of this runs offline. CI runs the same tests on Python 3.12.

## What was not tested

- Real data. Only the six invented chunks and six invented questions were used.
- Scale. The index is in memory and scores every chunk per query.
- Languages other than English. Stop words and tokenizing are English only (no accents handling).
- The citation check is a word-overlap rule. It cannot tell a paraphrase that is wrong from one that is right, and it is not an LLM judge.
- No embeddings, vector database or LLM was used, so there is no comparison with them yet.
