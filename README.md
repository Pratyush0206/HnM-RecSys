# Two-Stage Fashion Recommender

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-2E8B57)
![FAISS](https://img.shields.io/badge/FAISS-CPU-0467DF)
![FastAPI](https://img.shields.io/badge/FastAPI-0.159-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
[![CI](https://github.com/Pratyush0206/HnM-RecSys/actions/workflows/docker-smoke.yml/badge.svg?branch=main)](https://github.com/Pratyush0206/HnM-RecSys/actions)
![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen)
![MAP@12](https://img.shields.io/badge/MAP%4012-0.0304%20(held--out%20test)-blueviolet)

[![Results](https://img.shields.io/badge/Results-jump-2563eb?style=for-the-badge)](#results)
[![Design](https://img.shields.io/badge/System%20design-jump-7c3aed?style=for-the-badge)](#system-design)
[![Ablations](https://img.shields.io/badge/Ablations-jump-db2777?style=for-the-badge)](#ablations)
[![Latency](https://img.shields.io/badge/Latency-jump-d97706?style=for-the-badge)](#serving-performance)
[![Run](https://img.shields.io/badge/Run%20it-jump-059669?style=for-the-badge)](#run-it)

A production-style recommendation system on the H&M Personalized Fashion Recommendations dataset: multi-source candidate generation, a LightGBM LambdaRank ranker, leakage-safe evaluation, and a FastAPI service with measured latency.

**Stack:** Python, PyTorch, FAISS, LightGBM, pandas, FastAPI, Docker, GitHub Actions, pytest

## At a glance

- On the held-out test week (scored once): MAP@12 **0.0304**, **8.6x** global popularity and **3.6x** recent popularity; the ranker adds **+21%** over the candidate order alone (paired-bootstrap interval excludes zero).
- **Leakage-safe by construction:** five consecutive weekly cuts, a runtime check that rejects any history reaching the cut date, and a held-out test week that is scored exactly once.
- **Ablations reported honestly:** feature, candidate-source and objective ablations with bootstrap intervals, including a two-tower feature gain that did not replicate at larger scale.
- **Serving:** online recommendations match the offline ranking code on all 5,000 stored customers (0 mismatches); p50 5.2 ms in-process, 7.6 ms over HTTP, 419 requests/s with 4 clients on a laptop CPU.
- **Engineering:** 109 automated tests, pinned dependencies, a Dockerfile smoke-tested in CI.

## Results

Data: a seeded 200,000-customer sample of the last 16 weeks of transactions. Metric: MAP@12. Absolute values are not comparable to the Kaggle leaderboard, which uses the full dataset and a different protocol.

**Held-out test week (2020-09-16 to 2020-09-22), scored once**

| Method | MAP@12 | vs global popularity |
|---|---|---|
| Global popularity | 0.00354 | 1.0x |
| Recent popularity (1-week window, chosen on training weeks) | 0.00838 | 2.4x |
| Candidate generation order (no ranker) | 0.02524 | 7.1x |
| **LightGBM LambdaRank ranker** | **0.03042** | **8.6x** |

- 23,490 customers scored; candidate recall 0.2141.
- Ranker minus candidate order: +0.00518, bootstrap interval [+0.00425, +0.00611], about +21%.
- The model was frozen before the test week was touched: full 40 features, LambdaRank, default parameters, tree count from early stopping on the validation week. Design choices were made on the 100k development sample. `hmrec/final_eval.py` refuses to run a second time, and the result file stores the model's SHA-256 hash.

**Validation week (2020-09-09 to 2020-09-15), development numbers**

| Method | MAP@12 | vs global popularity |
|---|---|---|
| Global popularity | 0.00384 | 1.0x |
| Recent popularity (1-week window, chosen on training weeks) | 0.00593 | 1.5x |
| Candidate generation order (no ranker) | 0.02576 | 6.7x |
| LightGBM LambdaRank ranker | 0.03008 | 7.8x |

- 24,272 customers scored; ranker minus candidate order +0.00433, interval [+0.00347, +0.00514].
- Candidate recall on the validation week is 0.2034, which caps what any ranker can achieve.
- The tree count (311) and early stopping use the validation week, so these numbers are optimistic by construction. The test score (0.03042) is close to the validation score (0.03008), so there is no visible overfitting to the validation week.

## System design

```
transactions -> time cuts -> candidate generation (4 sources) -> features -> LightGBM LambdaRank -> top 12
                                                                                  |
                                     offline batch jobs  ->  compact store -> FastAPI service
```

**Evaluation protocol.** Five consecutive 7-day label weeks end on 2020-09-22: three training weeks, one validation week and one test week. For every cut, the history used for candidates and features ends exactly where its label week begins. A check raises an error if any history contains a date at or after the cut, and tests prove it fires. Differences are reported with a paired bootstrap over customers (1,000 resamples, seed 42).

**Candidate generation** (200 per customer, merged and de-duplicated):
- repurchase of recently bought items
- item-to-item co-purchase neighbours
- two-tower retrieval with a FAISS index
- recent popularity (also the cold-start source)

The two-tower model is a shallow PyTorch model: customer and item embedding tables (dimension 32), L2-normalised, trained for 20 epochs on CPU with an in-batch sampled softmax (temperature 0.1), a logQ popularity correction and false-negative masking.

**Ranking.** 40 features: customer activity and recency, item popularity over several windows, customer-item pair features, article and customer metadata, the two-tower score and rank, and each source's rank. LightGBM `LGBMRanker` with the LambdaRank objective, grouped per (week, customer), early stopping on the validation week.

**Serving.** Candidate generation and feature computation are offline batch jobs. A compact store holds each customer's candidates and precomputed features. The online path looks a customer up, scores the candidates on a single thread, sorts and returns the top 12; unknown customers get the popularity list.

Endpoints: `GET /health`, `GET /recommend/{customer_id}`, `POST /recommend` (1 to 100 ids).

## Ablations

Each row drops one component and retrains; "full minus variant" is positive when the full model is better. The full table comes from the 100,000-customer development sample, where the design was chosen; the 200k run replicates the two feature groups. All ablation numbers are validation-week numbers.

100k development sample (full model MAP@12 0.03033):

| Variant | MAP@12 | Full minus variant [bootstrap CI] |
|---|---|---|
| Drop metadata features | 0.02750 | +0.00284 [+0.00183, +0.00387] |
| Drop two-tower features | 0.02853 | +0.00180 [+0.00097, +0.00264] |
| Drop customer-item pair features | 0.02978 | +0.00055 [-0.00035, +0.00139] |
| Drop recent item popularity features | 0.03022 | +0.00011 [-0.00061, +0.00078] |
| Classifier objective instead of LambdaRank | 0.02922 | +0.00111 [+0.00017, +0.00191] |
| Remove repurchase candidates | 0.02768 | +0.00265 [+0.00150, +0.00378] |
| Remove co-purchase candidates | 0.02935 | +0.00099 [+0.00019, +0.00176] |
| Remove two-tower candidates | 0.02878 | +0.00155 [+0.00082, +0.00229] |
| Remove popularity candidates | 0.02685 | +0.00348 [+0.00246, +0.00451] |

200k replication (full model 0.03008):

| Variant | MAP@12 | Full minus variant [bootstrap CI] |
|---|---|---|
| Drop metadata features | 0.02842 | +0.00166 [+0.00090, +0.00246] |
| Drop two-tower features | 0.02956 | +0.00053 [-0.00013, +0.00118] |

**Takeaways**
- Metadata features are the most useful feature group, and the gain replicates at 200k, though smaller.
- The two-tower feature gain did not replicate at 200k (the interval includes zero), so I treat it as small and not reliably detectable. The model was trained with one seed.
- Among candidate sources, popularity (the cold-start source) and repurchase matter most.
- Pair features and recent item popularity features show no reliable gain.
- Source removal is approximate: a removed source's unique candidates are dropped, but other sources do not refill the freed slots.

## Serving performance

Windows 11 laptop (8 logical cores), one server process, 200 candidates x 40 features, 311 trees, 2,000 timed requests after a 200-request warm-up, client and server on the same machine.

| Setting | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| In-process (no HTTP) | 5.15 ms | 5.70 ms | 6.20 ms | n/a |
| HTTP, 1 client | 7.61 ms | 9.04 ms | 10.53 ms | 129 req/s |
| HTTP, 4 clients | 8.88 ms | 12.55 ms | 25.69 ms | 419 req/s |

Model scoring is about 99% of the in-process time. Two in-process runs agreed within 1% at p50. The Dockerfile is built and smoke-tested in GitHub Actions with a small synthetic store and model; latency was measured locally without Docker.

## Run it

Requires Python 3.14 (developed on 3.14.3) and the Kaggle files `transactions_train.csv`, `customers.csv` and `articles.csv` in `data/raw/` (the data is not included).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

Pipeline for the 200k sample (building candidates takes about 35 minutes per week on a laptop CPU):

```powershell
python -m hmrec.prepare --out-dir data/processed_200k_16w --weeks 16 --customers 200000
python -m hmrec.metadata --processed-dir data/processed_200k_16w
python -m hmrec.evaluate --processed-dir data/processed_200k_16w
python -u -m hmrec.ranker_data --processed-dir data/processed_200k_16w --epochs 20 --dim 32 --k 30 --max-per-customer 200
python -m hmrec.features --processed-dir data/processed_200k_16w
python -m hmrec.ranker --processed-dir data/processed_200k_16w
python -m hmrec.ranker --processed-dir data/processed_200k_16w --drop metadata
python -m hmrec.ranker --processed-dir data/processed_200k_16w --drop twotower
python -m hmrec.ablate --processed-dir data/processed_200k_16w --variants no_metadata no_twotower
python -m hmrec.report --processed-dir data/processed_200k_16w
```

Serving and benchmarks:

```powershell
python -m hmrec.store --processed-dir data/processed_200k_16w --customers 5000
python -m hmrec.parity --processed-dir data/processed_200k_16w
uvicorn hmrec.serve:app --port 8000
python -m hmrec.bench_serving --processed-dir data/processed_200k_16w --mode inprocess
python -m hmrec.bench_serving --processed-dir data/processed_200k_16w --mode http --url http://127.0.0.1:8000
```

The serving snapshot is built from the validation week; `hmrec.store` refuses to build from the test week. The one-shot test evaluation is `python -u -m hmrec.final_eval --processed-dir data/processed_200k_16w`, and the test-week popularity baseline is `python -m hmrec.final_baselines --processed-dir data/processed_200k_16w`.

## Limitations

- One validation week drives the tree count, early stopping and comparisons. The test week was scored once after all choices were frozen, but it is still a single week, and the intervals do not capture week-to-week variation.
- Bootstrap intervals cover customer sampling only, not variation across training seeds or weeks. The two-tower model used a single seed.
- The data is a 200,000-customer sample of the last 16 weeks, trained and evaluated on a laptop CPU.
- Candidate recall is about 0.20, which limits the ranker.
- Candidate-source ablations are approximate and were run on the 100k sample only.
- The service is a demo: one process, no authentication, a 5,000-customer sample, a validation-week snapshot and precomputed features. Latency was measured on one laptop with client and server on the same machine.

## Repository layout

- `hmrec/`: splits, metrics, candidates, two-tower, features, ranker, ablations, store, recommender, service, benchmarks, report
- `tests/`: 109 automated test runs
- `Dockerfile`, `requirements.txt` (full project), `requirements-serve.txt` (serving image)
- `.github/workflows/docker-smoke.yml`: image build and smoke test