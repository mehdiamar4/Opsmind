# OpsMind Day 7 benchmark

Measured source files: `backend/evals/results_topk3.json`, `backend/evals/results_topk5.json`, `backend/evals/results_topk10.json`.
Each run contains 10 questions and uses the optional LLM judge.

## Measured results

| top_k | Recall@1 | Recall@k | Section@1 | FactMatch | LLM judge | Avg RAG latency (ms) | Avg estimated AI API cost/request (USD) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 100% | 100% | 100% | 80% | 100% | 1725.6537625 | 0.000053582 |
| 5 | 100% | 100% | 100% | 90% | 100% | 1659.8823541999998 | 0.000075662 |
| 10 | 100% | 100% | 100% | 90% | 100% | 1770.0031 | 0.000121792 |

## Monthly API cost projection

Measured average estimated AI API cost/request × monthly request count; this is a linear projection, not a new measurement.

| top_k | 1,000 requests | 10,000 requests | 100,000 requests | 500,000 requests |
| ---: | ---: | ---: | ---: | ---: |
| 3 | $0.053582000 | $0.535820000 | $5.358200000 | $26.791000000 |
| 5 | $0.075662000 | $0.756620000 | $7.566200000 | $37.831000000 |
| 10 | $0.121792000 | $1.217920000 | $12.179200000 | $60.896000000 |

## Scope

- Costs are estimated OpenAI API costs for the measured RAG flow, not total infrastructure costs.
- The optional LLM judge is paid but excluded from measured RAG latency and cost; PDF upload embeddings are also not logged in `ai_usage_events`.
- The projection assumes the measured question mix and per-request cost remain constant; it excludes database, hosting, and other operational costs.
- Source JSON values are copied without smoothing; chart bars show those same aggregates.
