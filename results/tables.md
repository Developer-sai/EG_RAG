# Results (10 questions)

## Summary

| generator   | system         |   questions |   errors |   accuracy |   doc_hit |   page_hit |   para_hit |   span_f1 |   status_correct |   false_grounding |   false_abstention |   conflict_detected |   latency_s |   unsupported_rate |
|:------------|:---------------|------------:|---------:|-----------:|----------:|-----------:|-----------:|----------:|-----------------:|------------------:|-------------------:|--------------------:|------------:|-------------------:|
| open_weight | S1             |          10 |        0 |        0.4 |   nan     |    nan     |    nan     |   nan     |            nan   |               nan |              nan   |                 nan |       0.006 |                nan |
| open_weight | S2             |          10 |        0 |        1   |   nan     |    nan     |    nan     |   nan     |            nan   |               nan |              nan   |                 nan |       0.455 |                nan |
| open_weight | S3             |          10 |        0 |        1   |     1     |      1     |      1     |     0.562 |              1   |                 0 |                0   |                   0 |       0.449 |                  0 |
| open_weight | S4             |          10 |        0 |        1   |     0.5   |      0.5   |      0.5   |     0.466 |              0.7 |                 0 |                0.3 |                   0 |       0.683 |                  0 |
| open_weight | S4v2           |          10 |        0 |        1   |     0.875 |      0.875 |      0.875 |     0.813 |              0.9 |                 0 |                0.1 |                   1 |       0.578 |                  0 |
| open_weight | S5             |          10 |        0 |        0.5 |     0.5   |      0.5   |      0.5   |     0.466 |              0.7 |                 0 |                0.3 |                   0 |       0.729 |                  0 |
| open_weight | S5_no_gate     |          10 |        0 |        0.5 |     0.5   |      0.5   |      0.5   |     0.466 |              0.7 |                 0 |                0.3 |                   0 |       0.881 |                  0 |
| open_weight | S5_no_quote    |          10 |        0 |        0.5 |     0.5   |      0.5   |      0.5   |     0.292 |              0.7 |                 0 |                0.3 |                   0 |       0.95  |                  0 |
| open_weight | S5_no_verifier |          10 |        0 |        0.9 |     1     |      1     |      1     |     0.937 |              1   |                 0 |                0   |                   0 |       0.525 |                  0 |
| open_weight | S5v2           |          10 |        0 |        0.8 |     0.875 |      0.875 |      0.875 |     0.813 |              0.9 |                 0 |                0.1 |                   1 |       0.936 |                  0 |
| proprietary | S1             |          10 |        0 |        0.5 |   nan     |    nan     |    nan     |   nan     |            nan   |               nan |              nan   |                 nan |       0.008 |                nan |
| proprietary | S2             |          10 |        0 |        0.9 |   nan     |    nan     |    nan     |   nan     |            nan   |               nan |              nan   |                 nan |       0.45  |                nan |
| proprietary | S3             |          10 |        0 |        1   |     1     |      1     |      1     |     0.562 |              1   |                 0 |                0   |                   0 |       0.446 |                  0 |
| proprietary | S4             |          10 |        0 |        1   |     0.375 |      0.375 |      0.375 |     0.321 |              0.6 |                 0 |                0.4 |                   0 |       0.752 |                  0 |
| proprietary | S4v2           |          10 |        0 |        1   |     0.875 |      0.875 |      0.875 |     0.728 |              0.9 |                 0 |                0.1 |                   1 |       0.594 |                  0 |
| proprietary | S5             |          10 |        0 |        0.5 |     0.375 |      0.375 |      0.375 |     0.321 |              0.6 |                 0 |                0.4 |                   0 |       0.733 |                  0 |
| proprietary | S5_no_gate     |          10 |        0 |        0.5 |     0.375 |      0.375 |      0.375 |     0.321 |              0.6 |                 0 |                0.4 |                   0 |       0.813 |                  0 |
| proprietary | S5_no_quote    |          10 |        0 |        0.5 |     0.375 |      0.375 |      0.375 |     0.218 |              0.6 |                 0 |                0.4 |                   0 |       0.735 |                  0 |
| proprietary | S5_no_verifier |          10 |        0 |        0.9 |     1     |      1     |      1     |     0.853 |              1   |                 0 |                0   |                   0 |       0.424 |                  0 |
| proprietary | S5v2           |          10 |        0 |        0.8 |     0.875 |      0.875 |      0.875 |     0.728 |              0.9 |                 0 |                0.1 |                   1 |       0.583 |                  0 |

## Per question

### open_weight: answer judged correct, and evidence status

| qid    | type         | S1   | S2   | S3   | S4   | S4v2   | S5   | S5v2   | S4 status   | S4v2 status   | S5 status   | S5v2 status   |
|:-------|:-------------|:-----|:-----|:-----|:-----|:-------|:-----|:-------|:------------|:--------------|:------------|:--------------|
| D3-01  | single       | yes  | yes  | yes  | yes  | yes    | no   | no     | grounded    | grounded      | grounded    | grounded      |
| D3-05  | single       | no   | yes  | yes  | yes  | yes    | yes  | yes    | grounded    | grounded      | grounded    | grounded      |
| D3-06  | single       | no   | yes  | yes  | yes  | yes    | no   | yes    | not_found   | grounded      | not_found   | grounded      |
| D3-09  | single       | no   | yes  | yes  | yes  | yes    | yes  | yes    | conflict    | conflict      | conflict    | conflict      |
| SYN-03 | single       | no   | yes  | yes  | yes  | yes    | yes  | no     | grounded    | not_found     | grounded    | not_found     |
| SYN-08 | single       | no   | yes  | yes  | yes  | yes    | no   | yes    | not_found   | grounded      | not_found   | grounded      |
| SYN-09 | single       | yes  | yes  | yes  | yes  | yes    | no   | yes    | conflict    | conflict      | conflict    | conflict      |
| SYN-10 | conflict     | no   | yes  | yes  | yes  | yes    | no   | yes    | not_found   | conflict      | not_found   | conflict      |
| SYN-25 | unanswerable | yes  | yes  | yes  | yes  | yes    | yes  | yes    | not_found   | not_found     | not_found   | not_found     |
| SYN-32 | unanswerable | yes  | yes  | yes  | yes  | yes    | yes  | yes    | not_found   | not_found     | not_found   | not_found     |

### proprietary: answer judged correct, and evidence status

| qid    | type         | S1   | S2   | S3   | S4   | S4v2   | S5   | S5v2   | S4 status   | S4v2 status   | S5 status   | S5v2 status   |
|:-------|:-------------|:-----|:-----|:-----|:-----|:-------|:-----|:-------|:------------|:--------------|:------------|:--------------|
| D3-01  | single       | yes  | yes  | yes  | yes  | yes    | yes  | yes    | grounded    | grounded      | grounded    | grounded      |
| D3-05  | single       | no   | yes  | yes  | yes  | yes    | yes  | yes    | grounded    | grounded      | grounded    | grounded      |
| D3-06  | single       | no   | yes  | yes  | yes  | yes    | no   | yes    | not_found   | grounded      | not_found   | grounded      |
| D3-09  | single       | yes  | yes  | yes  | yes  | yes    | yes  | yes    | conflict    | conflict      | conflict    | conflict      |
| SYN-03 | single       | no   | yes  | yes  | yes  | yes    | no   | no     | not_found   | not_found     | not_found   | not_found     |
| SYN-08 | single       | no   | yes  | yes  | yes  | yes    | no   | yes    | not_found   | grounded      | not_found   | grounded      |
| SYN-09 | single       | yes  | yes  | yes  | yes  | yes    | no   | yes    | conflict    | conflict      | conflict    | conflict      |
| SYN-10 | conflict     | no   | no   | yes  | yes  | yes    | no   | no     | not_found   | conflict      | not_found   | conflict      |
| SYN-25 | unanswerable | yes  | yes  | yes  | yes  | yes    | yes  | yes    | not_found   | not_found     | not_found   | not_found     |
| SYN-32 | unanswerable | yes  | yes  | yes  | yes  | yes    | yes  | yes    | not_found   | not_found     | not_found   | not_found     |

