---
license: other
pretty_name: Visual Intelligence Leaderboard — Ground Truth (PRIVATE)
viewer: false
---

# Visual Intelligence Leaderboard — Ground Truth (PRIVATE)

🔒 **Do not make public.** These are the withheld answer keys for the public benchmark
[`amolharsh/visual-intelligence-leaderboard`](https://huggingface.co/datasets/amolharsh/visual-intelligence-leaderboard).
Making this repository public would leak the benchmark and make every leaderboard score gameable.

## Layout

```
<subset>/ground_truth.jsonl
```

One JSON object per line:

```json
{"question_id": "t1_3d_visual_spatial_easy_0000", "answer": 1}
```

Match `question_id` against the public `questions.jsonl` to score a prediction.
`answer` is the reference value in the format named by that question's `answer_type`
(integer, MCQ letter, or text).

| Subset | Answers |
|---|---:|
| `dysm_2d_v1` | 3,000 |
| `dysm_3d_v1` | 1,500 |
| `minds_eye_fresh_v1` | 799 |
