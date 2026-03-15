# GFlowPO BBH Debugging

BBH 5-task 기준으로 GFlowPO와 `debugging_gflownet.py`를 함께 비교할 수 있게 정리한 독립 실행형 저장소다. 목표는 다음 두 가지다.

- full GFlowPO BBH5 runner를 그대로 재현
- fixed meta prompt + fixed 5-shot에서 prompt policy만 학습하는 `debugging_gflownet.py`를 별도로 실행

## Environment

권장 환경
- conda env: `rd_test`
- Python: `3.10`
- agent/eval model: `meta-llama/Meta-Llama-3-8B-Instruct`

설치 순서
```bash
conda create -n rd_test python=3.10 -y
conda activate rd_test
pip install -U pip setuptools wheel
pip install -U torch==2.6.0 torchvision torchaudio accelerate transformers
pip install vllm==0.8.4
pip install flash-attn==2.7.1.post4 --no-build-isolation
pip install -r requirements.txt
```

## Repository Layout

- `debugging_gflownet.py`: fixed-context GFlowNet debugger
- `junmo/train.py`: 학습 엔트리포인트
- `junmo/trainer/gfn_em_ema_revision.py`: GFlowPO trainer
- `junmo/dataset_utils.py`: BBH split/data loader
- `junmo/bbh_eval_gfnpo.py`: reasoning -> extractor -> exact-match evaluator
- `run_bbh5_greater_compare_tmux.sh`: canonical 5-task launcher
- `tools/inspect_bbh_setup.py`: split/예시 sanity check
- `tools/analyze_reward_balance.py`: prior vs acc scale 분석

## Debugging Run

`debugging_gflownet.py`는 아래 요소를 고정하고 prompt policy만 학습한다.

- fixed meta prompt
- fixed 5-shot examples
- BBH train accuracy reward
- TB loss update

예시:

```bash
python debugging_gflownet.py \
  --task_names object_counting \
  --train_steps 30 \
  --batch_size 2 \
  --grad_acc_steps 1 \
  --num_example 5 \
  --fewshot_strategy first \
  --test_eval_period 5 \
  --run_name object_counting_debug
```

주요 출력:

- `debugging_gflownet_runs/<run_name>/<task>/config.json`
- `debugging_gflownet_runs/<run_name>/<task>/logs/metrics.jsonl`
- `debugging_gflownet_runs/<run_name>/<task>/logs/prompts.jsonl`
- `debugging_gflownet_runs/<run_name>/<task>/top_prompts.json`

## Canonical Setting

canonical BBH run 설정
- split: `train=50`, `test=100`
- reward: `log_reward = log_prior / gamma + log_accuracy / beta`
- `prior_reduction=mean`
- `policy_eval_num_samples=4`
- `policy_eval_start_step=100`
- `policy_eval_period=10`
- `condition_queue_samples=1`
- `condition_buffer_samples=1`
- `prior_chunk_size=2`
- `max_prompt_length=150`

task별 batch 정책
- `object_counting`, `movie_recommendation`, `hyperbaton`: `bs=4`, `ga=4`
- `causal_judgment`, `tracking_shuffled_objects_five_objects`: `bs=2`, `ga=8`

## Full GFlowPO Run

```bash
bash run_bbh5_greater_compare_tmux.sh
```

검증 후 canonical full-run까지 자동으로 이어서 실행하려면:

```bash
bash scripts/verify_and_run.sh
```

## Output

각 run 디렉터리에는 아래 파일이 생성된다.
- `train.jsonl`: current batch 기준 train metric
- `eval.jsonl`: queue 기반 held-out eval
- `final.jsonl`: 마지막 queue/final eval
- `bbh_*_debug.jsonl`: BBH reasoning/extractor debug

## Metric Definitions

- `train/*`: current batch on train split
- `eval/*`: queue-based held-out evaluation
- `policy_eval/*`: current policy fresh-sample held-out evaluation

우선 해석 지표
- `policy_eval/mean_test_acc`
- `policy_eval/median_test_acc`
- `policy_eval/top1_train_selected_test_acc`
- `eval/mean_acc`
- `eval/queue_prefill_fraction`
- `eval/queue_offline_fraction`

## Interpretation Guide

- `eval/mean_acc`가 높고 `policy_eval/mean_test_acc`가 낮으면 queue/history 효과가 큰 상태다.
- `train/max_acc`는 현재 batch에서 좋은 prompt가 나오고 있는지 보는 용도이고, 단독으로 일반화 성능을 의미하지 않는다.
- `policy_eval/train_test_gap_top1`가 크면 train-best 선택이 test로 일반화되지 않는다는 뜻이다.

## Known Limitations

- queue metric과 policy metric은 의미가 다르다.
- heavy task는 memory pressure가 크므로 task별 batch 정책을 유지해야 한다.
- `eval/max_acc` 단독으로 학습 성공을 판단하면 안 된다.
