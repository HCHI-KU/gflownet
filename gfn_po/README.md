# GFlowPO BBH Reasoning

BBH 5-task 전용 독립 실행형 GFlowPO 연구 정리본이다. 이 저장소는 GFLOWPO 학습 로직을 유지하면서, BBH 태스크 입력/평가를 reasoning 기반 BBH 프로토콜에 맞춰 재현 가능하게 정리한 버전이다.

## Environment

권장 환경
- conda env: `rd_test`
- Python: `3.10`
- agent/eval model: `/home/work/jaeyoon/models/Meta-Llama-3-8B-Instruct`
- HF cache: `/home/work/jaeyoon/.hf_cache`

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

- `junmo/train.py`: 학습 엔트리포인트
- `junmo/trainer/gfn_em_ema_revision.py`: GFlowPO trainer
- `junmo/dataset_utils.py`: BBH split/data loader
- `junmo/bbh_eval_gfnpo.py`: reasoning -> extractor -> exact-match evaluator
- `run_bbh5_greater_compare_tmux.sh`: canonical 5-task launcher
- `tools/inspect_bbh_setup.py`: split/예시 sanity check
- `tools/analyze_reward_balance.py`: prior vs acc scale 분석

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

## Run

```bash
cd /home/work/jaeyoon/gflowpo_reasoning
bash run_bbh5_greater_compare_tmux.sh
```

검증 후 canonical full-run까지 자동으로 이어서 실행하려면:

```bash
cd /home/work/jaeyoon/gflowpo_reasoning
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
