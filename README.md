# Mini GRPO

This repo contains a simple implementation of the GRPO algorithm which we use for training qwen/Qwen2-1.5B-Instruct in order to improve its skill on simple math problems.

Since the model is very lightweight, you can run this entire script on a single 80GB GPU.

**Example problem**:

* What's the difference between 67 and 41?
* What do you get when you subtract -839 from -642?
* What do 1000 and 1000 sum to?



## Expected Results

At the start of training, you should see the model have roughly a 5-15% pass rate for numbers in the range of [-1000, 1000]:

```py
=== Evaluation Scorecard ===
Total samples evaluated: 80
Pass@1: 5.0% above 50% | 5.0% at 100% (across 80 samples with 1 rollout(s) each)
```

After training the model for 5 epochs, you should see the pass rate climb to ~95% on the eval set:

```py
=== Evaluation Scorecard ===
Total samples evaluated: 80
Pass@1: 93.8% above 50% | 93.8% at 100% (across 80 samples with 1 rollout(s) each)
```

To replicate these results, run the example commands provided further down.



## Install:

Create a new Python environment and activate it:

```md
uv venv --python=3.12 
source .venv/bin/activate
uv pip install -e . && uv pip install flash-attn --no-build-isolation
```

## Usage

To use mini-grpo, you can follow these steps:

1. Create training data
2. Train the model
3. Evaluate it on the test set


### Create Training Data

You can create a training dataset consisting of math problems using the following command:

```bash
python cli.py generate-data \
    --num-problems 1000 \
    --min-num -1000 \
    --max-num 1000 \
    --test-split 0.2
```

This will create a training and test dataset of simple randomized math problems consisting of
addition and subtraction problems, which are written to a `generated_data/` directory by default.

The `--num-problems` flag controls the total size of the dataset, while the `--min-num` and `--max-num` flags are used to limit the range of integers used in the math problems.

You can use these increase/decrease the difficulty of the problems needed to solve the problem.

### Train the model with GRPO

You can launch GRPO training with the following command:

```bash
python cli.py train \
    --train-path generated_data/train.jsonl \
    --batch-size 4 \
    --group-size 8 \
    --inner-batch-size=16 \
    --inner-epochs 2 \
    --kl 0.01 \
    --clip-eps 0.2 \
    --lr 1e-6  \
    --eval-split 0.1 \
    --epochs 5 \
    --output-dir checkpoints
```

This will launch GRPO training of qwen/Qwen2-1.5B-Instruct on the math data we just generated.

### Evaluate the model on the test set

If you specified the `--test-split` flag during data generation, a test dataset was written to `generated_data/test.jsonl`.

You can evaluate models on the test dataset with the `eval` command.


**Evaluate the base model**
To evaluate the base model, simply pass `--model qwen/Qwen2-1.5B-Instruct`. 

```bash
# evaluate the base
python cli.py eval --eval-path generated_data/test.jsonl --model qwen/Qwen2-1.5B-Instruct
```
You should see a result like this:

```bash
=== Evaluation Results ===
Model: qwen/Qwen2-1.5B-Instruct
Samples: 200
Pass@1: 13.5% above 50% | 13.5% at 100%
```

**Evaluate the trained model**

To evaluate the trained model, simply pass a path to one of the checkpoints into the script:

```bash
# evaluate the base
python cli.py eval --eval-path generated_data/test.jsonl --model checkpoints/epoch_0
```

Results:

```py
=== Evaluation Results ===
Model: grpo-checkpoints/epoch_0/
Samples: 200
Pass@1: 91.5% above 50% | 91.5% at 100%
```


## Data Format

The data is formatted so the model learns to perform **addition** and **subtraction**, and always outputs its answers
using `<answer>...</answer>` tags. 


Here's some example messages from each task:

**Subtraction**:

```py
[system]: You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>
[user]: What do you get when you subtract -839 from -642?
```

**Addition**:

```py
[system]: You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>
[user]: If you have -437 and add -876, what is the total?
```

