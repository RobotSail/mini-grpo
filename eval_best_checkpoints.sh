#!/bin/bash
# Evaluate best checkpoints from each variant on GSM8K test set and compute KL divergence

set -e  # Exit on error

BASE_DIR="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds"

echo "============================================================"
echo "VARIANT 1 - GSM8K Test Evaluation"
echo "============================================================"
# python eval_gsm8k_parallel.py \
#     --checkpoints "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_1/tokens_1055958,${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_1/tokens_2269204,${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_1/hf_format/samples_1600.0_tokens_150109,${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_1/hf_format/samples_17396.0_tokens_1654787" \
#     --gpus 0,1,2,3 \
#     --output "${BASE_DIR}/variant1_best_gsm8k_test.json"

echo "============================================================"
echo "VARIANT 1 - KL Divergence"
echo "============================================================"
python eval_kl_multi_checkpoint.py \
    --grpo-adamw "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_1/tokens_1055958" \
    --grpo-muon "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_1/tokens_2269204" \
    --sft-adamw "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_1/hf_format/samples_1600.0_tokens_150109" \
    --sft-muon "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_1/hf_format/samples_17396.0_tokens_1654787" \
    --output "${BASE_DIR}/variant1_best_kl_results.json"

# echo "============================================================"
# echo "VARIANT 2 - GSM8K Test Evaluation"
# echo "============================================================"
# python eval_gsm8k_parallel.py \
#     --checkpoints "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_2/tokens_452930,${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_2/tokens_1515262,${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_2/hf_format/samples_1600.0_tokens_154149,${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_2/hf_format/samples_12660.0_tokens_1204845" \
#     --gpus 0,1,2,3 \
#     --output "${BASE_DIR}/variant2_best_gsm8k_test.json"
# 
# echo "============================================================"
# echo "VARIANT 2 - KL Divergence"
# echo "============================================================"
# python eval_kl_multi_checkpoint.py \
#     --grpo-adamw "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_2/tokens_452930" \
#     --grpo-muon "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_2/tokens_1515262" \
#     --sft-adamw "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_2/hf_format/samples_1600.0_tokens_154149" \
#     --sft-muon "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_2/hf_format/samples_12660.0_tokens_1204845" \
#     --output "${BASE_DIR}/variant2_best_kl_results.json"
# 
# echo "============================================================"
# echo "VARIANT 3 - GSM8K Test Evaluation"
# echo "============================================================"
# python eval_gsm8k_parallel.py \
#     --checkpoints "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_3/tokens_1053310,${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_3/tokens_1812578,${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_3/hf_format/samples_1600.0_tokens_152611,${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_3/hf_format/samples_18958.0_tokens_1801224" \
#     --gpus 0,1,2,3 \
#     --output "${BASE_DIR}/variant3_best_gsm8k_test.json"
# 
# echo "============================================================"
# echo "VARIANT 3 - KL Divergence"
# echo "============================================================"
# python eval_kl_multi_checkpoint.py \
#     --grpo-adamw "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-adamw_verify_3/tokens_1053310" \
#     --grpo-muon "${BASE_DIR}/qwen2-1.5b-gsm8k-grpo-muon_verify_3/tokens_1812578" \
#     --sft-adamw "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-adamw_verify_3/hf_format/samples_1600.0_tokens_152611" \
#     --sft-muon "${BASE_DIR}/qwen2-1.5b-gsm8k-sft-muon_verify_3/hf_format/samples_18958.0_tokens_1801224" \
#     --output "${BASE_DIR}/variant3_best_kl_results.json"
# 
# echo "============================================================"
# echo "ALL EVALUATIONS COMPLETE"
# echo "============================================================"
# echo "Results saved to:"
# echo "  ${BASE_DIR}/variant1_best_gsm8k_test.json"
# echo "  ${BASE_DIR}/variant1_best_kl_results.json"
# echo "  ${BASE_DIR}/variant2_best_gsm8k_test.json"
# echo "  ${BASE_DIR}/variant2_best_kl_results.json"
# echo "  ${BASE_DIR}/variant3_best_gsm8k_test.json"
# echo "  ${BASE_DIR}/variant3_best_kl_results.json"
