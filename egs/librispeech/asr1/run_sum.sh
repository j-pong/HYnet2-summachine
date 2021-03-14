#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_sum_transformer.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

# --speed_perturb_factors "0.9 1.0 1.1" \

./sum.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" "$@"
