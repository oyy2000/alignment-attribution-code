model="llama2-7b-chat-hf"
method="wanda"
type="unstructured"
suffix="weightonly"
prune_data="GSM8K_direct_120"
sparsity_ratio=0
save_dir="out/$model/$type/${method}_${suffix}/$prune_data/"
nsamples=2

echo "Running with model: $model, method: $method, type: $type, sparsity_ratio: $sparsity_ratio, prune_data: $prune_data"

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $prune_data \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $type \
    --save $save_dir \
    --nsamples $nsamples \
    --eval_gsm8k \
    --eval_type random \
    # --save_mask \
    # --eval_zero_shot \
    # --dump_wanda_score \
