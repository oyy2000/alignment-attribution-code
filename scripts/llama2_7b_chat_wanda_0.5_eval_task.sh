model="llama2-7b-chat-hf"
method="wanda"
type="unstructured"
suffix="weightonly"
prune_data="GSM8K_cot0shot_goldreason"
sparsity_ratio=0.1
save_dir="out/$model/$type/${method}_${suffix}/$prune_data/"
nsamples=120

echo "Running with model: $model, method: $method, type: $type, sparsity_ratio: $sparsity_ratio, prune_data: $prune_data"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $prune_data \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $type \
    --save $save_dir \
    --nsamples $nsamples \
    --save_mask \
    --eval_attack \
    # --eval_zero_shot \
    # --dump_wanda_score \


CUDA_VISIBLE_DEVICES=0 python main.py --model llama2-7b-chat-hf --prune_method wanda --prune_data GSM8K_direct_120 --sparsity_ratio 0 --sparsity_type unstructured --save out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500 --nsamples 500 --eval_gsm8k --eval_type held_out --prompt_method direct
CUDA_VISIBLE_DEVICES=0 python main.py --model llama2-7b-chat-hf --prune_method wanda --prune_data GSM8K_direct_120 --sparsity_ratio 0 --sparsity_type unstructured --save out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500 --nsamples 500 --eval_gsm8k --eval_type held_out --prompt_method cot4shot

CUDA_VISIBLE_DEVICES=7 python main.py --model llama2-7b-chat-hf --prune_method wanda --prune_data GSM8K_direct_120 --sparsity_ratio 0 --sparsity_type unstructured --save out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500 --nsamples 500 --eval_gsm8k --eval_type held_out --prompt_method cot0shot_goldreason
