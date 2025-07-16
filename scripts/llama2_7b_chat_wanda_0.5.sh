model="llama2-7b-chat-hf"
method="wanda"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/${method}_${suffix}/align/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res \
