from huggingface_hub import HfApi

api = HfApi()
api.upload_large_folder(
    folder_path="/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/wanda_weightonly",
    repo_id="OriDragon2000/wanda_weightonly",   # 用户名/仓库名
    repo_type="dataset",
)
