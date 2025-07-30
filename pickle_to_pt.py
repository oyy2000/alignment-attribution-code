import glob, torch, pickle, os, io

pkls = glob.glob("out/**/wanda_score/**/*.pkl", recursive=True)

for p in pkls:
    # 1. 读取旧 pickle
    with open(p, "rb") as f:
        t = pickle.load(f)

    # 2. 生成新文件名：xxx.pkl → xxx_torch.pt
    new_p = p.replace(".pkl", "_torch.pt")

    # 3. 存为 torch 格式（到 CPU 再存）
    torch.save(t.cpu(), new_p)

    print(f"Converted {p}  →  {new_p}")
