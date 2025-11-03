import os, glob, torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# ---------- 1. 重建模型 ----------
from model.SMENET import SMENET      # 换成你真实的模型类
model = SMENET(2).to(device)          # 初始化结构

# ---------- 2. 选出要加载的 .pth ----------
ckpt_dir = "save_model/yesno"
# 若你想手动指定，直接把下面这行赋值为 'save_model/yesno/xxx.pth'
def find_best_pth(path):
    pths = glob.glob(os.path.join(path, "best_yesno_model___0.9619.pth"))
    if not pths:
        raise FileNotFoundError(f"未在 {path} 找到 .pth 文件")
    # 假设文件名里带 “…_f1{score}.pth”
    def score_from_name(p):
        try:
            return float(os.path.basename(p).split("_f1")[1].replace(".pth", ""))
        except Exception:
            return -1.0
    return max(pths, key=score_from_name)

ckpt_path = find_best_pth(ckpt_dir)
print(f"加载模型权重：{ckpt_path}")

# ---------- 3. 加载参数 ----------
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ---------- 4. 运行验证集 ----------

torch.cuda.empty_cache()

tang_X = pd.read_csv('dataset/yesno/5w_train_rep32_1113.csv')
tang_Y = pd.read_csv('dataset/yesno/5w_Y.csv')
tang_X = tang_X.values.reshape(50544, 1, 2560)  # yesno: 50544
id_ec = pd.read_csv('dataset/yesno/enzymeAndNoEnzyme1.csv')
id_ec = id_ec['entry']
tang_Y = tang_Y.values
tang_Y = tang_Y.flatten()
# ==== 1. 数据划分（一次性） ====
# tang_X, tang_Y, id_ec 已经预先准备好
test_ratio = 0.2  # 20% 作为验证集
random_state = 20

train_X, test_X, train_Y, test_Y, train_ids8, test_ids8 = train_test_split(
    tang_X,
    tang_Y,
    id_ec,
    test_size=0.2,
    stratify=tang_Y,  # 保持类别分布一致
    random_state=random_state
)

batch_size = 64
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_Y = torch.tensor(test_Y, dtype=torch.long).to(device)
preds = []
with torch.no_grad():
    for i in range(0, len(test_X), batch_size):
        outputs = model(test_X[i:i + batch_size])
        _, batch_pred = torch.max(outputs, 1)
        preds.append(batch_pred.cpu())
predicted = torch.cat(preds)

# ---------- 5. 计算并输出指标 ----------
acc  = accuracy_score(test_Y.cpu(), predicted)
f1ma = f1_score(test_Y.cpu(), predicted, average='macro')

print(f"Acc: {acc*100:.2f}%  |  Macro-F1: {f1ma:.4f}")
