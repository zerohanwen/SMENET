import os, glob, torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from tools import *
# ---------- 1. 重建模型 ----------
from model.SMENET import SMENET      # 换成你真实的模型类
model = SMENET(402).to(device)          # 初始化结构

# ---------- 2. 选出要加载的 .pth ----------
ckpt_dir = "save_model/level4"
# 若你想手动指定，直接把下面这行赋值为 'save_model/yesno/xxx.pth'
def find_best_pth(path):
    pths = glob.glob(os.path.join(path, "best_level4_model_ep_f10.9082.pth"))
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
# 3. 加载 EC 映射表，并调用多级评估函数
dict_ec_label = np.load('G:/acdemic\hanwen\data\ec_dict_level4.npy', allow_pickle=True).item()
# ---------- 3. 加载参数 ----------
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ---------- 4. 运行验证集 ----------

torch.cuda.empty_cache()

##tang_ec
tang_X = pd.read_csv('dataset/level4/tang_3b_33_X_2560.csv')
tang_Y = pd.read_csv('dataset/level4/Y_ec_label.csv')


tang_Y = tang_Y.values
tang_Y = tang_Y.flatten()
tang_Y -= 1
tang_X = tang_X.values.reshape(11354, 1, 2560)  # yesno: 50544
id_ec = pd.read_csv('dataset/level4/enzyme.csv')
id_ec = id_ec['id']

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

batch_size = 1024
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
# 4. 执行多级评估
true_labels = test_Y.cpu().numpy()
pred_labels = predicted.numpy()
metrics, result_df = evaluate_ec_levels(true_labels, pred_labels, dict_ec_label)