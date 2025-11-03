from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

def evaluate_ec_levels(true_labels, pred_labels, ec_dict, verbose=True):
    """
    修复版：支持 EC 号层级评估，确保 Level 4 Acc == 整体 acc。
    """
    result = pd.DataFrame({
        'ec_label': true_labels,
        'predicted_x': pred_labels
    })

    # 第一步：计算 Level 4 Acc（和主任务一致）
    level4_acc = (result['ec_label'] == result['predicted_x']).mean()
    level4_f1 = f1_score(true_labels, pred_labels, average='macro')
    #if verbose:
        #print(f"[Level 4] Accuracy: {level4_acc * 100:.2f}% | Macro-F1: {level4_f1:.4f}")

    # 第二步：映射为 EC 字符串
    result['ec_number'] = result['ec_label'].map(ec_dict).fillna('0.0.0.0')
    result['predicted_level4'] = result['predicted_x'].map(ec_dict).fillna('0.0.0.0')

    metrics = {
        'level_4_accuracy': level4_acc,
        'level_4_f1': level4_f1
    }

    # 第三步：分级评估（1-4层）
    for i, level_name in enumerate(['first', 'second', 'third', 'fourth']):
        true_col = f'ec_{level_name}_layer'
        pred_col = f'prediction_{level_name}_layer'
        correct_col = f'{level_name}_layer_correct'

        result[true_col] = result['ec_number'].apply(lambda x: x.split('.')[i] if len(x.split('.')) > i else '')
        result[pred_col] = result['predicted_level4'].apply(lambda x: x.split('.')[i] if len(x.split('.')) > i else '')

        # print(f"[{level_name} level] true_col has {result[true_col].nunique()} unique classes")

        result[correct_col] = result[true_col] == result[pred_col]
        acc = result[correct_col].mean()
        f1 = f1_score(result[true_col], result[pred_col], average='macro', zero_division=0)

        metrics[f'level_{i+1}_accuracy'] = acc
        metrics[f'level_{i+1}_f1'] = f1

        if verbose:
            print(f"[Level {i+1}] Accuracy: {acc * 100:.2f}% | Weighted F1: {f1:.4f}")

    return metrics, result
