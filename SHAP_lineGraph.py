import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import shap
import numpy as np
import matplotlib.pyplot as plt

# load data
sheetname = 'Negative'
sheetname2 = 'Positive'
sheetname3 = 'TRUE_All'
sheetname4 = 'FAKE_All'
excelname = 'Dataset_label.xlsx'
# sheetname = 'fake_negative'
# sheetname2 = 'fake_positive'
# sheetname3 = 'true_negative'
# sheetname4 = 'true_positive'
data_train = pd.read_excel(excelname, sheet_name='train')
data_test = pd.read_excel(excelname, sheet_name=sheetname)
data_test2 = pd.read_excel(excelname, sheet_name=sheetname2)
data_test3 = pd.read_excel(excelname, sheet_name=sheetname3)
data_test4 = pd.read_excel(excelname, sheet_name=sheetname4)

X_train = data_train.iloc[:, 1:21]
y_train = data_train.iloc[:, 0]
X_test = data_test.iloc[:, 1:21]
y_test = data_test.iloc[:, 0]
X_test2 = data_test2.iloc[:, 1:21]
y_test2 = data_test2.iloc[:, 0]
X_test3 = data_test3.iloc[:, 1:21]
y_test3 = data_test3.iloc[:, 0]
X_test4 = data_test4.iloc[:, 1:21]
y_test4 = data_test4.iloc[:, 0]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Test 1): {accuracy}')

y_pred2 = model.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f'Accuracy (Test 2): {accuracy2}')

y_pred3 = model.predict(X_test3)
accuracy3 = accuracy_score(y_test3, y_pred3)
print(f'Accuracy (Test 3): {accuracy3}')

y_pred4 = model.predict(X_test4)
accuracy4 = accuracy_score(y_test4, y_pred4)
print(f'Accuracy (Test 4): {accuracy4}')

explainer = shap.Explainer(model)

shap_values1 = explainer(X_test)
shap_values2 = explainer(X_test2)
shap_values3 = explainer(X_test3)
shap_values4 = explainer(X_test4)

def process_shap_values(shap_values):
    shap_values_lists = [[] for _ in range(6)]
    num_rows, num_cols = shap_values.data.shape
    for i in range(num_rows):
        for j in range(num_cols):
            data_value = int(shap_values.data[i, j])  # 转为整数索引
            shap_value = shap_values.values[i, j]
            if data_value >= 0 and data_value < len(shap_values_lists):  # 确保索引有效
                if shap_value > 0:  # 只选择大于0的 SHAP 值
                    shap_values_lists[data_value].append(shap_value)
    sum_values = [np.sum(lst) if lst else 0 for lst in shap_values_lists]
    std_values = [np.std(lst) if lst else 0 for lst in shap_values_lists]
    return sum_values, std_values

sum_values1, std_values1 = process_shap_values(shap_values1)
sum_values2, std_values2 = process_shap_values(shap_values2)
sum_values3, std_values3 = process_shap_values(shap_values3)
sum_values4, std_values4 = process_shap_values(shap_values4)

plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(14, 10))

x = np.arange(len(sum_values1))  # 动态计算长度

plt.errorbar(x, sum_values1, fmt='-o', capsize=5, capthick=2, elinewidth=2,
             label=sheetname, color='#4C72B0', ecolor='#FF9999', linestyle='-', marker='o', markersize=8)

plt.errorbar(x, sum_values2, fmt='-s', capsize=5, capthick=2, elinewidth=2,
             label=sheetname2, color='#55A868', ecolor='#FFCC00', linestyle='-', marker='s', markersize=8)

plt.errorbar(x, sum_values3, fmt='-^', capsize=5, capthick=2, elinewidth=2,
             label=sheetname3, color='#C44E52', ecolor='#6A3D9A', linestyle='-.', marker='^', markersize=8)

plt.errorbar(x, sum_values4, fmt='-d', capsize=5, capthick=2, elinewidth=2,
             label=sheetname4, color='#8172B3', ecolor='#FFA07A', linestyle='-.', marker='d', markersize=8)

plt.xticks(x, [f'Level {i}' for i in x], fontsize=14)
plt.xlabel('Sarcasm Levels', fontsize=18, weight='bold')
plt.ylabel('Sum Of SHAP Value', fontsize=18, weight='bold')


plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True)

plt.tight_layout()

plt.savefig('Figure_5.jpg', dpi=1200, bbox_inches='tight')

plt.show()
