import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import shap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sheetname = 'true_negative'
data_train = pd.read_excel('Dataset_label.xlsx', sheet_name='train')
data_test = pd.read_excel('Dataset_label.xlsx', sheet_name=sheetname)
X_train = data_train.iloc[:, 1:21]
y_train = data_train.iloc[:, 0]
X_test = data_test.iloc[:, 1:21]
y_test = data_test.iloc[:, 0]
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

explainer = shap.Explainer(model)

shap_values = explainer(X_test)

max_indices = np.argmax(shap_values.values, axis=1)

max_values_data = np.array([shap_values.data[i, max_indices[i]] for i in range(shap_values.values.shape[0])])

counts = np.bincount(max_values_data, minlength=6)

print("Counts of each value (0 to 5) in max_values_data:")
for i, count in enumerate(counts):
    print(f"Value {i}: {count} times")

max_values_data = np.array([shap_values.data[i, max_indices[i]] for i in range(shap_values.values.shape[0])])

shap_values_lists = [[] for _ in range(6)]
num_rows, num_cols = shap_values.data.shape
for i in range(num_rows):
    for j in range(num_cols):
        data_value = shap_values.data[i, j]
        shap_value = shap_values.values[i, j]
        shap_values_lists[data_value].append(shap_value)
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(18, 18))
ax = plt.subplot(111, projection='polar')
colors = cm.viridis(np.linspace(0, 1, len(shap_values_lists)))

for i, (shap_values_list, color) in enumerate(zip(shap_values_lists, colors)):
    theta = np.linspace(0, 2 * np.pi, len(shap_values_list))
    r = np.array(shap_values_list)
    mask = r > 0
    theta = theta[mask]
    r = r[mask]
    ax.scatter(theta, r, label=f'Level {i}', color=color, s=100, alpha=0.7, edgecolors='w', linewidth=0.5)

r_ticks = np.linspace(0, max(max(r) for r in shap_values_lists), num=6)
ax.set_yticks(r_ticks)
ax.set_yticklabels([f'{int(tick)}' for tick in r_ticks], fontsize=16)

ax.set_xticklabels([])
ax.set_xticks([])
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

ax.legend(
    title='Sarcasm Level',
    bbox_to_anchor=(1.05, 1),
    loc='upper right',
    fontsize=18,
    title_fontsize=20,
    handletextpad=0.8,
    labelspacing=0.6,
    markerscale=1.2,
    borderpad=0.3,
    edgecolor='black'
)

title = 'SHAP Values for ' + sheetname + ' in Polar Coordinates'
ax.set_title(title, fontsize=24)
plt.savefig('Figure_6_sub1.jpg', dpi=1000, bbox_inches='tight')
# plt.show()
