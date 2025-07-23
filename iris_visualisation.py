import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
preds = pd.read_csv('iris_predictions.csv')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(preds['y_true'], preds['y_pred'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Setosa','Versicolor','Virginica'], yticklabels=['Setosa','Versicolor','Virginica'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion Iris')
plt.tight_layout()
plt.savefig('iris_confusion_matrix.png')
plt.show()
for i, name in enumerate(['Setosa','Versicolor','Virginica']):
    total = cm[i].sum()
    correct = cm[i,i]
    print(f"Classe {name}: {100*correct/total:.2f}%")
