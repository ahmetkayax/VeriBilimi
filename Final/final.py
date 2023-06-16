# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import warnings


# Veri setinin yüklenmesi
diabetes_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
                             header=None, names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Bağımsız değişkenleri ve hedef değişkeni ayırma
X = diabetes_data.iloc[:, :-1].values
y = diabetes_data.iloc[:, -1].values

# Veri setini eğitim ve test setleri olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Özellik ölçeklendirme
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Lojistik Regresyon sınıflandırma modeli
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Test seti üzerinde modelin performansını değerlendirme
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test setindeki doğruluk oranı: {:.2f}%".format(accuracy*100))

# Tüm veri seti üzerinde modelin performansını değerlendirme
y_pred_all = classifier.predict(X)
accuracy_all = accuracy_score(y, y_pred_all)
print("Tüm veri setindeki doğruluk oranı: {:.2f}%".format(accuracy_all*100))

# Yeni bir hasta verisi üzerinde modelin kullanılması
new_patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
new_patient = sc.transform(new_patient)
prediction = classifier.predict(new_patient)[0]

if prediction == 0:
    print("Hasta diyabet değil")
else:
    print("Hasta diyabet hastası")
    
# Diyabet hastası olma olasılığı yüzdesinin hesaplanması
prob = classifier.predict_proba(new_patient)[0][1] * 100
print("Diyabet hastası olma olasılığı: {:.2f}%".format(prob))




# ROC eğrisi çizimi
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
y_pred_proba
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC eğrisi (Alan = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix çizimi
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Diyabet Değil', 'Diyabet'])
plt.yticks(tick_marks, ['Diyabet Değil', 'Diyabet'])
plt.tight_layout()
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.show()


# Histogram çizimi
plt.figure()
sns.histplot(diabetes_data['plas'], kde=False, bins=10, color='skyblue')
plt.title('plas Değişkeninin Histogramı')
plt.xlabel('plas Değeri')
plt.ylabel('Frekans')
plt.show()

# İlk çıkan Mute Inline Plotting uyarısını kapatma 

warnings.filterwarnings('ignore')






