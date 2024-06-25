import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report)
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plot

dataInti = pd.read_csv(r'data/label.csv')
dataInti = dataInti[['klasifikasi','cleaned_text']]

y = dataInti['klasifikasi']
x = dataInti['cleaned_text']

xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=112)

print(xtrain.shape,xtest.shape)

tf = TfidfVectorizer()
tftrain = tf.fit_transform(xtrain)
tftest = tf.transform(xtest)

mnb = MultinomialNB(alpha= 0.8, fit_prior=True, force_alpha=True)
mnb.fit(tftrain,ytrain)
ypred = mnb.predict(tftest)

acc = accuracy_score(ytest, ypred)
cr = classification_report(ytest, ypred)
cm = confusion_matrix(ytest, ypred)

print("MNB Accuracy", acc)
print("Confussion Matrix \n\n", cm)
print("Classification Report \n\n", cr)

sns.heatmap(cm,square=False, annot=True, cbar=False,
            cmap='RdBu', xticklabels=['Negatif','Netral','Positif'],
            yticklabels=['Negatif','Netral','Positif'], fmt='d')
plot.xlabel('Prediction')
plot.ylabel('True')
plot.show()