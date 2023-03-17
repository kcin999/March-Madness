import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report
from machine_learning import load_model, get_data, clean_data

df = get_data('NCAA', True)
df = clean_data(df, True)

y = df['Result']
x = df.drop(columns=['Result'])

model = load_model()

predictions = model.predict(x)


print(precision_score(y, predictions, average=None))
print(recall_score(y, predictions, average=None))

print(classification_report(y, predictions))

cm = confusion_matrix(y, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()