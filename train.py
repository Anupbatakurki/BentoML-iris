
from sklearn import svm
from sklearn import datasets
import bentoml

# Load Iris dataset
iris = datasets.load_iris()

X, y = iris.data, iris.target

# Train SVM classifier
clf = svm.SVC(gamma="scale")

clf.fit(X, y)

# Check model predictions
print("Classes:", clf.classes_)
print("Unique predictions:", set(clf.predict(X)))

# Save model to BentoML Model Store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)

print(f"Model saved: {saved_model}")

