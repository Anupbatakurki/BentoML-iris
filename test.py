
import bentoml

# Get latest trained model
iris_clf_runner = (
    bentoml.sklearn.get("iris_clf:latest")
    .to_runner()
)

# Initialize locally
iris_clf_runner.init_local()

# Test Setosa
print("Setosa:", iris_clf_runner.predict.run([
    [5.1, 3.5, 1.4, 0.2]
]))

# Test Versicolor
print("Versicolor:", iris_clf_runner.predict.run([
    [6.0, 2.9, 4.5, 1.5]
]))

# Test Virginica
print("Virginica:", iris_clf_runner.predict.run([
    [6.5, 3.0, 5.8, 2.2]
]))
