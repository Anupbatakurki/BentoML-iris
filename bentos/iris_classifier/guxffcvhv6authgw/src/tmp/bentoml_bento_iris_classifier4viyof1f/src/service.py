
import bentoml
import numpy as np


# Load latest trained model
iris_model = bentoml.sklearn.load_model(
    "iris_clf:latest"
)


@bentoml.service
class IrisClassifier:

    @bentoml.api
    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> int:

        data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = iris_model.predict(data)

        return int(prediction[0])

