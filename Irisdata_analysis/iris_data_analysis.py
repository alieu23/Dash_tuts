import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import plotly.express as px

iris = load_iris()  # It returns simple dictionary like object with all data.
print("IRIS Dataset Size : ", iris.data.shape, iris.target.shape)
print("IRIS Flower Names : ", iris.target_names)
print("IRIS Flower Feature Names : ", iris.feature_names)

# Creating dataframe of total data
iris_df = pd.DataFrame(data=np.concatenate((iris.data, iris.target.reshape(-1, 1)), axis=1),
                       columns=(iris.feature_names + ['Flower Type']))
iris_df["Flower Name"] = [iris.target_names[int(i)] for i in iris_df["Flower Type"]]
print(iris_df.head())

chart1 = px.scatter(data_frame=iris_df,
                    x="sepal length (cm)",
                    y="petal length (cm)",
                    color="Flower Name",
                    size=[1.0] * 150,
                    title="sepal length (cm) vs petal length (cm) color-encoded by flower type")
chart1
