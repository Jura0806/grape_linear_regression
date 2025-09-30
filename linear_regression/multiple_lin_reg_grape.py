import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def main():

    df = pd.read_csv("grapes.csv", delimiter='\t')

    print(df.drop("color", axis=1).corr())

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Diameter')
    ax.set_ylabel('Length')
    ax.set_zlabel('Weight (Cube root)')
    ax.scatter(df['diameter'], df['length'], df['weight'] ** (1/3))

    X = df[["diameter", "length"]]
    y = df['weight'] ** (1/3)

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, train_size=0.7, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    plt.legend()
    plt.show()

    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    print(r2)


main()
