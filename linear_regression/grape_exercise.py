import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def sklearn_predict(x, y):
    (x_train, x_test, y_train, y_test) = train_test_split(
        x, y, shuffle=False, train_size=0.7)

    X_train = [[n] for n in x_train]
    X_test = [[n] for n in x_test]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    return X_test, y_test, y_predicted, x_train, y_train


def main():

    df = pd.read_csv('grapes.csv', sep='\t')
    print(df)

    weight_vs_length = df['weight'].corr(df['length'])
    print(weight_vs_length)

    weight_vs_diametr = df['weight'].corr(df['diameter'])
    print(weight_vs_diametr)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Prediction of grapes weight')

    X_test, y_test, y_predicted, x_train, y_train = sklearn_predict(
        df['length'], df['weight'])

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(X_test, y_predicted, color='red')
    ax1.scatter(x=df['length'], y=df['weight'], color='blue')
    ax1.set_title('Length VS Weight')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Weight')
    r2_length = r2_score(y_test, y_predicted)
    print(f"Length R2: {r2_length}")

    X_test, y_test, y_predicted, x_train, y_train = sklearn_predict(
        df['diameter'], df['weight'])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(X_test, y_predicted, color='red')
    ax2.scatter(x=df['diameter'], y=df['weight'], color='green')
    ax2.set_title('Diameter VS Weight')
    ax2.set_xlabel('Diameter')
    ax2.set_ylabel('Weight')

    r2_diameter = r2_score(y_test, y_predicted)
    print(f"Diameter R2: {r2_diameter}")

    plt.show()


main()
