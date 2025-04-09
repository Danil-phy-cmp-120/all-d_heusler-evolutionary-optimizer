import pandas as pd
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import pickle

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('df.csv')

    # Convert structure strings to pymatgen structures
    df['structures_aust'] = df['structures_aust'].map(lambda x: Structure.from_str(x, fmt='json'))
    df['structures_mart'] = df['structures_mart'].map(lambda x: Structure.from_str(x, fmt='json'))

    # Drop NaN values
    df = df.dropna()

    # Perform featurization
    df = StrToComposition().featurize_dataframe(df, "compositions")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")

    df['volume'] = df['structures_aust'].map(lambda x: x.lattice.volume)

    excluded_cols = ['Unnamed: 0', 'composition', 'compositions', 'structures_aust', 'structures_mart',
                     'm_aust', 'm_mart', 'e_aust', 'e_mart']
    df = df.drop(excluded_cols, axis=1)

    # Shuffle the DataFrame
    df = df.sample(frac=1)

    # Separate features (X) and target variable (y)
    y = df['volume'].values
    X = df.drop(columns=['volume'])

    mean_columns = [col for col in df.columns if "mean" in col]
    X = X[mean_columns]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define initial labeled dataset size
    start_point = 10

    # Initialize labeled dataset
    X_active = X_train.iloc[:start_point]
    y_active = y_train[:start_point]

    # Initialize model
    regression = RandomForestRegressor(n_estimators=200)
    regression.fit(X_active, y_active)

    step = 10
    errors = []
    consecutive_iteration = 0
    previous_mean_absolute_error = 0
    consecutive_step = 7
    min_step = 200
    for n in range(start_point, X_train.shape[0], step):
        # Test dataset
        X_test_batch = X_train.iloc[n:n + step]
        y_test_batch = y_train[n:n + step]

        # Make predictions on test dataset
        y_pred_test = regression.predict(X_test_batch)

        # Calculate absolute error
        absolute_error = np.abs(y_pred_test - y_test_batch)

        # Calculate mean absolute error
        mean_absolute_error = np.mean(absolute_error)
        print(n, mean_absolute_error, abs(mean_absolute_error - previous_mean_absolute_error), consecutive_iteration)
        errors.append([n, mean_absolute_error])

        if (abs(mean_absolute_error - previous_mean_absolute_error) < 1 and n > min_step):
            consecutive_iteration += 1
        else:
            consecutive_iteration = 0
        if consecutive_iteration == consecutive_step:
            break
        previous_mean_absolute_error = mean_absolute_error

        # Sort samples by absolute error and select the top ones
        selected_indices = np.where(absolute_error >= 1)[0]

        # Add selected samples to the labeled dataset
        X_active = pd.concat([X_active, X_test_batch.iloc[selected_indices]])
        y_active = np.hstack([y_active, y_test_batch[selected_indices]])

        # Retrain the model with the new labeled samples
        regression.fit(X_active, y_active)

    pickle.dump(regression, open('model_volume.pickle', "wb"))

    # Evaluate the model on the separate testing set
    # Evaluate the model on the separate testing set
    print("size:", X_active.shape[0])

    mse_test = mean_squared_error(y_test, regression.predict(X_test))
    r2_test = r2_score(y_test, regression.predict(X_test))
    print("Test MSE:", mse_test)
    print("Test R-squared:", r2_test)

    mse_train = mean_squared_error(y_train, regression.predict(X_train))
    r2_train = r2_score(y_train, regression.predict(X_train))
    print("Train MSE:", mse_train)
    print("Train R-squared:", r2_train)

    # Plot histogram of predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Deviation density function", size=28, labelpad=3.0)
    ax.set_xlabel("($V_{pred} - V_{DFT})/V_{DFT}$ (%)", size=28, labelpad=3.0)
    ax.hist(100 * (regression.predict(X_train) - y_train) / y_train, alpha=0.5, density=True, color="#138A07",
            label='Training Set')
    ax.hist(100 * (regression.predict(X_test) - y_test) / y_test, alpha=0.5, density=True, color="#bc4749",
            label='Testing Set')

    # Plot density functions
    density_train = gaussian_kde(100 * (regression.predict(X_train) - y_train) / y_train)
    density_train.covariance_factor = lambda: .25
    density_train._compute_covariance()
    xs_train = np.linspace(-10, 10, 200)
    ax.plot(xs_train, density_train(xs_train), linewidth=6, color="#138A07")

    density_test = gaussian_kde(100 * (regression.predict(X_test) - y_test) / y_test)
    density_test.covariance_factor = lambda: .25
    density_test._compute_covariance()
    xs_test = np.linspace(-10, 10, 200)
    ax.plot(xs_test, density_test(xs_test), linewidth=6, color="#bc4749")

    ax.axvline(0, linewidth=3, color='black', linestyle='--')

    ax.tick_params(axis='both',  # Применяем параметры к обеим осям
                   which='major',  # Применяем параметры к вспомогательным делениям
                   direction='in',  # Рисуем деления внутри и снаружи графика
                   # length = 10,    #  Длинна делений
                   # width = 2,     #  Ширина делений
                   # color = 'm',    #  Цвет делений
                   pad=10,  # Расстояние между черточкой и ее подписью
                   labelsize=24,  # Размер подписи
                   labelcolor='k',  # Цвет подписи
                   bottom=True,  # Рисуем метки снизу
                   top=True,  # сверху
                   left=True,  # слева
                   right=True,
                   labelbottom=True,  # Отображаем подписи снизу
                   labeltop=False,  # сверху нет
                   labelleft=False,  # слева да
                   labelright=False)  # справа нет

    legend = ax.legend(fontsize=22,
                       ncol=1,  # количество столбцов
                       loc='best',
                       # bbox_to_anchor=(0, -0.05),
                       facecolor='white',  # цвет области
                       framealpha=1,
                       # mode="expand",
                       borderaxespad=0.5,
                       # edgecolor = 'None',    #  цвет крайней линии
                       # title = 'External pressure:',    #  заголовок
                       # title_fontsize = 20   #  размер шрифта заголовка
                       )

    ax.set_xlim(-6.5, 6.5)
    fig.set_size_inches(10, 8)
    fig.savefig('histogram_volume_active.png', transparent=False, bbox_inches='tight', dpi=300)

    # Plot variation of error with number of iterations
    errors = np.array(errors)
    fig, ax = plt.subplots()
    ax.plot(errors[:, 0], errors[:, 1], marker='o', linestyle='-')
    a, b = np.polyfit(errors[:, 0], errors[:, 1], 1)
    ax.plot(errors[:, 0], a * errors[:, 0] + b, marker='None', linestyle='-', color='orange', linewidth=3)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Mean Absolute Error')

    ax.tick_params(axis='both',  # Применяем параметры к обеим осям
                   which='major',  # Применяем параметры к вспомогательным делениям
                   direction='in',  # Рисуем деления внутри и снаружи графика
                   # length = 10,    #  Длинна делений
                   # width = 2,     #  Ширина делений
                   # color = 'm',    #  Цвет делений
                   pad=10,  # Расстояние между черточкой и ее подписью
                   labelsize=12,  # Размер подписи
                   labelcolor='k',  # Цвет подписи
                   bottom=True,  # Рисуем метки снизу
                   top=True,  # сверху
                   left=True,  # слева
                   right=True,
                   labelbottom=True,  # Отображаем подписи снизу
                   labeltop=False,  # сверху нет
                   labelleft=True,  # слева да
                   labelright=False)  # справа нет

    ax.grid(True)
    fig.savefig('error_vol_active.png', transparent=False, bbox_inches='tight', dpi=300)

    importances = regression.feature_importances_
    # included = np.asarray(included)
    included = X.columns.values
    indices = np.argsort(importances)[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(included[indices][0:10], importances[indices][0:10])
    plt.xticks(rotation='vertical')
    fig.savefig('feature_importances_vol.png', transparent=False, bbox_inches='tight', dpi=300)
    print(included[indices][0:10])

