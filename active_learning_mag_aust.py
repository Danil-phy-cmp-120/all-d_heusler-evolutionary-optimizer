import pandas as pd
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure, Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde, entropy
import pickle
import itertools


def shannon_entropy(composition_str):
    comp = Composition(composition_str)
    frac_comp = comp.fractional_composition.get_el_amt_dict()
    entropy = -np.sum([frac * np.log(frac) for frac in frac_comp.values()])
    return entropy

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('df.csv')

    conditions = [df['m_mart'] > 0.1,
                  df['m_aust'] > 0.1]
    df = df.loc[np.all(conditions, axis=0)]

    # Convert structure strings to pymatgen structures
    df['structures_aust'] = df['structures_aust'].map(lambda x: Structure.from_str(x, fmt='json'))
    df['structures_mart'] = df['structures_mart'].map(lambda x: Structure.from_str(x, fmt='json'))

    # Drop NaN values
    df = df.dropna()

    # Perform featurization
    df = StrToComposition().featurize_dataframe(df, "compositions")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")

    # df['tetragonal_ratio'] = df['structures_mart'].map(lambda x: x.lattice.c / x.lattice.a)
    df['shannon_entropy'] = df['compositions'].map(lambda x: shannon_entropy(x))
    df['volume'] = df['structures_aust'].map(lambda x: x.lattice.volume)

    # Drop unnecessary columns
    excluded = ['Unnamed: 0', 'composition', 'compositions', 'structures_aust', 'structures_mart',
                'm_mart', 'e_aust', 'e_mart']
    df = df.drop(excluded, axis=1)

    # Shuffle the DataFrame
    df = df.sample(frac=1)

    # Separate features (X) and target variable (y)
    y = df['m_aust'].values
    X = df.drop(columns=['m_aust'])

    mean_columns = [col for col in df.columns if "mean" in col]
    X = X[mean_columns]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define initial labeled dataset size
    start_point = 20

    # Initialize labeled dataset
    X_active = X_train.iloc[:start_point]
    y_active = y_train[:start_point]

    # Initialize model
    # regression = RandomForestRegressor(n_estimators=200)
    regression = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    regression.fit(X_active, y_active)

    step = 40
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

        if (abs(mean_absolute_error - previous_mean_absolute_error) < 0.04 and n > min_step):
            consecutive_iteration += 1
        else:
            consecutive_iteration = 0
        if consecutive_iteration == consecutive_step:
            break
        previous_mean_absolute_error = mean_absolute_error

        # Select samples
        # prediction_variance = np.var(y_pred_test)
        # threshold = np.percentile(prediction_variance, 80)
        # selected_indices = np.where(prediction_variance >= threshold)[0]

        selected_indices = np.where(absolute_error >= 0.04)[0]

        # Calculate the entropy of the predicted values
        # selected_indices = np.argsort(entropy(y_pred_test / np.sum(y_pred_test, keepdims=True)))[::-1][
        #                   :int(0.2 * len(X_test_batch))]  # Selecting top 20% with highest entropy

        # Add selected samples to the labeled dataset
        X_active = pd.concat([X_active, X_test_batch.iloc[selected_indices]])
        y_active = np.hstack([y_active, y_test_batch[selected_indices]])

        # Retrain the model with the new labeled samples
        regression.fit(X_active, y_active)

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
    ax.set_xlabel(r"$(\mu^{aust}_{pred} - \mu^{aust}_{DFT})/\mu^{aust}_{DFT}$ (%)", size=28, labelpad=3.0)
    ax.hist(100 * (regression.predict(X_train) - y_train) / y_train, bins=40, alpha=0.5, density=True, color="#138A07",
            label='Training Set')
    ax.hist(100 * (regression.predict(X_test) - y_test) / y_test, bins=40, alpha=0.5, density=True, color="#bc4749",
            label='Testing Set')

    # Plot density functions
    density_train = gaussian_kde(100 * (regression.predict(X_train) - y_train) / y_train)
    density_train.covariance_factor = lambda: 0.2
    density_train._compute_covariance()
    xs_train = np.linspace(-110, 110, 200)
    ax.plot(xs_train, density_train(xs_train), linewidth=6, color="#138A07")

    density_test = gaussian_kde(100 * (regression.predict(X_test) - y_test) / y_test)
    density_test.covariance_factor = lambda: 0.2
    density_test._compute_covariance()
    xs_test = np.linspace(-110, 110, 200)
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

    ax.set_xlim(-55, 105)
    fig.set_size_inches(10, 8)
    fig.savefig('histogram_mag_aust_active.png', transparent=False, bbox_inches='tight', dpi=300)

    # Plot variation of error with number of iterations
    errors = np.array(errors)
    fig = plt.figure()
    plt.plot(errors[:, 0], errors[:, 1], marker='o', linestyle='-')
    a, b = np.polyfit(errors[:, 0], errors[:, 1], 1)
    plt.plot(errors[:, 0], a * errors[:, 0] + b, marker='None', linestyle='-', color='orange', linewidth=3)
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean Absolute Error')
    plt.title('Variation of Error with Number of Iterations')
    plt.grid(True)
    fig.savefig('error_mag_aust_active.png', transparent=False, bbox_inches='tight', dpi=300)

    importances = regression.feature_importances_
    # included = np.asarray(included)
    included = X.columns.values
    indices = np.argsort(importances)[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(included[indices][0:10], importances[indices][0:10])
    plt.xticks(rotation='vertical')
    fig.savefig('feature_importances_tetr.png', transparent=False, bbox_inches='tight', dpi=300)
    print(importances[indices][0:10] * 100, sum(importances[indices][0:10]))

    importances = regression.feature_importances_
    # included = np.asarray(included)
    included = X.columns.values
    indices = np.argsort(importances)[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    names = [item.replace('MagpieData mean ', '') for item in included[indices][0:10]]

    print(importances[indices][0:10] * 100, sum(importances[indices][0:10]))
    print(names)

    bars = ax.bar(range(len(names)), importances[indices][0:10])

    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_yticklabels([int(label * 100) for label in ax.get_yticks()])

    ax.set_ylabel(r'Importance (%)', size=14, labelpad=3.0)
    ax.set_xlabel(r'Fitch ($x_i$)', size=14, labelpad=3.0)

    # Add names to the bars
    for i, (bar, name) in enumerate(zip(bars, names)):
        height = bar.get_height()
        if i == 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height / 2, name, ha='center', va='bottom',
                    rotation='vertical', color='white')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.006, name, ha='center', va='bottom',
                    rotation='vertical', color='black')

    ax.tick_params(axis='both',  # Apply parameters to both axes
                   which='major',  # Apply parameters to major ticks
                   direction='in',  # Draw ticks inside and outside the plot
                   pad=10,  # Distance between tick and label
                   labelsize=12,  # Label size
                   labelcolor='k',  # Label color
                   bottom=True,  # Draw ticks at the bottom
                   top=True,  # Draw ticks at the top
                   left=True,  # Draw ticks on the left
                   right=True,  # Draw ticks on the right
                   labelbottom=True,  # Display labels at the bottom
                   labeltop=False,  # No labels at the top
                   labelleft=True,  # Display labels on the left
                   labelright=False)  # No labels on the right

    fig.savefig('feature_importances_mag_aust.png', transparent=False, bbox_inches='tight', dpi=300)
    print(included[indices][0:10])


    pickle.dump(regression, open('model_mag_aust.pickle', "wb"))