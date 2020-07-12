import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from functools import reduce

def compare_tuning(df1, feature_col_1, feature_col_2, metric_col, keep_fixed):
    df = df1[reduce(lambda x, y: x&y, 
        [df1[col] == val for (col, val) in keep_fixed.items()])]
    feature_col_1_values = df[feature_col_1].unique().tolist()

    colors = ['blue', 'green', 'orange', 'red', 'black']
    for i, x in enumerate(feature_col_1_values):
        plt.plot(df[df[feature_col_1] == x][feature_col_2], 
                 df[df[feature_col_1] == x][metric_col], 
                 label=f'{feature_col_1}={x}', color=colors[i%len(colors)])
    plt.title(f'({feature_col_1}, {feature_col_2}) vs {metric_col} ')
    plt.legend()
    plt.xlabel(feature_col_2)
    plt.ylabel(metric_col)

def compare_two_features_3D(df1, feature_col_1, feature_col_2, metric_col, keep_fixed):
    # https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725
    df = df1[reduce(lambda x, y: x&y, 
        [df1[col] == val for (col, val) in keep_fixed.items()])]
    feature_col_1_values = df[feature_col_1].unique().tolist()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # ax.plot3D(df[feature_col_1], df[feature_col_2], df[metric_col])
    ax.plot_wireframe(df[feature_col_1], df[feature_col_2], df[metric_col])
    ax.set_xlabel(feature_col_1)
    ax.set_ylabel(feature_col_2)
    ax.set_zlabel(metric_col)
    ax.set_title(f'({feature_col_1}, {feature_col_2}) vs {metric_col} ')
    plt.show()


def compare_three_features():
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(x_line, y_line, z_line, 'gray')

