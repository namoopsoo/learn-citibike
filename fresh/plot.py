import matplotlib.pyplot as plt

def compare_tuning(df, feature_col_1, feature_col_2, metric_col):

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

