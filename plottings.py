
from ggplot import (aes, geom_point, scale_x_continuous, ggplot,
                    ggtitle, stat_smooth)

import pandas as pd
import settings as s
import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def _make_title(title, num_rows):
    return '{} (n={})'.format(title, num_rows)

def plot_age_speed(df):
    num_rows = df.shape[0]
    title = 'age v speed'

    print ggplot(df, aes(s.AGE_COL_NAME, s.SPEED_COL_NAME)) + \
            ggtitle(_make_title(title, num_rows))+ \
            geom_point(colour='steelblue') + \
            scale_x_continuous(
                    # breaks=[10,20,30],
                    # labels=["horrible", "ok", "awesome"]
                    )

    return df 


def plot_age_speed_histogram(df):
    '''
    overlay speed histograms over several buckets,

    '''
    age_buckets = [
            [15, 19],
            [20, 24],
            [25, 29],
            [30, 34],
            [35, 39],
            [40, 49],
            [50, 59],
            [60, 69],
            [70, 500],
            ]

    # s.AGE_COL_NAME, s.SPEED_COL_NAME

    df2 = df.copy(deep=True)
    # df2 = pd.DataFrame(index=df.age)

    new_cols = []

    # TODO... round speeds to nearest 0.1 mph

    for left, right in age_buckets:
        new_col_name = 'speeds_%sto%s' % (left, right)
        # df2['speeds_15to19'] = df2[df2.age <= 19]['speed']
        df2[new_col_name] = df2[(df2.age >= left) & (df2.age <= right)]['speed'].round(decimals=1)

        new_cols.append(new_col_name)
    
    import ipdb; ipdb.set_trace()
    plt.figure()
    # df.speed.hist(color='k', alpha=0.5, bins=20, figsize=(6, 4))

    # df2.hist(column=new_cols, alpha=0.5)

    df3 = df2[new_cols]

    df3.plot(kind='hist', alpha=0.5, bins=100)


    now = datetime.datetime.now()
    filename = '%s/p.%s.jpg' % (s.PLOTS_DIR, now.strftime('%m-%d-%YT%H%M%S'))
    plt.savefig(filename)

def plot_age_histogram(df):

    plt.figure()
    df[['age']].hist(color='k', alpha=0.5, bins=20, figsize=(6, 4))

    now = datetime.datetime.now()
    filename = '%s/p.%s.jpg' % (s.PLOTS_DIR, now.strftime('%m-%d-%YT%H%M%S'))
    plt.savefig(filename)
    # df.diff().hist(color='k', alpha=0.5, bins=50, figsize=(6, 4))

    pass


def plot_age_speed_histogram_w_box_plots(df):
    pass

def plot_distance_trip_time(df):
    num_rows = df.shape[0]
    title = 'trip duration v distance travelled'

    print ggplot(df, aes(s.TRIP_DURATION_COL, s.DISTANCE_TRAVELED_COL_NAME)) + \
            ggtitle(_make_title(title, num_rows))+ \
            stat_smooth(colour="red") + \
            geom_point(colour='steelblue') + \
            scale_x_continuous(
                    # breaks=[10,20,30], 
                    #labels=["horrible", "ok", "awesome"]
                    )

    return df 



if __name__ == '__main__':
    df = pd.read_csv('speeds_small.csv')
    import ipdb; ipdb.set_trace()

    plot_age_speed_histogram(df)
    plot_distance_trip_time(df)
    plot_age_speed(df)

    pass

    df = pd.read_csv('foo.csv')

    plot_age_speed(df)

    plot_distance_trip_time(df)

