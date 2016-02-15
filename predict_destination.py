
import pandas as pd
import settings as s

def separate_by_source_stations(df):
    '''

given df, determine the sources, and filter for each,

from predict_destination import separate_by_source_stations
df = load_data('foo.csv', num_rows=2000)

separate_by_source_stations(df)

In [40]: df[:20][df[:20][s.START_STATION_ID] == 3107].shape
Out[40]: (2, 15)


    '''
    import ipdb; ipdb.set_trace()

    ###############################################################
    frames_by_source = {}
    # Group by source station name
    grpby = df.groupby([s.START_STATION_NAME,])

    source_station_names = grpby.groups.keys()
    for start_station in source_station_names:
        # Get DataFrame for each group.
        frames_by_source[start_station] = grpby.get_group(start_station)




    ###############################################################
    # this group by ends up giving me end station counts...
    #       .......but not unique of course.... so need to do that 
    df.groupby([
        s.START_STATION_NAME, 
    ]).count()[s.END_STATION_NAME]
    '''
In [80]: df.groupby([
        s.START_STATION_NAME, 
    ]).count()[s.END_STATION_NAME]
Out[80]: 
start station name
1 Ave & E 15 St                  5895
1 Ave & E 18 St                  3777
1 Ave & E 30 St                  4350
1 Ave & E 44 St                  2151
1 Ave & E 62 St                  3228
1 Ave & E 68 St                  5565
1 Ave & E 78 St                  3521
10 Ave & W 28 St                 4427
11 Ave & W 27 St                 4802
11 Ave & W 41 St                 3648
11 Ave & W 59 St                 2811
12 Ave & W 40 St                 5405
2 Ave & E 31 St                  3812
2 Ave & E 58 St                  2466
21 St & 41 Ave                     10
21 St & 43 Ave                    203
21 St & Queens Plaza North          3
3 Ave & E 62 St                  2543
3 Ave & Schermerhorn St           484
31 St & Thomson Ave               211
44 Dr & Jackson Ave               723
45 Rd & 11 St                     584
46 Ave & 5 St                    1169
47 Ave & 31 St                    187
48 Ave & 5 St                     321
5 Ave & E 29 St                  3989
5 Ave & E 63 St                  2261
5 Ave & E 73 St                  3775
5 Ave & E 78 St                  3359
6 Ave & Broome St                3323
                                 ... 
W 59 St & 10 Ave                 2161
W 63 St & Broadway               3400
W 64 St & West End Ave           1917
W 67 St & Broadway               3127
W 70 St & Amsterdam Ave          2946
W 74 St & Columbus Ave           1740
W 76 St & Columbus Ave           1282
W 78 St & Broadway               1238
W 82 St & Central Park West      1691
W 84 St & Broadway               3433
W 84 St & Columbus Ave           2923
W Broadway & Spring St           5596
Warren St & Church St            2002
Washington Ave & Greene Ave       536
Washington Ave & Park Ave         995
Washington Park                   697
Washington Pl & 6 Ave            2738
Washington Pl & Broadway         5400
Washington St & Gansevoort St    5544
Water - Whitehall Plaza          1545
Watts St & Greenwich St          2917
West St & Chambers St            7922
West Thames St                   3585
William St & Pine St             2052
Willoughby Ave & Hall St          793
Willoughby Ave & Tompkins Ave     365
Willoughby Ave & Walworth St      441
Willoughby St & Fleet St         1024
Wythe Ave & Metropolitan Ave     1974
York St & Jay St                 1993
Name: end station name, dtype: int64


    '''

    ###############################################################
    # Also getting interesting potential meta engineered features.
    gpby = df.groupby([
        s.START_STATION_NAME, 
    ])
    some_stats = gpby.describe()[['birth year', 'tripduration']]



    ###############################################################
    # Also count up the unique destination (end) stations for each
    grpby = df.groupby([s.START_STATION_NAME,])
    grouped_end_station = grpby[s.END_STATION_NAME]

    # This produces the sorted num of times that for each start station, a trip was taken,
    #   to the corresponding end station. 
    #
    #   => This is probably a more sophisticated intermediary measure,
    #   for 
    grouped_end_station.value_counts().to_csv('results/start_end_station_counts.021516.csv')
    '''
    In [151]: grouped_end_station.value_counts()[:10]
Out[151]: 
start station name                          
1 Ave & E 15 St     E 23 St & 1 Ave             242
                    E 15 St & 3 Ave             240
                    E 25 St & 1 Ave             192
                    1 Ave & E 30 St             166
                    E 20 St & FDR Drive         141
                    E 27 St & 1 Ave             130
                    E 33 St & 2 Ave             122
                    E 17 St & Broadway          111
                    Washington Pl & Broadway    108
                    Broadway & E 14 St           97
dtype: int64

    '''




    ###############################################################
    # Maybe the situation is better phrased as, for what source stations, 
    #   do we have enough sparseness that predicting the end station is meaningful?
    #   - And what are the predictive features for doing this prediction?


def neighborhood_grouping(df):

    # look at the min/max long lat, to determine the box edges.
    '''
    In [158]: df.describe()[['end station latitude', 'end station longitude']]
Out[158]: 
       end station latitude  end station longitude
count        1212277.000000         1212277.000000
mean              40.737799             -73.987498
std                0.021743               0.015490
min               40.646768             -74.046305
25%               40.722174             -73.998393
50%               40.739126             -73.989151
75%               40.752996             -73.978059
max               40.787209             -73.929891


    => min/max....
    
       end station latitude  end station longitude
min               40.646768             -74.046305
max               40.787209             -73.929891

    So given a choice of n neighborhoods, we can split this up among the data.

    Maybe make use of the groupby with a function feature.

    '''
    pass





