"""
Input data analysis
This was created after official competition time.
"""
import pandas as pd
import matplotlib.pyplot as plt


def main():
    for f in ['a_example', 'b_should_be_easy', 'c_no_hurry', 'd_metropolis', 'e_high_bonus']:
        analyse_file(f + '.in')


def analyse_file(f):
    df = pd.read_csv(f, delim_whitespace=True)
    map_info = dict(zip(['rows', 'columns', 'vehicles', 'rides', 'bonus', 'steps'], df.columns))
    df.columns = ['initial_x', 'initial_y', 'final_x', 'final_y', 'earliest_start', 'latest_finish']

    df['distance'] = abs(df['final_x'] - df['initial_x'] + df['final_y'] - df['initial_y'])

    print map_info
    print df.sort_values('earliest_start').head()
    df.hist()
    plt.show()


if __name__ == '__main__':
    main()
