
def convert_to_num(s):
    str = s.split('.')
    final = ''
    player1 = True
    for c in str[0]:
        if c == ';' or c == '/':
            if player1:
                player1 = False
            else:
                player1 = True
        elif c == 'S' or c == 'A':
            if player1:
                final += '1'
            else:
                final += '0'
        elif c == 'R' or c == 'D':
            if player1:
                final += '0'
            else:
                final += '1'

    final = final[0:20]
    f, s = points_won(final)
    return final, float(f), float(s)

def points_won(str):
    first, second = 0, 0
    for c in str:
        if c == '1':
            first += 1
        elif c == '0':
            second += 1
    return first, second

def get_score(str):
    f = float(str[0])
    s = float(str[2])
    return f, s


def f(x1, x2):
    return x1 - x2


#
# import pandas as pd
# df = pd.read_csv('pbp_matches_atp_main_archive.csv')
# data = df[['server1', 'server2', 'winner', 'pbp', 'score']]
# data['pbp'], data['first_win_points'], data['second_win_points'] = zip(*data['pbp'].map(convert_to_num))
# data['first_win_games'], data['second_win_games'] = zip(*data['score'].map(get_score))
# data['target'] = data['winner'].map(lambda x: 0 if x == 2 else 1)
# data = data.dropna()
# print(data.head())
# exit(1)
