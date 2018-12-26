import random
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Nadam
from keras.layers import *
from keras.models import *
import keras
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

pd.options.mode.chained_assignment = None  # default='warn'

from keras_layer_normalization import LayerNormalization
from sklearn import metrics, preprocessing
import helper_funcs as hp

random.seed(25)

file_list = [
             'pbp_matches_atp_main_archive',
             'pbp_matches_itf_main_archive',
             'pbp_matches_atp_main_current',
             'pbp_matches_itf_main_current'
             ]

frames = []

for i in file_list:
    csv = i + '.csv'
    df = pd.read_csv(csv)
    # print(df)
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# print(df)
# exit(1)

# EXECUTE_MODEL = False
# TRAIN = True

EXECUTE_MODEL = True
TRAIN = False

# 1 pair 1:00 am
# player1_details = ('Olivier Rochus', 10.0)
# player2_details = ('Fabio Fognini', 0.0)

player1_details = ('Alex Bolt', 6.0)
player2_details = ('Luke Saville', 4.0)

# player1_details = ('Luke Saville', 6.0)
# player2_details = ('Alex Bolt', 4.0)

# exit(1)
data = df[['server1', 'server2', 'winner', 'pbp', 'score']]

data['pbp'], data['first_win_points'], data['second_win_points'] = zip(*data['pbp'].map(hp.convert_to_num))
data['first_win_games'], data['second_win_games'] = zip(*data['score'].map(hp.get_score))
data['target'] = data['winner'].map(lambda x: 0 if x == 2 else 1)
data = data.dropna()

s1 = np.array(data['server1'])
s2 = np.array(data['server2'])
all_players = np.append(s1, s2, axis=0)

player1_exists = False
player2_exists = False
if player1_details[0] in all_players:
    player1_exists = True
    print('p1 exists')
if player2_details[0] in all_players:
    player2_exists = True
    print('p2 exists')
if not player1_exists or not player2_exists:
    print('players dont exist')
    exit(1)


# for idx, row in data.iterrows():
#     str_score = data.loc[idx, 'score'].split(' ')
#     f_games, s_games = hp.get_score(str_score[0])
#
#     # str_pbp = data.loc[idx, 'pbp'].split('.')
#     # pbp, f_points, s_points = hp.convert_to_num(str_pbp[0])
#     #
#     # data.loc[idx, 'first_win_points'] = f_points
#     # data.loc[idx, 'second_win_games'] = s_points
#     data.loc[idx, 'first_win_games'] = f_games
#     data.loc[idx, 'second_win_games'] = s_games
#     if data.loc[idx, 'winner'] == 2:
#         data.loc[idx, 'target'] = 0


# new_data = data[['server1', 'server2', 'winner', 'pbp', 'score', 'first_win_points', 'second_win_points', 'first_win_games', 'second_win_games']]
# new_data.rename(columns = {'server1': 'server2',
#                                       'server2' : 'server1',
#                                       'first_win_points': 'second_win_points',
#                                       'second_win_points' : 'first_win_points',
#                                       'first_win_games' : 'second_win_games',
#                                       'second_win_games' : 'first_win_games'
#                                       }, inplace=True)
# new_data['target'] = new_data['winner'].map(lambda x: 1 if x == 2 else 0)


# data = data.drop(columns=['pbp', 'score'])
# new_data = new_data.drop(columns=['pbp', 'score'])

# print(len(data))
# print(data.tail(2))

# data = pd.concat([data, new_data], ignore_index=True)

# print(data)
# print(data.tail(2))
# print(len(data))
# exit(1)

# first = data['first_win_points'].values.reshape(-1, 1)
# second = data['second_win_points'].values.reshape(-1, 1)
# first_set = data['first_win_games'].values.reshape(-1, 1)
# second_set = data['second_win_games'].values.reshape(-1, 1)

p3 = np.unique(all_players)
p3_map = {i: val for i, val in enumerate(p3)}
inverse_p3_map = {val: i for i, val in enumerate(p3)}

data['server1'] = data['server1'].map(inverse_p3_map)
data['server2'] = data['server2'].map(inverse_p3_map)

n1 = inverse_p3_map[player1_details[0]]
n2 = inverse_p3_map[player2_details[0]]
print('p1_name: {}, p2_name: {}'.format(n1, n2))
print('p1_name: {}, p2_name: {}'.format(p3_map[n1], p3_map[n2]))

scalars = {
            'scalar_server_1' : preprocessing.MinMaxScaler(),
            'scalar_server_2': preprocessing.MinMaxScaler(),
            'scalar_1_win_points' : preprocessing.MinMaxScaler(),
            'scalar_2_win_points' : preprocessing.MinMaxScaler(),
            'scalar_1_win_games': preprocessing.MinMaxScaler(),
            'scalar_2_win_games': preprocessing.MinMaxScaler(),
            }

# data['scalar_server_1'] = scalars['scalar_server_1'].fit_transform(data['server1'].values.reshape(-1,1))
# data['scalar_server_2'] = scalars['scalar_server_2'].fit_transform(data['server2'].values.reshape(-1,1))

data['scalar_1_win_points'] = scalars['scalar_1_win_points'].fit_transform(data['first_win_points'].values.reshape(-1,1))
data['scalar_2_win_points'] = scalars['scalar_2_win_points'].fit_transform(data['second_win_points'].values.reshape(-1,1))

data['scalar_1_win_games'] = scalars['scalar_1_win_games'].fit_transform(data['first_win_games'].values.reshape(-1,1))
data['scalar_2_win_games'] = scalars['scalar_2_win_games'].fit_transform(data['second_win_games'].values.reshape(-1,1))

# print(data.head(10))
# exit(1)

# FIRST_SCALED_SERVER = scalars['scalar_server_1'].transform([[n1]])
# print('scaled first server: {}'.format(FIRST_SCALED_SERVER))
#
# SECOND_SCALED_SERVER = scalars['scalar_server_2'].transform([[n2]])
# print('scaled second server: {}'.format(SECOND_SCALED_SERVER))

FIRST_SCALED_POINTS = scalars['scalar_1_win_points'].transform([[player1_details[1]]])
print('scaled points first: {}'.format(FIRST_SCALED_POINTS))

SECOND_SCALED_POINTS = scalars['scalar_2_win_points'].transform([[player2_details[1]]])
print('scaled points second: {}'.format(SECOND_SCALED_POINTS))

# FIRST_SCALED_GAMES = scalars['scalar_1_win_games'].transform([[player1_details[2]]])
# print('scaled first_set: {}'.format(FIRST_SCALED_GAMES))
#
# SECOND_SCALED_GAMES = scalars['scalar_2_win_games'].transform([[player2_details[2]]])
# print('scaled second_set: {}'.format(SECOND_SCALED_GAMES))


X = data
y = data['target']

# print(X.head(10))
# exit(1)

# print(p3_map[n1])
# print(p3_map[n2])
# exit(1)


if EXECUTE_MODEL:
    model = keras.models.load_model('models/m4.dhf5',
                                    custom_objects={'LayerNormalization': LayerNormalization})

    arr = []
    for g in range(10):
        for i in range(100):
            pred = model.predict([
                                     [[n1]],
                                     [[n2]]])
                                  # [FIRST_SCALED_SERVER[0][0]],
                                  # # [SECOND_SCALED_SERVER[0][0]],
                                  # [FIRST_SCALED_POINTS[0][0]],
                                  # [SECOND_SCALED_POINTS[0][0]]])
                                  # [FIRST_SCALED_GAMES[0][0]],
                                  # [SECOND_SCALED_GAMES[0][0]]])

            arr.append(pred)
            # print('out: {}'.format(pred))
        nparr = np.array(arr)
        mean = nparr.mean()
        std = nparr.std()
        arr = []
        print('mean: {}, std: {}, skew: {}, kurtosis: {}'.format(mean, std, skew(nparr), kurtosis(nparr)))
    # print(model.summary())
    exit(1)

if TRAIN:
    p1_input = Input(shape=(1,))
    p2_input = Input(shape=(1,))
    p3_input = Input(shape=(1,))
    p4_input = Input(shape=(1,))
    p5_input = Input(shape=(1,))
    p6_input = Input(shape=(1,))

    embedding_size = 30

    p1_embedding = Embedding(output_dim=embedding_size, input_dim=all_players.shape[0], input_length=1)(p1_input)
    p1_embedding = Flatten()(p1_embedding)
    p1_embedding = LayerNormalization()(p1_embedding)
    p1_embedding = LeakyReLU()(p1_embedding)
    p1_embedding = Dropout(0.4)(p1_embedding, training=True)

    p2_embedding = Embedding(output_dim=embedding_size, input_dim=all_players.shape[0], input_length=1)(p2_input)
    p2_embedding = Flatten()(p2_embedding)
    p2_embedding = LayerNormalization()(p2_embedding)
    p2_embedding = LeakyReLU()(p2_embedding)
    p2_embedding = Dropout(0.35)(p2_embedding, training=True)


    # p1 = Dense(256, input_dim=X['scalar_server_1'].shape[0])(p1_input)
    # p1 = LayerNormalization()(p1)
    # p1 = LeakyReLU()(p1)
    # p1 = Dropout(0.5)(p1, training=True)
    #
    # p2 = Dense(256, input_dim=X['scalar_server_2'].shape[0])(p2_input)
    # p2 = LayerNormalization()(p2)
    # p2 = LeakyReLU()(p2)
    # p2 = Dropout(0.5)(p2, training=True)

    p3 = Dense(112, input_dim=X['scalar_1_win_points'].shape[0])(p3_input)
    p3 = LayerNormalization()(p3)
    p3 = LeakyReLU()(p3)
    p3 = Dropout(0.5)(p3, training=True)

    p4 = Dense(112, input_dim=X['scalar_2_win_points'].shape[0])(p4_input)
    p4 = LayerNormalization()(p4)
    p4 = LeakyReLU()(p4)
    p4 = Dropout(0.4)(p4, training=True)

    p5 = Dense(112, input_dim=X['scalar_1_win_games'].shape[0])(p5_input)
    p5 = LayerNormalization()(p5)
    p5 = LeakyReLU()(p5)
    p5 = Dropout(0.5)(p5, training=True)

    p6 = Dense(112, input_dim=X['scalar_2_win_games'].shape[0])(p6_input)
    p6 = LayerNormalization()(p6)
    p6 = LeakyReLU()(p6)
    p6 = Dropout(0.5)(p6, training=True)

    # player1 = Reshape([embedding_size])(p1_embedding)
    # player2 = Reshape([embedding_size])(p2_embedding)

    # Add dense towers or not.
    # user_vecs = Dense(64, activation='relu')(player1)
    # item_vecs = Dense(64, activation='relu')(player2)

    input_vecs = Concatenate()([p1_embedding, p2_embedding])
    # input_vecs = Concatenate()([p1, p2, p3, p4, p5, p6])
    # input_vecs = Dropout(0.2)(input_vecs)

    # input_vecs = Dense(units=int(512 / 2), bias_initializer='ones')(input_vecs)
    # input_vecs = LayerNormalization()(input_vecs)
    # input_vecs = LeakyReLU()(input_vecs)
    # input_vecs = Dropout(0.5)(input_vecs, training=True)

    input_vecs = Dense(units=int(256 / 2), bias_initializer='ones')(input_vecs)
    input_vecs = LayerNormalization()(input_vecs)
    input_vecs = LeakyReLU()(input_vecs)
    input_vecs = Dropout(0.5)(input_vecs, training=True)

    input_vecs = Dense(units=int(128 / 2), bias_initializer='ones')(input_vecs)
    input_vecs = LayerNormalization()(input_vecs)
    input_vecs = LeakyReLU()(input_vecs)
    input_vecs = Dropout(0.4)(input_vecs, training=True)

    # input_vecs = Convolution1D(nb_filter=32, filter_length=4, bias_initializer='ones', padding='same')(input_vecs)
    # input_vecs = BatchNormalization()(input_vecs)

    # input_vecs = Dense(128, activation='relu')(input_vecs)

    yy = Dense(1)(input_vecs)

    # model = Model(inputs=[p1_input, p2_input, p3_input], outputs=yy)
    model = Model(inputs=[p1_input, p2_input], outputs=yy)

    print(model.summary())

    opt = Nadam(lr=0.002, clipnorm=1.0, clipvalue=0.5)
    # opt = 'adam'
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    # split data...
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    the_train_ratio = 0.9

    test_x = X[int(len(X) * the_train_ratio):]
    train_x = X[:int(len(X) * the_train_ratio)]
    # print(test_x)

    # test_inputs = [test_x['server1'], test_x['server2'], test_x['scalar_1_win_points'], test_x['scalar_2_win_points']]
    # train_inputs = [train_x['server1'], train_x['server2'], train_x['scalar_1_win_points'], train_x['scalar_2_win_points']]

    test_inputs = [test_x['server1'], test_x['server2']]
    train_inputs = [train_x['server1'], train_x['server2']]

    test_y = y[int(len(y) * the_train_ratio):]
    y = y[:int(len(y) * the_train_ratio)]

    the_model_name = 'models/' + 'm4' + '.dhf5'
    the_mcp_save = ModelCheckpoint(the_model_name, save_best_only=True, monitor='val_loss', mode='min')

    the_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=2, min_lr=0.00001, verbose=1)
    the_early_stopping = EarlyStopping(patience=25)
    train_early_stopping = EarlyStopping(patience=10, monitor='loss')

    history = model.fit(train_inputs,
                        y,
                        batch_size=64,
                        epochs=100,
                        validation_split=0.2,
                        shuffle=False,
                        callbacks=[the_reduce_lr, the_early_stopping, train_early_stopping, the_mcp_save],
                        verbose=2)

    the_score_best = model.evaluate(test_inputs, test_y, verbose=0)
    print('model best test loss:', the_score_best[0], 'acc:', the_score_best[1])

    # save_model(model, 'm1')

    # pred = model.predict([[1432], [1683], [1], [1]])
    # print('out: {}'.format(pred))

    print(history.history.keys())
    print(history.history)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
