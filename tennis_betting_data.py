import random
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Nadam
from keras.layers import *
from keras.models import *
import keras
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, stats
import helper_funcs as hf
from ReduceLRWithWarmRestart import ReduceLRWithWarmRestart

pd.options.mode.chained_assignment = None  # default='warn'

from keras_layer_normalization import LayerNormalization
from sklearn import metrics, preprocessing
import helper_funcs as hp

random.seed(25)

FINAL_FILE = True
EXECUTE_MODEL = False


player1_details = ('Federer R.', 10, 1.3)
player2_details = ('Mayer F.', 160, 2.3)

# player1_details = ('Stepanek R.', 300, 2.15)
# player2_details = ('Federer R.', 300, 2.15)

ROUND = '1st Round'


MODEL_NAME = 'm5'

# data = None
if FINAL_FILE:
    data = pd.read_csv('tennis.csv')
else:

    file_list = []
    for i in range(2008, 2018):
        s = 'tennis_betting/' + str(i)
        file_list.append(s)

        # if i > 2009:
        #     sw = 'tennis_betting/w' + str(i)
        #     file_list.append(sw)

    frames = []

    for i in file_list:
        csv = i + '.csv'
        df = pd.read_csv(csv)
        # print(df)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # print(file_list)
    # print(frames)
    # exit(1)

    # https://www.kaggle.com/jordangoblet/atp-tour-20002016
    data = df[['W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'Best of', 'Court', 'Surface', 'Round', 'Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', 'B365W', 'B365L']]
    data.insert(1, 'target', 1)

    for idx, row in data.iterrows():

        winner = data.loc[idx, 'Winner']
        loser = data.loc[idx, 'Loser']

        wRank = data.loc[idx, 'WRank']
        lRank = data.loc[idx, 'LRank']

        wpts = data.loc[idx, 'WPts']
        lpts = data.loc[idx, 'LPts']

        avgW = data.loc[idx, 'B365W']
        avgL = data.loc[idx, 'B365L']

        rand = random.randint(0, 1)

        if rand > 0:
            data.loc[idx, 'Winner'] = loser
            data.loc[idx, 'WRank'] = lRank
            data.loc[idx, 'WPts'] = lpts
            data.loc[idx, 'B365W'] = avgL

            data.loc[idx, 'Loser'] = winner
            data.loc[idx, 'LRank'] = wRank
            data.loc[idx, 'LPts'] = wpts
            data.loc[idx, 'B365L'] = avgW

            data.loc[idx, 'target'] = 0

    data = data.rename(index=str, columns={'Winner': 'p1', 'Loser': 'p2',
                                           'WRank': 'p1Rank', 'LRank': 'p2Rank',
                                           'WPts': 'p1Pts', 'LPts': 'p2Pts',
                                           'B365W': 'p1Avg', 'B365L': 'p2Avg'})

    print(data)
    data.to_csv('tennis.csv')
    exit(1)

data['W1'].fillna(0, inplace=True)
data['W2'].fillna(0, inplace=True)
data['W3'].fillna(0, inplace=True)
data['L1'].fillna(0, inplace=True)
data['L2'].fillna(0, inplace=True)
data['L3'].fillna(0, inplace=True)
data.dropna(inplace=True)

data['num_games'] = data.W1 + data.L1 + data.W2 + data.L2 + data.W3 + data.L3
data = data[data['Best of'] != 5]
data['num_games_target'] = data['num_games'].apply(lambda x: 1 if x > 20 else 0)

# print(data['num_games_target'].values.count(0))
print(data.head(10))
exit(1)

s1 = np.array(data['p1'])
s2 = np.array(data['p2'])
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

p2 = np.unique(data['Round'])
p2_map = {i: val for i, val in enumerate(p2)}
inverse_p2_map = {val: i for i, val in enumerate(p2)}

data['Round'] = data['Round'].map(inverse_p2_map)

p3 = np.unique(all_players)
p3_map = {i: val for i, val in enumerate(p3)}
inverse_p3_map = {val: i for i, val in enumerate(p3)}

data['p1'] = data['p1'].map(inverse_p3_map)
data['p2'] = data['p2'].map(inverse_p3_map)

n1 = inverse_p3_map[player1_details[0]]
n2 = inverse_p3_map[player2_details[0]]
print('p1_name: {}, p2_name: {}'.format(n1, n2))
print('p1_name: {}, p2_name: {}'.format(p3_map[n1], p3_map[n2]))

n3 = inverse_p2_map[ROUND]
print('round: {}'.format(p2_map[n3]))

scalars = {
    'p1Rank': preprocessing.MinMaxScaler(),
    'p2Rank': preprocessing.MinMaxScaler(),
    'p1Avg': preprocessing.MinMaxScaler(),
    'p2Avg': preprocessing.MinMaxScaler()
}


data = data[((data.p1Avg - data.p1Avg.mean()) / data.p1Avg.std()).abs() < 3]
data = data[((data.p2Avg - data.p2Avg.mean()) / data.p2Avg.std()).abs() < 3]

# data = data[((data.p1Rank - data.p1Rank.mean()) / data.p1Rank.std()).abs() < 3]
# data = data[((data.p2Rank - data.p2Rank.mean()) / data.p2Rank.std()).abs() < 3]

data['p1Rank'] = scalars['p1Rank'].fit_transform(data['p1Rank'].values.reshape(-1, 1))
data['p2Rank'] = scalars['p2Rank'].fit_transform(data['p2Rank'].values.reshape(-1, 1))

# data['p2RankDelta'] = scalars['p2RankDelta'].fit_transform(data['p2RankDelta'].values.reshape(-1, 1))

data['p1Avg'] = scalars['p1Avg'].fit_transform(data['p1Avg'].values.reshape(-1, 1))
data['p2Avg'] = scalars['p2Avg'].fit_transform(data['p2Avg'].values.reshape(-1, 1))


# print(data)
# exit(1)

P1_SCALED_RANK = scalars['p1Rank'].transform([[player1_details[1]]])
print('scaled first server: {}'.format(P1_SCALED_RANK))

P2_SCALED_RANK = scalars['p1Rank'].transform([[player2_details[1]]])
print('scaled first server: {}'.format(P2_SCALED_RANK))


FIRST_SCALED_AVG = scalars['p1Avg'].transform([[player1_details[2]]])
print('scaled points first: {}'.format(FIRST_SCALED_AVG))

SECOND_SCALED_AVG = scalars['p2Avg'].transform([[player2_details[2]]])
print('scaled points second: {}'.format(SECOND_SCALED_AVG))

# exit(1)

X = data
y = data['target']

if EXECUTE_MODEL:
    model = keras.models.load_model('models/' + MODEL_NAME + '.dhf5',
                                    custom_objects={'LayerNormalization': LayerNormalization})

    arr = []
    for g in range(10):
        for i in range(100):
            pred = model.predict([
                [[n1]],
                [[n2]],
                [[n3]],
                [P1_SCALED_RANK[0][0]],
                [P2_SCALED_RANK[0][0]],
                [FIRST_SCALED_AVG[0][0]],
                [SECOND_SCALED_AVG[0][0]]])
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

else:
    p1_input = Input(shape=(1,))
    p2_input = Input(shape=(1,))
    p3_input = Input(shape=(1,))
    p3_input_round = Input(shape=(1,))
    p4_input = Input(shape=(1,))
    p5_input = Input(shape=(1,))
    p6_input = Input(shape=(1,))
    p7_input = Input(shape=(1,))
    p8_input = Input(shape=(1,))

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
    p2_embedding = Dropout(0.45)(p2_embedding, training=True)

    p3_embedding_round = Embedding(output_dim=embedding_size, input_dim=all_players.shape[0], input_length=1)(p3_input_round)
    p3_embedding_round = Flatten()(p3_embedding_round)
    p3_embedding_round = LayerNormalization()(p3_embedding_round)
    p3_embedding_round = LeakyReLU()(p3_embedding_round)
    p3_embedding_round = Dropout(0.45)(p3_embedding_round, training=True)

    p3 = Dense(112, input_dim=X['p1Rank'].shape[0])(p3_input)
    p3 = LayerNormalization()(p3)
    p3 = LeakyReLU()(p3)
    p3 = Dropout(0.5)(p3, training=True)

    p4 = Dense(112, input_dim=X['p2Rank'].shape[0])(p4_input)
    p4 = LayerNormalization()(p4)
    p4 = LeakyReLU()(p4)
    p4 = Dropout(0.4)(p4, training=True)

    p7 = Dense(112, input_dim=X['p1Avg'].shape[0])(p7_input)
    p7 = LayerNormalization()(p7)
    p7 = LeakyReLU()(p7)
    p7 = Dropout(0.5)(p7, training=True)

    p8 = Dense(112, input_dim=X['p2Avg'].shape[0])(p8_input)
    p8 = LayerNormalization()(p8)
    p8 = LeakyReLU()(p8)
    p8 = Dropout(0.5)(p8, training=True)

    # player1 = Reshape([embedding_size])(p1_embedding)
    # player2 = Reshape([embedding_size])(p2_embedding)

    # Add dense towers or not.
    # user_vecs = Dense(64, activation='relu')(player1)
    # item_vecs = Dense(64, activation='relu')(player2)
    # input_vecs = Concatenate()([p1_embedding, p2_embedding, p7, p8])
    input_vecs = Concatenate()([p1_embedding, p2_embedding, p3_embedding_round, p3, p4, p7, p8])
    # input_vecs = Dropout(0.2)(input_vecs)

    input_vecs = Dense(units=int(512 / 2), bias_initializer='ones')(input_vecs)
    input_vecs = LayerNormalization()(input_vecs)
    input_vecs = LeakyReLU()(input_vecs)
    input_vecs = Dropout(0.5)(input_vecs, training=True)

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

    model = Model(inputs=[p1_input, p2_input, p3_input_round, p3_input, p4_input, p7_input, p8_input], outputs=yy)
    # model = Model(inputs=[p1_input, p2_input, p7_input, p8_input], outputs=yy)

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

    test_inputs = [test_x['p1'], test_x['p2'], test_x['Round'], test_x['p1Rank'], test_x['p2Rank'],
                   test_x['p1Avg'], test_x['p2Avg']]
    train_inputs = [train_x['p1'], train_x['p2'], train_x['Round'], train_x['p1Rank'], train_x['p2Rank'],
                    train_x['p1Avg'], train_x['p2Avg']]

    # test_inputs = [test_x['p1'], test_x['p2'], test_x['p1Avg'], test_x['p2Avg']]
    # train_inputs = [train_x['p1'], train_x['p2'], train_x['p1Avg'], train_x['p2Avg']]

    # print(test_inputs)
    # exit(1)

    test_y = y[int(len(y) * the_train_ratio):]
    y = y[:int(len(y) * the_train_ratio)]

    the_model_name = 'models/' + MODEL_NAME + '.dhf5'
    the_mcp_save = ModelCheckpoint(the_model_name, save_best_only=True, monitor='val_loss', mode='min')

    the_reduce_lr = ReduceLRWithWarmRestart(arg_t_multiplier=1.5, arg_max_epoch_count=32)
    # the_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=2, min_lr=0.00001, verbose=1)
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
