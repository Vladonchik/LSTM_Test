import IPython
import numpy as np
import random
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Embedding
from keras.optimizers import Nadam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib import learn, layers
import tensorflow as tf

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import pydot

from keras.layers import *
from keras.models import *
import keras
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

from sklearn import metrics, preprocessing
from tensorflow.python.estimator.inputs.pandas_io import pandas_input_fn

random.seed(25)

file_name = 'pbp_matches_atp_main_archive'
dataset = '{}.csv'.format(file_name)
df = pd.read_csv(dataset)
df.insert(1, 'target', 1)
df.insert(1, 'score_num', 1)

# pbp_id	date	tny_name	tour	draw	server1	server2	winner	pbp	score	adf_flag	wh_minutes
# data = df[['server1', 'server2', 'winner', 'target', 'pbp', 'pbp_num']].head(10)
data = df[['server1', 'server2', 'winner', 'score', 'score_num', 'target']]

# model = load_model('models/m_by_set.h5')
# pred = model.predict([[73], [553], [0.6031746], [0.43076923]])
# print('out: {}'.format(pred))
# print(model.summary())
# exit(1)

def points_won(str):
    first, second = 0, 0
    for c in str:
        if c == '1':
            first += 1
        elif c == '0':
            second += 1

    return first, second

def get_score(str):

    f = int(str[0])
    s = int(str[2])

    return f, s

for idx, row in data.iterrows():
    str = data.loc[idx, 'score'].split(' ')
    f, s = get_score(str[0])

    # first, second = points_won(str)
    data.loc[idx, 'first_win_games'] = f
    data.loc[idx, 'second_win_games'] = s
    if data.loc[idx, 'winner'] == 2:
        data.loc[idx, 'target'] = 0

print(data.head(10))
exit(1)

first = data['first_win_games'].values.reshape(-1,1)
second = data['second_win_games'].values.reshape(-1,1)


# stack = np.column_stack((first, second))
# print(stack)
# exit(1)


scaler_first = preprocessing.MinMaxScaler()
x_scaled_first = scaler_first.fit_transform(first)
scaler_second = preprocessing.MinMaxScaler()
x_scaled_second = scaler_second.fit_transform(second)
# stack = np.column_stack((x_scaled_first, x_scaled_second))

# print(stack)
# exit(1)

# g = scaler_first.transform([[38]])
# print('scaled first: {}'.format(g))
#
# g = scaler_second.transform([[28]])
# print('scaled second: {}'.format(g))
# exit(1)

# print(data.head(10))
# exit(1)
# m = load_model('m1')
# print(m)
# exit(1)


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/{}.h5".format(name))
    print("Saved model to disk")


def load_model(name):
    # load json and create model
    json_file = open('models/{}.json'.format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/{}.h5".format(name))
    print("Loaded model from disk")
    return loaded_model


# shuflle data...
# for index, row in data.iterrows():
#     win = row['winner_slug']
#     lose = row['loser_slug']
#     ran = random.randint(0, 1)
#     if ran > 0:
#         data.at[index, 'winner_slug'] = lose
#         data.at[index, 'loser_slug'] = win
#         data.at[index, 'target'] = 0

X = data[['server1', 'server2', 'score_num']]
y = data['target']

p1 = X['server1'].unique()
p1_map = {i: val for i, val in enumerate(p1)}
inverse_p1_map = {val: i for i, val in enumerate(p1)}

p2 = X['server2'].unique()
p2_map = {i: val for i, val in enumerate(p2)}
inverse_p2_map = {val: i for i, val in enumerate(p2)}

# p3 = X['pbp_num'].unique()
# p3_map = {i: val for i, val in enumerate(p3)}
# inverse_p4_map = {val: i for i, val in enumerate(p3)}

X['server1'] = X['server1'].map(inverse_p1_map)
X['server2'] = X['server2'].map(inverse_p2_map)
# X['pbp_num'] = X['pbp_num'].map(inverse_p4_map)

# print(X['tourney_slug'])
# exit(1)

n1 = inverse_p1_map['Jaimee Fourlis']
n2 = inverse_p2_map['Maddison Inglis']
print(n1, n2)
exit(1)
print(X.head(5))

# print(X.shape)
# exit(1)

p1_input = Input(shape=(1,))
p2_input = Input(shape=(1,))
p3_input = Input(shape=(1,))
p4_input = Input(shape=(1,))

embedding_size = 30

p1_embedding = Embedding(output_dim=embedding_size, input_dim=p1.shape[0], input_length=1, name='p1_embedding')(
    p1_input)
p1_embedding = BatchNormalization()(p1_embedding)
p1_embedding = LeakyReLU()(p1_embedding)
p1_embedding = Dropout(0.4)(p1_embedding)

p2_embedding = Embedding(output_dim=embedding_size, input_dim=p2.shape[0], input_length=1, name='p2_embedding')(
    p2_input)
p2_embedding = BatchNormalization()(p2_embedding)
p2_embedding = LeakyReLU()(p2_embedding)
p2_embedding = Dropout(0.35)(p2_embedding)
# p3_embedding = Embedding(output_dim=embedding_size, input_dim=p3.shape[0], input_length=1, name='p3_embedding')(
#     p3_input)

# print('haha: {}'.format(x_scaled_first.shape))
p3 = Dense(256, input_dim=x_scaled_first.shape[0])(p3_input)
p3 = BatchNormalization()(p3)
p3 = LeakyReLU()(p3)
p3 = Dropout(0.5)(p3)

p4 = Dense(256, input_dim=x_scaled_second.shape[0])(p4_input)
p4 = BatchNormalization()(p4)
p4 = LeakyReLU()(p4)
p4 = Dropout(0.4)(p4)

player1 = Reshape([embedding_size])(p1_embedding)
player2 = Reshape([embedding_size])(p2_embedding)

# first_set = Reshape([embedding_size])(p3_embedding)

# Add dense towers or not.
# user_vecs = Dense(64, activation='relu')(user_vecs)
# item_vecs = Dense(64, activation='relu')(item_vecs)

# input_vecs = Concatenate()([player1, player2, first_set])
input_vecs = Concatenate()([player1, player2, p3, p4])
# input_vecs = Dropout(0.2)(input_vecs)

input_vecs = Dense(units=int(512 / 2), bias_initializer='ones')(input_vecs)
input_vecs = BatchNormalization()(input_vecs)
input_vecs = LeakyReLU()(input_vecs)
input_vecs = Dropout(0.45)(input_vecs)

input_vecs = Dense(units=int(256 / 2), bias_initializer='ones')(input_vecs)
input_vecs = BatchNormalization()(input_vecs)
input_vecs = LeakyReLU()(input_vecs)
input_vecs = Dropout(0.45)(input_vecs)

input_vecs = Dense(units=int(128 / 2), bias_initializer='ones')(input_vecs)
input_vecs = BatchNormalization()(input_vecs)
input_vecs = LeakyReLU()(input_vecs)
input_vecs = Dropout(0.4)(input_vecs)

# input_vecs = Convolution1D(nb_filter=32, filter_length=4, bias_initializer='ones', padding='same')(input_vecs)
# input_vecs = BatchNormalization()(input_vecs)

# input_vecs = Dense(128, activation='relu')(input_vecs)

yy = Dense(1)(input_vecs)

# model = Model(inputs=[p1_input, p2_input, p3_input], outputs=yy)
model = Model(inputs=[p1_input, p2_input, p3_input, p4_input], outputs=yy)

print(model.summary())

opt = Nadam(lr=0.0004, clipnorm=1.0, clipvalue=0.5)
# opt = 'adam'
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

# split data...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(y)

the_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=4, min_lr=0.00001, verbose=1)
the_early_stopping = EarlyStopping(patience=20)

# two_d = [X["server1"], X["server2"], X['pbp_num']]
# train_inputs = [X["server1"], X["server2"], stack]

the_train_ratio = 0.9

test_x_1 = X["server1"][int(len(X["server1"]) * the_train_ratio):]
train_x_1 = X["server1"][:int(len(X["server1"]) * the_train_ratio)]

test_x_2 = X["server2"][int(len(X["server2"]) * the_train_ratio):]
train_x_2 = X["server2"][:int(len(X["server2"]) * the_train_ratio)]

test_x_3 = x_scaled_first[int(len(x_scaled_first) * the_train_ratio):]
train_x_3 = x_scaled_first[:int(len(x_scaled_first) * the_train_ratio)]

test_x_4 = x_scaled_second[int(len(x_scaled_second) * the_train_ratio):]
train_x_4 = x_scaled_second[:int(len(x_scaled_second) * the_train_ratio)]

# print(train_inputs)

test_inputs = [test_x_1, test_x_2, test_x_3, test_x_4]
train_inputs = [train_x_1, train_x_2, train_x_3, train_x_4]

test_y = y[int(len(y) * the_train_ratio):]
y = y[:int(len(y) * the_train_ratio)]

the_model_name = 'models/' + 'm_by_set' + '.h5'
the_mcp_save = ModelCheckpoint(the_model_name, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_inputs,
                    y,
                    batch_size=64,
                    epochs=1000,
                    validation_split=0.1,
                    shuffle=False,
                    callbacks=[the_reduce_lr, the_early_stopping, the_mcp_save],
                    verbose=2)


the_score_best = model.evaluate(test_inputs, test_y, verbose=0)
print('model best loss:', the_score_best[0], 'acc:', the_score_best[1])
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

