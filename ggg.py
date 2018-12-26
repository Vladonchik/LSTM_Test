import IPython
import numpy as np
import random
import pandas as pd
from keras.callbacks import ModelCheckpoint
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

from sklearn import metrics
from tensorflow.python.estimator.inputs.pandas_io import pandas_input_fn

random.seed(25)

file_name = 'match_scores_1991-2016_unindexed'
dataset = '{}.csv'.format(file_name)
df = pd.read_csv(dataset)
df.insert(1, 'target', 1)

# data = df[['winner_slug', 'loser_slug', 'round_order','target']].head(200)
data = df[['winner_slug', 'loser_slug', 'round_order', 'tourney_slug', 'target']]

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
for index, row in data.iterrows():
    win = row['winner_slug']
    lose = row['loser_slug']
    ran = random.randint(0, 1)
    if ran > 0:
        data.at[index, 'winner_slug'] = lose
        data.at[index, 'loser_slug'] = win
        data.at[index, 'target'] = 0

data = data.rename(columns={'winner_slug': 'player1', 'loser_slug': 'player2'})
X = data[['player1', 'player2', 'tourney_slug', 'round_order']]
y = data['target']


p1 = X['player1'].unique()
p1_map = {i:val for i,val in enumerate(p1)}
inverse_p1_map = {val:i for i,val in enumerate(p1)}

p2 = X['player2'].unique()
p2_map = {i:val for i,val in enumerate(p2)}
inverse_p2_map = {val:i for i,val in enumerate(p2)}

p4 = X['tourney_slug'].unique()
p4_map = {i:val for i,val in enumerate(p4)}
inverse_p4_map = {val:i for i,val in enumerate(p4)}

X['player1'] = X['player1'].map(inverse_p1_map)
X['player2'] = X['player2'].map(inverse_p2_map)
X['tourney_slug'] = X['tourney_slug'].map(inverse_p4_map)

# print(X['tourney_slug'])
# exit(1)

# n1 = inverse_p1_map['dominic-pagon']
# n2 = inverse_p2_map['kei-nishikori']
# print(n1, n2)
# exit(1)
# print(X.head(5))

# print(X.shape)
# exit(1)

p1_input = Input(shape=(1,), name='p1')
p2_input = Input(shape=(1,))
p4_input = Input(shape=(1,))
p3_input = Input(shape=(1,))

embedding_size = 30

p1_embedding = Embedding(output_dim=embedding_size, input_dim=p1.shape[0], input_length=1, name='p1_embedding')(p1_input)
p2_embedding = Embedding(output_dim=embedding_size, input_dim=p2.shape[0], input_length=1, name='p2_embedding')(p2_input)
p4_embedding = Embedding(output_dim=embedding_size, input_dim=p4.shape[0], input_length=1, name='p4_embedding')(p4_input)

p3 = Dense(128, input_dim=p3_input.shape[1], name='p3_')(p3_input)

user_vecs = Reshape([embedding_size])(p1_embedding)
item_vecs = Reshape([embedding_size])(p2_embedding)
tourney_slug = Reshape([embedding_size])(p4_embedding)

# round = Reshape([embedding_size])(p3)

# Add dense towers or not.
# user_vecs = Dense(64, activation='relu')(user_vecs)
# item_vecs = Dense(64, activation='relu')(item_vecs)

input_vecs = Concatenate()([user_vecs, item_vecs, tourney_slug, p3])

input_vecs = Dropout(0.2)(input_vecs)

input_vecs = Dense(128, activation='relu')(input_vecs)

yy = Dense(1)(input_vecs)

model = Model(inputs=[p1_input, p2_input, p4_input, p3_input], outputs=yy)

print(model.summary())

# opt= Nadam(lr=0.004, clipnorm=1.0, clipvalue=0.5)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# split data...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(y)

two_d = [X["player1"], X["player2"], X['tourney_slug'], X['round_order']]
history = model.fit( two_d,
                     y,
                     batch_size=64,
                     epochs=4,
                     validation_split=0.15,
                     shuffle=False)


# save_model(model, 'm1')

pred = model.predict([[1432], [1683], [1], [1]])
print('out: {}'.format(pred))

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

# two_d = np.array(X_test["player1"], X_test["player2"])
# print(two_d)
# test_loss, test_acc = model.evaluate([X_test["player1"], X_test["player2"]], y_test, batch_size=64)
# print('Test accuracy:', test_acc)


# exit(1)



# winner_classes = X_train["player1"].unique()
# n_classes = len(winner_classes) + 1
# print('winner_classes has next classes: ', winner_classes)
# Process categorical variables into ids.

# categorical_vars = ['player1', 'player2']
# continues_vars = ['tourney_order']
#
# models = []
# for categoical_var in categorical_vars :
#     model = Sequential()
#     model.reset_states()
#     no_of_unique_cat = X[categoical_var].nunique()
#     embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
#     embedding_size = int(embedding_size)
#     model.add(Embedding(no_of_unique_cat+1, embedding_size, input_length = 1))
#     model.add(Reshape(target_shape=(embedding_size,)))
#     models.append(model)
#
# model_rest = Sequential()
# model_rest.add(Dense(64, input_dim=6))
# model_rest.reset_states()
# models.append(model_rest)
#
#
# concatenated = concatenate([model.input for model in models])
# out = Dense(1, activation='softmax', name='output_layer')(concatenated)
# merged_model = Model([model.input for model in models], out)
#
#
# # history  =  merged_model.fit([], y, epochs =200 , batch_size =16, verbose= 1)
# # merged_model.layers
# print(merged_model.summary())
#
# full_model = Sequential()
# full_model.add(merged_model)
#
#
#
# full_model.add(Dense(512))
# full_model.add(Activation('sigmoid'))
# full_model.add(Dropout(0.2))
#
# full_model.add(Dense(32))
# full_model.add(Activation('sigmoid'))
# full_model.add(Dropout(0.2))
#
# full_model.add(Dense(1))
