import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
# from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model

import sklearn
from sklearn.model_selection import train_test_split

"""STEP PROCEDURE
1. DATA COLLECTION = DATA SEKUNDER DARI DATA KPU
2. PREPROCESSING DATA
3. TRAINING
4. TESTING
5. DEPLOY TO GUI
"""

data_url = "https://drive.google.com/file/d/1Mr1niApw73MvOrXq3SA8hqoxt90jf0lV/view?usp=sharing"
data_url = 'https://drive.google.com/uc?id=' + data_url.split('/')[-2]
names_df = pd.read_csv(data_url)
names_df.columns = ["Name", "Gender"]

names_df.info()
names_df.head(5)

def preprocess(names_df, train=True):
    # Step 1: Lowercase the names
    names_df['Name'] = names_df['Name'].str.lower()

    # Step 2: Split The names to individual characters
    names_df['Name'] = [list(name) for name in names_df['Name']]

    # Step 3: Pad names with spaces to make all names same length
    name_length = 50
    names_df['Name'] = [
        (name + [' ']*name_length)[:name_length] 
        for name in names_df['Name']
    ]

    # Step 4: Encode Names to Array of Numbers
    names_df['Name'] = [
        [
            max(0.0, ord(char)-96.0) 
            for char in name
        ]
        for name in names_df['Name']
    ]
    
    if train:
        # Step 5: Encode Gender to Numbers
        names_df['Gender'] = [
            0.0 if gender=='Perempuan' else 1.0 
            for gender in names_df['Gender']
        ]
    
    return names_df

# names_df = preprocess(names_df)
# names_df.head()

def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
    model = Sequential([
        Embedding(num_alphabets, embedding_dim, input_length=name_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model

# # Step 1: Instantiate the model
# model = lstm_model(num_alphabets=27, name_length=50, embedding_dim=256)

# # Step 2: Split Training and Test Data
# X = np.asarray(names_df['Name'].values.tolist())
# y = np.asarray(names_df['Gender'].values.tolist())

# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,
#                                                     random_state=0)

# # Step 3: Train the model
# callbacks = [
#     EarlyStopping(monitor='val_accuracy',
#                   min_delta=1e-3,
#                   patience=5,
#                   mode='max',
#                   restore_best_weights=True,
#                   verbose=1),
# ]

# history = model.fit(x=X_train,
#                     y=y_train,
#                     batch_size=64,
#                     epochs=5,
#                     validation_data=(X_test, y_test),
#                     callbacks=callbacks)

# # Step 4: Save the model
# model.save('gender_prediction.h5')

# # Step 5: Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print('Accuracy:', accuracy)

# # Step 6: Plot accuracies
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='val')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

"""# Testing: Predictions"""
def predict(my_list):
    pred_model = load_model('gender_prediction.h5')

    # Input names
    # names = ['husni']

    # Convert to dataframe
    pred_df = pd.DataFrame({'Name': my_list})

    print(pred_df)

    # Preprocess
    pred_df = preprocess(pred_df, train=False)

    # Predictions
    result = pred_model.predict(np.asarray(
        pred_df['Name'].values.tolist())).squeeze(axis=1)

    pred_df['Pria atau Wanita?'] = [
      'Pria' if logit > 0.5 else 'Wanita' for logit in result
    ]

    pred_df['Probability'] = [
        logit if logit > 0.5 else 1.0 - logit for logit in result
        ]

    # Format the output
    pred_df['Name'] = my_list
    #pred_df.rename(columns={'Name': 'Name'}, inplace=True)
    pred_df['Probability'] = pred_df['Probability'].round(2)
    pred_df.drop_duplicates(inplace=True)

    print(pred_df)

    return pred_df