"""Model builders: a baseline and a neural network.

Always build the baseline first. It's your sanity check — if the neural net
can't beat plain logistic regression on this dataset, you have a bug, not a
breakthrough.
"""

from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall


def build_baseline(config):
    """Logistic regression baseline.

    Returns a scikit-learn estimator with a `.fit(X, y)` / `.predict(X)` /
    `.predict_proba(X)` interface.

    TODO:
      - Import LogisticRegression from sklearn.linear_model.
      - Return LogisticRegression(max_iter=1000, random_state=config.seed).
      - (Optional) expose C as a config knob to tie into your regularization
        notes — smaller C = stronger regularization.
    """
    
    return LogisticRegression(max_iter=1000, random_state=config.seed)


def build_nn(input_dim, config):
    """Build and compile a Keras Sequential neural network.

    Parameters
    ----------
    input_dim : int   # number of features (X_train.shape[1])
    config : Config

    Returns
    -------
    a compiled tf.keras.Model

    TODO:
      - Build a Sequential model:
          * an Input layer of shape (input_dim,)
          * one Dense layer per entry in config.hidden_units, using
            config.hidden_activation (relu). Add kernel_regularizer=l2(config.l2)
            and a Dropout(config.dropout) layer if you want the stretch goal.
          * a final Dense(1, activation=config.output_activation)  # sigmoid
      - Compile with:
          * optimizer = Adam(learning_rate=config.learning_rate)
          * loss = 'binary_crossentropy'   (matches your logistic-loss notes)
          * metrics = ['accuracy', Precision, Recall]  (recall matters here)
      - Return the compiled model.

    Reminder from your notes: ReLU in the hidden layers, sigmoid on the output
    for binary classification. Binary cross-entropy is the same loss you derived
    for logistic regression.
    """

    # instantiate model
    model = Sequential()

    # add input layer
    model.add(Input(shape=(input_dim,)))

    # add hidden layers 
    for units in config.hidden_units:
        model.add(Dense(units, activation=config.hidden_activation, kernel_regularizer=L2(config.l2)))
    
    # add output layer
    model.add(Dense(1, activation=config.output_activation))

    # compile model
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy', Precision(), Recall()])
    
    return model
