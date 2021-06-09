import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import GlobalMaxPooling1D, Dense, Bidirectional, \
                                    LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                                    Dropout, concatenate
import tensorflow_hub as hub
import tensorflow_text as text
from preprocessing import TextPreprocess


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """position: the number of positions
       d_model: the number of dimension for every position"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
          values are ethier 0: present, 1: masking (look-ahead) or padding.

    Returns:
    output, attention_weights
    """
    # print("query shape: {}".format(q.shape))
    # print("key shape: {}".format(k.shape))
    # print("value shape: {}".format(v.shape))
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention consists of four parts:
        1. Linear layers and split into heads.
        2. Scaled dot-product attention.
        3. Concatenation of heads.
        4. Final linear layer.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]  # Positional encoding does not participate in training.

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class TQC(tf.keras.Model):

    def __init__(self, preprocessor_dir, LaBSE_dir, bi_lstm_dim=768, dropout_rate=0.3):
        super(TQC, self).__init__()
        self.preprocessor = hub.KerasLayer(preprocessor_dir, trainable=False, input_shape=(None,))
        self.encoder = hub.KerasLayer(LaBSE_dir, trainable=False)
        self.bi_lstm = Bidirectional(LSTM(bi_lstm_dim, return_sequences=True))
        self.global_avg_pooling = GlobalAveragePooling1D()
        self.global_max_pooling = GlobalMaxPooling1D()
        self.dropout = Dropout(dropout_rate)
        self.ff_layer1 = Dense(2048, activation="relu")
        self.ff_layer2 = Dense(64, activation="relu")
        self.ff_layer3 = Dense(8, activation="relu")
        self.output_layer = Dense(1, activation="sigmoid")

    def call(self, data):
        # src_texts = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_src_text")
        # tgt_texts = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_tgt_text")

        src_x = self.preprocessor(data["input_src_text"])
        tgt_x = self.preprocessor(data["input_tgt_text"])

        src_x = self.encoder(src_x)["sequence_output"]
        tgt_x = self.encoder(tgt_x)["sequence_output"]

        src_x = tf.math.l2_normalize(src_x, axis=-1, epsilon=1e-12, name=None)
        tgt_x = tf.math.l2_normalize(tgt_x, axis=-1, epsilon=1e-12, name=None)

        # sequence_output = tf.concat([src_x, tgt_x], axis=-1)
        sequence_output = concatenate([src_x, tgt_x])

        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = self.bi_lstm(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = self.global_avg_pooling(bi_lstm)
        max_pool = self.global_max_pooling(bi_lstm)
        concat = concatenate([avg_pool, max_pool])
        dropout = self.dropout(concat)

        x = self.ff_layer1(dropout)
        x = self.ff_layer2(x)
        x = self.ff_layer3(8, activation="relu")(x)

        output = self.output_layer(x)

        return output

    # def train_step(self, data):


def build_model_with_preprocessor(max_seq_len, preprocessor_dir, LaBSE_dir):
    src_texts = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_src_text")
    tgt_texts = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_tgt_text")

    preprocessor = hub.KerasLayer(preprocessor_dir, trainable=False)
    encoder = hub.KerasLayer(LaBSE_dir, trainable=False)

    src_x = preprocessor(src_texts)
    tgt_x = preprocessor(tgt_texts)

    src_x = encoder(src_x)["default"]
    tgt_x = encoder(tgt_x)["default"]

    src_x = tf.math.l2_normalize(src_x, axis=1, epsilon=1e-12, name=None)
    tgt_x = tf.math.l2_normalize(tgt_x, axis=1, epsilon=1e-12, name=None)

    # np.matmul(english_embeds, np.transpose(italian_embeds))
    x = tf.concat([src_x, tgt_x], axis=1)
    #  x = GlobalMaxPooling1D(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model([src_texts, tgt_texts], output)

    return model


def build_model_with_preprocessor_and_lstm(preprocessor_dir, LaBSE_dir, softmax=False):
    """Once softmax output layer is turned on, make sure to onehot encode labeled data to shape (n, num_classes)"""

    src_texts = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_src_text")
    tgt_texts = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_tgt_text")

    preprocessor = hub.KerasLayer(preprocessor_dir, trainable=False)
    encoder = hub.KerasLayer(LaBSE_dir, trainable=False)

    src_x = preprocessor(src_texts)
    tgt_x = preprocessor(tgt_texts)

    src_x = encoder(src_x)["sequence_output"]
    tgt_x = encoder(tgt_x)["sequence_output"]

    # src_x = tf.math.l2_normalize(src_x, axis=-1, epsilon=1e-12, name=None)
    # tgt_x = tf.math.l2_normalize(tgt_x, axis=-1, epsilon=1e-12, name=None)

    # np.matmul(english_embeds, np.transpose(italian_embeds))
    # sequence_output = tf.concat([src_x, tgt_x], axis=-1)
    sequence_output = concatenate([src_x, tgt_x])

    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = Bidirectional(LSTM(768, return_sequences=True))(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = GlobalAveragePooling1D()(bi_lstm)
    max_pool = GlobalMaxPooling1D()(bi_lstm)
    concat = concatenate([avg_pool, max_pool])
    dropout = Dropout(0.3)(concat)

    x = Dense(2048, activation="relu")(dropout)
    x = Dense(512, activation="relu")(x)
    x = Dense(8, activation="relu")(x)

    if softmax:
        output = Dense(2, activation='softmax')(x)

    else:
        output = Dense(1, activation='sigmoid')(x)

    model = keras.Model([src_texts, tgt_texts], output)

    return model


if __name__ == "__main__":

    model = TQC_Model((100), (100), 2, 128, 4, 256, 20000, 20000, 10000)
    print(model.summary())
    keras.utils.plot_model(model, "tqc_sample.png", show_shapes=True)
