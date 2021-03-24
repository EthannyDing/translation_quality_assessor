import os
from model import TQC_Model
from preprocessing import TextPreprocess


def train(model, train_ds, train_labels, epochs, optimizer="adam", **hparams):

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    model.fit(train_ds, train_labels, epochs=epochs)

    return model


def Main():

    # hyper-parameters
    srcLang = "eng"
    tgtLang = "fra"
    src_vocab_size = 20000
    src_len = 150
    tgt_vocab_size = 20000
    tgt_len = 150

    num_layers = 6  # the number of encoder layer for both source and target
    d_model = 128   # dimension of word for both source and target
    num_heads = 8   # the number of heads for both source and target
    dff = 2048
    maximum_position_encoding = 10000

    epochs = 3
    optimizer = "adam"

    label_class_map = {"good": 1, "bad": 0}

    # get data ready
    print("------------------------------------------------------------")
    print("Reading and preprocessing data.")
    rootpath = os.path.abspath("..")
    train_data_dir = os.path.join(rootpath, "datasets/tqa/train")
    test_data_dir = os.path.join(rootpath, "datasets/tqa/test")

    # src_file = os.path.join(rootpath, "datasets/tqa", "test.eng")
    # tgt_file = os.path.join(rootpath, "datasets/tqa", "test.fra")
    # label_file = os.path.join(rootpath, "datasets/tqa", "test.labels")

    tp = TextPreprocess(src_vocab_size=src_vocab_size, src_len=src_len,
                        tgt_vocab_size=tgt_vocab_size, tgt_len=tgt_len)
    src_integers, tgt_integers, labels = tp.create_datasets(train_data_dir, label_class_map, mode='train')
    test_src_integers, test_tgt_integers, test_labels = tp.create_datasets(test_data_dir, label_class_map, mode='test')

    # get model and start training
    print("------------------------------------------------------------")
    print("Initializing and training model.")
    model = TQC_Model((src_len), (tgt_len),
                      num_layers, d_model, num_heads, dff,
                      src_vocab_size, tgt_vocab_size, maximum_position_encoding)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    model.fit(x=[src_integers, tgt_integers], y=labels, validation_split=0.1, epochs=epochs)


if __name__ == "__main__":

    Main()
