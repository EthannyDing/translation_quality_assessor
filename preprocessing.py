import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re, string, os
import random

# Force Keras to use CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def read_text(file):
    with open(file, "r") as f:
        lines = f.read().splitlines()

    return lines


def read_dataset_from_files(src_file, tgt_file, label_file, shuffle=True):
    """Read data from txt file and shuffle if specified."""
    src_lines = read_text(src_file)
    tgt_lines = read_text(tgt_file)

    labels = read_text(label_file)
    labels = list(map(int, labels))
    labels = np.array(labels).reshape(len(labels), 1)

    if shuffle:
        random.seed(1)
        random.shuffle(src_lines)
        random.seed(1)
        random.shuffle(tgt_lines)
        random.seed(1)
        random.shuffle(labels)

    return src_lines, tgt_lines, labels


class TextPreprocess(tf.keras.layers.Layer):

    def __init__(self, srcLang="eng", tgtLang="fra",
                       src_vocab_size=20000, src_len=200,
                       tgt_vocab_size=20000, tgt_len=200):
        super(TextPreprocess, self).__init__()
        # self.batch_size = batch_size
        self.srcLang = srcLang
        self.tgtLang = tgtLang
        self.src_vocab_size = src_vocab_size
        self.src_len = src_len
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_len = tgt_len
        self.src_text_vectorizer = TextVectorization(standardize=self.custom_standardization,
                                                     max_tokens=self.src_vocab_size,
                                                     output_mode="int",
                                                     output_sequence_length=self.src_len)
        self.tgt_text_vectorizer = TextVectorization(standardize=self.custom_standardization,
                                                     max_tokens=self.tgt_vocab_size,
                                                     output_mode="int",
                                                     output_sequence_length=self.tgt_len)

    def suffle_data(self, src_lines, tgt_lines, labels, random_state=1):
        """Shuffle data mainly for training data."""
        random.seed(random_state)
        random.shuffle(src_lines)
        random.seed(random_state)
        random.shuffle(tgt_lines)
        random.seed(random_state)
        random.shuffle(labels)

        return src_lines, tgt_lines, labels

    def read_dataset_from_directory(self, data_dir, label_class_map, shuffle=True):
        """Read TQA data from directory where the label is indicated in file name."""
        print("\nImporting Data")
        files = os.listdir(data_dir)
        good_src_prefix = [file.replace(".good." + self.srcLang, "") for file in files if file.endswith(".good." + self.srcLang)]
        good_tgt_prefix = [file.replace(".good." + self.tgtLang, "") for file in files if file.endswith(".good." + self.tgtLang)]
        bad_src_prefix = [file.replace(".bad." + self.srcLang, "") for file in files if file.endswith(".bad." + self.srcLang)]
        bad_tgt_prefix = [file.replace(".bad." + self.tgtLang, "") for file in files if file.endswith(".bad." + self.tgtLang)]

        assert set(good_src_prefix) == set(good_tgt_prefix), \
            "The number of good English and French file pairs not equal."

        assert set(bad_src_prefix) == set(bad_tgt_prefix), \
            "The number of bad English and French file pairs not equal."

        print("\t{} pairs of good English-French files found.".format(len(good_src_prefix)))
        print("\t{} pairs of bad English-French files found.".format(len(bad_src_prefix)))

        all_prefix_by_class = [prefix + ".good" for prefix in good_src_prefix] + \
                              [prefix + ".bad" for prefix in bad_src_prefix]
        src_lines = []
        tgt_lines = []
        labels = []

        for prefix in all_prefix_by_class:

            label = prefix.split(".")[-1]
            en_path = os.path.join(data_dir, prefix + "." + self.srcLang)
            fr_path = os.path.join(data_dir, prefix + "." + self.tgtLang)
            g_en_lines = read_text(en_path)
            g_fr_lines = read_text(fr_path)

            if len(g_en_lines) == len(g_fr_lines):
                class_num = label_class_map.get(label)
                src_lines += g_en_lines
                tgt_lines += g_fr_lines
                labels += [class_num] * len(g_en_lines)

        if shuffle:
            src_lines, tgt_lines, labels = self.suffle_data(src_lines, tgt_lines, labels)
        counter = Counter(labels)
        print("Importing Data Complete.")
        print("\t{} good entries".format(counter[label_class_map["good"]]))
        print("\t{} bad entries".format(counter[label_class_map["bad"]]))

        # src_lines = np.array(src_lines).reshape(len(src_lines), 1)
        # tgt_lines = np.array(tgt_lines).reshape(len(tgt_lines), 1)
        # labels = np.array(labels).shape(len(labels), 1)

        return src_lines, tgt_lines, labels

    def custom_standardization(self, input_data):
        """Customized manipulations on raw text."""
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape(string.punctuation), ""
        )

    def create_integer_ds(self, src_text, tgt_text):
        # src_text, tgt_text = samples

        src_int_samples = self.src_text_vectorizer(src_text)
        tgt_int_samples = self.tgt_text_vectorizer(tgt_text)

        return src_int_samples, tgt_int_samples

    def create_datasets(self, data_dir, label_class_map, mode="train", batch_size=32):
        """Create datasets used for training and testing from local files"""

        src_lines, tgt_lines, labels = self.read_dataset_from_directory(data_dir, label_class_map, shuffle=True)

        # labels = tf.data.Dataset.from_tensor_slices((labels)).batch(batch_size)
        if mode == "train":

            # train_src_text = dataset.map(lambda src, tgt: src)
            # train_tgt_text = dataset.map(lambda src, tgt: tgt)
            # print("Creating vocabulary for training source texts...")
            # self.src_text_vectorizer.adapt(train_src_text)
            # print("Creating vocabulary for training target texts...")
            # self.tgt_text_vectorizer.adapt(train_tgt_text)
            # dataset = dataset.map(self.create_integer_ds)
            print("Creating vocabulary for training source and target texts...")
            self.src_text_vectorizer.adapt(src_lines)
            self.tgt_text_vectorizer.adapt(tgt_lines)
            print("Mapping texts into integer repsentations...")
            src_integers, tgt_integers = self.create_integer_ds(src_lines, tgt_lines)
            # print(src_integers.shape)
            # print(labels.shape)
            # dataset = tf.data.Dataset.from_tensor_slices(({"input_1": src_integers, "input_2": tgt_integers},
            #                                               labels)).batch(batch_size)

        elif mode == "test":

            print("Mapping texts into integer repsentations...")
            src_integers, tgt_integers = self.create_integer_ds(src_lines, tgt_lines)
            # dataset = tf.data.Dataset.from_tensor_slices(({"input_1": src_integers, "input_2": tgt_integers},
            #                                               labels)).batch(batch_size)
            # test_ds = tf.data.Dataset.from_tensor_slices(([src_lines, tgt_lines], labels)).batch(batch_size)
            # dataset = dataset.map(self.create_integer_ds)

        else:
            raise ValueError("Please select mode between 'train' and 'test'.")

        return src_integers, tgt_integers, labels


def test_read_dataset_from_directory():
    tp = TextPreprocess(src_vocab_size=20000, src_len=100, tgt_vocab_size=20000, tgt_len=100)
    data_dir = "/linguistics/ethan/DL_prototype/datasets/tqa/train"
    label_class_map = {"good": 1, "bad": 0}
    src_lines, tgt_lines, labels = tp.read_dataset_from_directory(data_dir, label_class_map, shuffle=True)

def test_create_datasets():

    tp = TextPreprocess(src_vocab_size=20000, src_len=100, tgt_vocab_size=20000, tgt_len=100)
    rootpath = os.path.abspath("..")
    src_file = os.path.join(rootpath, "datasets/tqa", "test.eng")
    tgt_file = os.path.join(rootpath, "datasets/tqa", "test.fra")
    label_file = os.path.join(rootpath, "datasets/tqa", "test.labels")
    print(rootpath)
    src_integers, tgt_integers, labels = tp.create_datasets(src_file, tgt_file, label_file, mode='train')
    # print(train_ds)
    # for inputs, l in train_ds.take(1):
    #     print(inputs["input_1"].shape)
    #     print(l.shape)
    print(src_integers)
    print(labels)

if __name__ == "__main__":

    test_read_dataset_from_directory()
