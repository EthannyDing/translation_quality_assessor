import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import sklearn
import re, string, os
import random

# Force Keras to use CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def read_text(file):
    with open(file, "r") as f:
        lines = f.read().splitlines()

    return lines

def write_text(filepath, lines):
    with open(filepath, 'w') as f:
        for l in lines:
            f.write(l.strip() + "\n")

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


def create_dev_from_train(train_files, test_files, output_dev_files, dev_num_per_class=400):
    """Create dev dataset from training data that's different from test dataset."""
    def create_dev_data_by_class(train_eng, train_fra, test_eng, test_fra):
        dev_eng, dev_fra = [], []
        for eng, fra in zip(train_eng, train_fra):
            if (eng not in test_eng) and (fra not in test_fra):
                dev_eng.append(eng)
                dev_fra.append(fra)
                if len(dev_eng) >= dev_num_per_class:
                    break

        return dev_eng, dev_fra

    train_eng_good = read_text(train_files[0])
    train_fra_good = read_text(train_files[1])
    train_eng_bad = read_text(train_files[2])
    train_fra_bad = read_text(train_files[3])

    test_eng_good = read_text(test_files[0])
    test_fra_good = read_text(test_files[1])
    test_eng_bad = read_text(test_files[2])
    test_fra_bad = read_text(test_files[3])

    # dev_eng_good, dev_fra_good, dev_eng_bad, dev_fra_bad = [], [], [], []

    dev_eng_good, dev_fra_good = create_dev_data_by_class(train_eng_good, train_fra_good,
                                                          test_eng_good, test_fra_good)
    dev_eng_bad, dev_fra_bad = create_dev_data_by_class(train_eng_bad, train_fra_bad,
                                                        test_eng_bad, test_fra_bad)
    write_text(output_dev_files[0], dev_eng_good)
    write_text(output_dev_files[1], dev_fra_good)
    write_text(output_dev_files[2], dev_eng_bad)
    write_text(output_dev_files[3], dev_fra_bad)


def extract_tb_from_tm(input_tm, output_tb, token_len=5):

    eng_tm = read_text(input_tm[0])
    fra_tm = read_text(input_tm[1])

    eng_tb = []
    fra_tb = []
    for eng, fra in zip(eng_tm, fra_tm):
        if len(eng.split()) < token_len and len(fra.split()) < token_len:
            eng_tb.append(eng)
            fra_tb.append(fra)

    write_text(output_tb[0], eng_tb)
    write_text(output_tb[1], fra_tb)


class TextPreprocessOld:

    def __init__(self, srcLang="eng", tgtLang="fra",
                       src_vocab_size=20000, src_len=200,
                       tgt_vocab_size=20000, tgt_len=200):
        super(TextPreprocessOld, self).__init__()
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

        src_lines = np.array(src_lines)
        tgt_lines = np.array(tgt_lines)
        labels = np.array(labels)

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


class TextPreprocess:

    def __init__(self, srcLang="eng", tgtLang="fra"):
        super(TextPreprocess, self).__init__()
        # self.batch_size = batch_size
        self.srcLang = srcLang
        self.tgtLang = tgtLang

    def suffle_data(self,
                    src_lines,
                    tgt_lines,
                    labels,
                    random_state=1):
        """Shuffle data mainly for training data."""
        random.seed(random_state)
        random.shuffle(src_lines)
        random.seed(random_state)
        random.shuffle(tgt_lines)
        random.seed(random_state)
        random.shuffle(labels)

        return src_lines, tgt_lines, labels

    def train_test_split(self, src_text, tgt_text, labels, train_ratio=0.8):

        num_train_samples = int(len(src_text) * train_ratio)

        train_src_texts = src_text[:num_train_samples]
        train_tgt_texts = tgt_text[:num_train_samples]
        train_labels = labels[:num_train_samples]

        test_src_texts = src_text[num_train_samples:]
        test_tgt_texts = tgt_text[num_train_samples:]
        test_labels = labels[num_train_samples:]

        print("\nTrain samples : {}".format(len(train_labels)))
        print("Test samples  : {}".format(len(test_labels)))

        return ((train_src_texts, train_tgt_texts), train_labels), ((test_src_texts, test_tgt_texts), test_labels)

    def read_dataset_from_directory(self,
                                    data_dir,
                                    label_class_map,
                                    shuffle=True,
                                    random_state=1,
                                    drop_duplicates=True):
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

        df = pd.DataFrame(zip(src_lines, tgt_lines, labels), columns=["src", "tgt", "label"])
        if drop_duplicates:
            df = df.drop_duplicates(keep="first")

        if shuffle:
            df = sklearn.utils.shuffle(df, random_state=random_state).reset_index(drop=True)

        counter = Counter(df["label"])
        print("Importing Data Complete.")
        print("\t{} good entries".format(counter[label_class_map["good"]]))
        print("\t{} bad entries".format(counter[label_class_map["bad"]]))

        src_lines = df["src"].to_numpy()
        tgt_lines = df["tgt"].to_numpy()
        labels = df["label"].to_numpy()

        return src_lines, tgt_lines, labels

    def onehot_encoding_label_data(self,
                                   label_data,
                                   num_classes=2):
        """One-hot encoding label data shape from (n, 1) to (n, num_class).
            For example:
                array([1,2,0,1,0,1])  ----->  array([[0, 1, 0], [0, 0, 1], [1, 0, 0],
                                                     [0, 1, 0], [1, 0, 0], [0, 1, 0]])
            """
        onehot_label_data = tf.keras.utils.to_categorical(label_data, num_classes=num_classes)
        return onehot_label_data

    def create_datasets(self,
                        data_dir,
                        label_class_map,
                        train_test_split_train_ratio=0.2,
                        data_generator=True,
                        batch_size=32,
                        num_classes=2,
                        onehot_encoding=False,
                        shuffle=True,
                        random_state=1):
        """Create datasets used for training and testing from local files"""

        src_lines, tgt_lines, labels = self.read_dataset_from_directory(data_dir,
                                                                        label_class_map,
                                                                        shuffle=shuffle,
                                                                        random_state=random_state)
        if onehot_encoding:
            labels = self.onehot_encoding_label_data(labels, num_classes)

        train_dataset, test_dataset = self.train_test_split(src_lines, tgt_lines, labels,
                                                            train_ratio=train_test_split_train_ratio)
        if data_generator:
            train_dataset = tf.data.Dataset.from_tensor_slices(({"input_src_text": train_dataset[0][0],
                                                                 "input_tgt_text": train_dataset[0][1]},
                                                                train_dataset[1])).batch(batch_size)

        # test_dataset = tf.data.Dataset.from_tensor_slices(({"input_src_text": test_dataset[0][0],
        #                                                     "input_tgt_text": test_dataset[0][1]},
        #                                                    test_dataset[1])).batch(batch_size)
        return train_dataset, test_dataset


def test_read_dataset_from_directory():
    tp = TextPreprocess()
    data_dir = "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/train"
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

def test_create_dev():

    train_files = ["/linguistics/ethan/DL_Prototype/datasets/TB_TQA/CPA_tb_QA_202103_shuffle.good.eng",
                   "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/CPA_tb_QA_202103_shuffle.good.fra",
                   "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/CPA_tb_QA_202103_shuffle.bad.eng",
                   "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/CPA_tb_QA_202103_shuffle.bad.fra"]
    test_files = ["/linguistics/ethan/DL_Prototype/datasets/TB_TQA/test/CPA_test_tb.good.eng",
                  "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/test/CPA_test_tb.good.fra",
                  "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/test/CPA_test_tb.bad.eng",
                  "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/test/CPA_test_tb.bad.fra"]
    output_dev_files = ["/linguistics/ethan/DL_Prototype/datasets/TB_TQA/dev/CPA_dev_tb.good.eng",
                        "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/dev/CPA_dev_tb.good.fra",
                        "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/dev/CPA_dev_tb.bad.eng",
                        "/linguistics/ethan/DL_Prototype/datasets/TB_TQA/dev/CPA_dev_tb.bad.fra"]
    create_dev_from_train(train_files, test_files, output_dev_files, dev_num_per_class=400)

def test_extract_tb_from_tm():

    input_tm = ["/linguistics/ethan/DL_Prototype/datasets/Human_QA_finance_202102.good.eng",
                "/linguistics/ethan/DL_Prototype/datasets/Human_QA_finance_202102.good.fra"]
    output_tb = ["/linguistics/ethan/DL_Prototype/datasets/tb_Human_QA_finance_202102.good.eng",
                 "/linguistics/ethan/DL_Prototype/datasets/tb_Human_QA_finance_202102.good.fra"]
    extract_tb_from_tm(input_tm, output_tb, token_len=5)


if __name__ == "__main__":

    # test_read_dataset_from_directory()
    # test_create_dev()
    test_extract_tb_from_tm()
