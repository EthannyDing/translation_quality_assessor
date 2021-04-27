import json
import requests
import time
import numpy as np
import tensorflow as tf
from preprocessing import read_text, write_text

"""For model serving using tensorflow-model-server, a step by step guidance reference to:
    https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

   For utilizing GPU for tensorflow serving, reference to:
   https://www.tensorflow.org/tfx/serving/docker

   While serving, the rest API call should follow the required data format, especially for
   input data, reference this page for json data format: 
    
    https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_row_format
    """

"tensorflow_model_server " \
"--rest_api_port=8501 " \
"--model_name='alexa_tqc' " \
"--model_base_path='/linguistics/ethan/DL_Prototype/models/TQA_models/Multilingual-LaBSE-Bidirectional-LSTM_ckpts/tqc-0010.ckpt'"

"""Command for lanuching model in nvidia docker container:
sudo docker run --runtime=nvidia -p 8501:8501 \
--mount type=bind,\
source=/linguistics/ethan/DL_Prototype/models/TQA_models/Multilingual-LaBSE-Bidirectional-LSTM_ckpts/tqc-0010.ckpt,\
target=/models/alexa_tqc \
-e MODEL_NAME=alexa_tqc -t tensorflow/serving:2.3.0-gpu

Note: Based on your CUDA toolkit and driver version, there could be incompatibility by 
      running tensorflow/serving:latest, which as of now requires CUDA 11.0. You need to make sure the version of 
      tensorflow/serving image is compatible with the CUDA your host server. Reference to the tensorflow/serving image
      page to get version information: https://github.com/tensorflow/serving/releases"""


def predict(src_texts, tgt_texts, predict_classes=False, cutoff=0.5):
    """Make sure model is in serving mode."""
    model_input_names = ["input_src_text", "input_tgt_text"]
    endpoint = 'http://localhost:8501/v1/models/alexa_tqc:predict'
    # endpoint = 'http://localhost:8501/v1/models/alexa_tqc:predict_classes'
    headers = {"content-type": "application/json"}
    data = json.dumps({"signature_name": "serving_default",
                       "instances": [{model_input_names[0]: src_texts,
                                      model_input_names[1]: tgt_texts},
                                     ]
                       })

    response = requests.post(endpoint, headers=headers, data=data)
    # print(json.loads(response.text))
    predictions = json.loads(response.text)["predictions"]

    if predict_classes:
        predictions = np.where(np.array(predictions) > cutoff, 1, 0)
        predictions = predictions.reshape(-1, )

    return predictions


def predict_large_inputs(src_texts, tgt_texts, predict_classes=False, batch_size=100, cutoff=0.5):
    """Used for predicting large number of input instances, e.g. 500.
       Divide total input data into batches and send each batch at a time."""

    total_num_inputs = len(src_texts)
    input_data = tf.data.Dataset.from_tensor_slices((src_texts, tgt_texts)).batch(batch_size)
    results = np.array([], dtype=int)
    for i, (src_batch, tgt_batch) in enumerate(input_data):

        # src_batch, tgt_batch = src_batch.numpy().astype(str).tolist(), tgt_batch.numpy().astype(str).tolist()
        src_batch = np.char.decode(src_batch.numpy().astype(np.bytes_), "UTF-8").tolist()
        tgt_batch = np.char.decode(tgt_batch.numpy().astype(np.bytes_), "UTF-8").tolist()

        predictions = predict(src_batch, tgt_batch, predict_classes=predict_classes)
        results = np.append(results, predictions)

        print("{} batches completed, {} pairs predicted".format(i + 1, min((i+1)*batch_size, total_num_inputs)))

    return results


def test_predict():

    src_texts = ["Property Group Company Limited",
                 "Amazon also indicated that it has moved its AI plans from hype to reality.",
                 "A regulation may not be made before the earliest of",
                 "Nancy J. Kyle is a vice chairman and CGTC.",
                 "You can copy-paste"] * 10
    tgt_texts = ["Poly Property Group Company Limited",
                 "Amazon a aussi indiqué que ses projets d’IA, jusqu’alors des mythes, étaient devenus réalité.",
                 "Le règlement ne peut être pris avant le premier en date des jours suivants",
                 "Nancy J. Kyle est vice-présidente du conseil d’administration et",
                 "Il est possible de copier-coller"] * 10

    start = time.time()
    predictions = predict(src_texts, tgt_texts, predict_classes=True)
    # predicted_classes = np.where(np.array(predictions) > 0.5, 1, 0)
    end = time.time()

    print(predictions)
    print("Time latency for predicting {} pairs of sentences: {} sec, {}s per pair".format(
            len(src_texts),
            round(end-start, 2),
            round(end-start, 2) / len(src_texts)))

def test_predict_large_input_data():

    eng_file = "/linguistics/ethan/DL_Prototype/evaluation/FRA-ENG/law/pem_law_eng_fra_20210310_test_pred.eng"
    fra_trans_file = "/linguistics/ethan/DL_Prototype/evaluation/FRA-ENG/law/pem_law_eng_fra_20210310_test.fra"
    output_classes_file = "/linguistics/ethan/DL_Prototype/evaluation/FRA-ENG/law/pem_law_eng_fra_20210310.tqa_pred"

    eng_lines = read_text(eng_file)
    fra_lines = read_text(fra_trans_file)

    classes = predict_large_inputs(eng_lines, fra_lines,
                                   predict_classes=True, batch_size=100, cutoff=0.5)
    class_1_num = sum(classes == 1)
    print("\nPredictions contribution: 1 - {}, 0 - {}".format(class_1_num, 1000 - class_1_num))
    classes = classes.astype(str)  # convert numpy int type to string to save in file.
    write_text(output_classes_file, classes)


if __name__ == "__main__":

    # test_predict()
    test_predict_large_input_data()
