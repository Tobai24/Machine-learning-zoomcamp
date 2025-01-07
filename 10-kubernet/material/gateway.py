import grpc
import tensorflow as tf
from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from flask import Flask, request, jsonify
from proto import np_to_protobuf



classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

host = 'localhost:8500'
channel = grpc.insecure_channel(host) 
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
preprocessor = create_preprocessor('xception', target_size = (299, 299))


def prepare_request(X):
    # Initialize the request
    pb_request = predict_pb2.PredictRequest()

    # Specify the model name
    pb_request.model_spec.name = 'clothing-model'
    # Specify the name of the signature
    pb_request.model_spec.signature_name = 'serving_default'
    # Specify the input data
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))
    return pb_request

def make_prediction(url):
    X = preprocessor.from_url(url)
    
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout = 20.0)

    # Get predictions
    preds = pb_response.outputs['dense_7'].float_val
    return dict(zip(classes, preds))

app = Flask("gateway")


@app.route('/predict', methods = ['POST'])
def predict_endpoint():
    # Get the data
    data = request.get_json()
    # Get url
    url = data['url']
    # Get result
    result = make_prediction(url)
    # output
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)

# if __name__ == '__main__':
#     url = 'http://bit.ly/mlbookcamp-pants'
#     response = make_prediction(url)
#     print(response)


