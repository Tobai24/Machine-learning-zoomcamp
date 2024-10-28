import pickle

from flask import Flask
from flask import request
from flask import jsonify

with open('model2.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask("homework")

@app.route("/run", methods=['POST'])
def predict():
    data = request.get_json()
    
    X = dv.transform(data)
    predict = model.predict_proba(X)[:,1]
    decision = (predict >= 0.5)
    
    result = {
        "Probability": float(predict),
        "Decision" : bool(decision)
    }
    
    return jsonify(result)
    
if __name__ == "__main__":    
    app.run(debug=True, host='0.0.0.0', port=9696)