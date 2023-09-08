from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pickle


app = Flask(__name__)


url = ("postgresql://fl0user:AcXRgUOS2m6L@ep-weathered-bar-74489110.ap-southeast-1.aws.neon.tech:5432/DDBB-ApiModel?sslmode=require")
engine = create_engine(url) 


with open('./Models/Model_Iris.pkl', 'rb') as archive:
    model = pickle.load(archive)


def transform_column(features, n_column=0):
    data = features.copy()
    for i, _ in enumerate(features):
        if data[i][n_column] > 3:
            data[i][n_column] = 1.0
        else:
            data[i][n_column] = 0.0
    return data


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/api/v0/get_predict', methods=['POST'])
def get_predict():
    try: 
        sepal_len = float(request.form.get('sepal_len', None))
        sepal_wid = float(request.form.get('sepal_wid', None))
        petal_len = float(request.form.get('petal_len', None))
        petal_wid = float(request.form.get('petal_wid', None))
        date = dt.today().strftime("%Y-%m-%d %H:%M:%S")

    except:
        doc = """<div style='color:red;font-size:30;'>
                Don't worry, I think so....</div>"""
        return doc 

    array_sample = np.array([sepal_len, sepal_wid, petal_len, petal_wid])


    # HAGO PREDICCIONES
    array_shape = array_sample.reshape(1, -1)
    array_trans = transform_column(array_shape, 0)
    y_pred = model.predict(array_trans)

    
    # CREO EL DATAFRAME
    model_class = str(model.__class__)[:32]

    columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
    data = pd.DataFrame(array_shape, columns=columns)
    data['model_predict'] = float(y_pred)
    data['model_name'] = model_class
    data['predict_date'] = pd.to_datetime(date)

    # SUBO LA DATA A LA BASE DE DATOS CON PANDAS
    data.to_sql('LogReg_Model', if_exists = "append", con=engine, index=False)
    
    # RETURN
    target = load_iris()['target_names'][y_pred[0]]
    header = '<div style="color:blue;font-size:18;">Result of prediction:</div>'
    response = f'{header} {target.title()}'

    return response


@app.route('/api/v0/retraining', methods=['GET'])
def retraining():
    df_querys = pd.read_sql_table('LogReg_Model', con=engine)
    
    df_iris = pd.DataFrame(load_iris()['data'], columns=df_querys.columns[1:5])
    df_iris['model_predict'] = load_iris()['target']

    df_merge = pd.merge(df_querys.iloc[:, 1:6], df_iris, how='outer', indicator=True)
    df_merge_filt = df_merge[(df_merge['_merge']=='left_only') | (df_merge['_merge']=='right_only')]

    X = df_merge_filt.iloc[:, :-2]
    y = df_merge_filt.iloc[:, -2]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    X_train_arr = np.array(X_train)
    X_test_arr = np.array(X_test)

    X_train_trans = transform_column(X_train_arr, 0)
    X_test_trans = transform_column(X_test_arr, 0)

    y_train_arr = y_train.astype(int)
    y_test_arr = y_test.astype(int)

    model.fit(X_train_trans, y_train_arr)
    y_pred = model.predict(X_test_trans)
    accuracy = str(accuracy_score(y_test_arr, y_pred))

    with open('./Models/Model_Iris_retraining.pkl', 'wb') as archivo:
        pickle.dump(model, archivo)

    doc = '''
<div style = "color:blue;font-size:27;">SUCCESSFUL UPDATE!</div>
The prediction score is:
'''

    score = str(round(float(accuracy), 4))

    return doc + score


if __name__ == '__main__':
    app.run(debug=True)
