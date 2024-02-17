import pickle
import numpy as np

__model = None

def get_estimated_risk(age, sbp, dbp, sugar, temp, heart):

    x = np.zeros(6)
    x[0] = age
    x[1] = sbp
    x[2] = dbp
    x[3] = sugar
    x[4] = temp
    x[5] = heart
    
    return (__model.predict([x])[0])


def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __model
    if __model is None:
        with open(r'C:\Users\SID\Desktop\minor_proj_latest\Maternal-wellness-system\server\artifacts\matRisk.pickle', 'rb') as f:
            f.seek(0)
            __model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    
    print(get_estimated_risk(29, 90, 70, 8, 100, 80))