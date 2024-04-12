import pickle
import numpy as np

__model = None
fetalModel = None

def get_estimated_risk(age, sbp, dbp, sugar, temp, heart):

    x = np.zeros(6)
    x[0] = age
    x[1] = sbp
    x[2] = dbp
    x[3] = sugar
    x[4] = temp
    x[5] = heart
    
    return (__model.predict([x])[0])

def get_fetal_risk(baseline, accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations,
                                                 prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability,
                                                 percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability,
                                                 histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_mode, histogram_mean,
                                                 histogram_median, histogram_variance, histogram_tendency):

    x = np.zeros(20)
    x[0] = baseline
    x[1] = accelerations
    x[2] = fetal_movement
    x[3] = uterine_contractions
    x[4] = light_decelerations
    x[5] = severe_decelerations
    x[6] = prolongued_decelerations
    x[7] = abnormal_short_term_variability
    x[8] = mean_value_of_short_term_variability
    x[9] = percentage_of_time_with_abnormal_long_term_variability
    x[10] = mean_value_of_long_term_variability
    x[11] = histogram_width
    x[12] = histogram_min
    x[13] = histogram_max
    x[14] = histogram_number_of_peaks
    x[15] = histogram_mode
    x[16] = histogram_mean
    x[17] = histogram_median
    x[18] = histogram_variance
    x[19] = histogram_tendency
    
    return (fetalModel.predict([x])[0])

def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __model
    global fetalModel
    if __model is None:
        with open(r'C:\Users\hp\OneDrive\Desktop\Maternal-wellness-system\server\artifacts\matRisk.pickle', 'rb') as f:
            f.seek(0)
            __model = pickle.load(f)
    if fetalModel is None:
        with open(r'C:\Users\hp\OneDrive\Desktop\Maternal-wellness-system\server\artifacts\fetRisk.pickle', 'rb') as f1:
            f1.seek(0)
            fetalModel = pickle.load(f1)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    
    # print(get_estimated_risk(29, 90, 70, 8, 100, 80))