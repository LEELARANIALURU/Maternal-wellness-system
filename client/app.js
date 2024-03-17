
function onClickedEstimateRisk() {

    console.log("Estimate risk button clicked");
    
    var age = document.getElementById("age");
    var sbp = document.getElementById("sbp");
    var dbp = document.getElementById("dbp");
    var sugar = document.getElementById("sugar");
    var temp = document.getElementById("temp");
    var heart = document.getElementById("heart");
    var estRisk = document.getElementById("estRisk");
    var url = "http://127.0.0.1:5000/predict_risk";
  
    $.post(url, {
        age: parseInt(age.value),
        sbp: parseInt(sbp.value),
        dbp: parseInt(dbp.value),
        sugar: parseInt(sugar.value),
        temp: parseInt(temp.value),
        heart: parseInt(heart.value),
    },
    
    function(data, status) {
        console.log(data.estRisk);
        estRisk.innerHTML = "<h2>" + data.estRisk.toString();
        console.log(status);
    });
  }

  
function onClickedFetalRisk() {

    console.log("Fetal risk button clicked");
    
    var baseline = document.getElementById("baseline");
    var accelerations = document.getElementById("accelerations");
    var fetal_movement = document.getElementById("fetal_movement");
    var uterine_contractions = document.getElementById("uterine_contractions");
    var light_decelerations = document.getElementById("light_decelerations");
    var severe_decelerations = document.getElementById("severe_decelerations");
    var prolongued_decelerations = document.getElementById("prolongued_decelerations");
    var abnormal_short_term_variability = document.getElementById("abnormal_short_term_variability");
    var mean_value_of_short_term_variability = document.getElementById("mean_value_of_short_term_variability");
    var percentage_of_time_with_abnormal_long_term_variability = document.getElementById("percentage_of_time_with_abnormal_long_term_variability");
    var mean_value_of_long_term_variability = document.getElementById("mean_value_of_long_term_variability");
    var histogram_width = document.getElementById("histogram_width");
    var histogram_min = document.getElementById("histogram_min");
    var histogram_max = document.getElementById("histogram_max");
    var histogram_number_of_peaks = document.getElementById("histogram_number_of_peaks");
    var histogram_mode = document.getElementById("histogram_mode");
    var histogram_mean = document.getElementById("histogram_mean");
    var histogram_median = document.getElementById("histogram_median");
    var histogram_variance = document.getElementById("histogram_variance");
    var histogram_tendency = document.getElementById("histogram_tendency");
    var fetalRisk = document.getElementById("fetalRisk");
    var url = "http://127.0.0.1:5000/fetal_risk";
  
    $.post(url, {
        baseline: parseFloat(baseline.value),
        accelerations: parseFloat(accelerations.value),
        fetal_movement: parseFloat(fetal_movement.value),
        uterine_contractions: parseFloat(uterine_contractions.value),
        light_decelerations: parseFloat(light_decelerations.value),
        severe_decelerations: parseFloat(severe_decelerations.value),
        prolongued_decelerations: parseFloat(prolongued_decelerations.value),
        abnormal_short_term_variability: parseFloat(abnormal_short_term_variability.value),
        mean_value_of_short_term_variability: parseFloat(mean_value_of_short_term_variability.value),
        percentage_of_time_with_abnormal_long_term_variability: parseFloat(percentage_of_time_with_abnormal_long_term_variability.value),
        mean_value_of_long_term_variability: parseFloat(mean_value_of_long_term_variability.value),
        histogram_width: parseFloat(histogram_width.value),
        histogram_min: parseFloat(histogram_min.value),
        histogram_max: parseFloat(histogram_max.value),
        histogram_number_of_peaks: parseFloat(histogram_number_of_peaks.value),
        histogram_mode: parseFloat(histogram_mode.value),
        histogram_mean: parseFloat(histogram_mean.value),
        histogram_median: parseFloat(histogram_median.value),
        histogram_variance: parseFloat(histogram_variance.value),
        histogram_tendency: parseFloat(histogram_tendency.value),
    },
    
    function(data, status) {
        console.log(data.fetalRisk);
        fetalRisk.innerHTML = "<h2>" + data.fetalRisk.toString();
        console.log(status);
    });
  }