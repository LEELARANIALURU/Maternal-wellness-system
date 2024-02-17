
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