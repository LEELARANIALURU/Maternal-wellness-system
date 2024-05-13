function onClickedVisceralFat() {
    console.log("Visceral fat button clicked");

    var age = document.getElementById("age").value;
    var estFat = document.getElementById("estFat");
    var result = parseInt(age) * 2.32;
    estFat.innerHTML = "<h2>" + result + "</h2>";
}

