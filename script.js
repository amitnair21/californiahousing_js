var regModel; // Final usable model
var regReady = false; // Indicates if model is ready
var epochvalue = 0; // Epoch value, selected by user
var splitpercent = 0.8; // Percent of data used for training

var rows = []; // Rows in data

var testFeatures;
var testLabels;
var trainFeatures;
var trainLabels;

// [ L O A D ]----------------------------------------------------------------------------------------------------

// To trigger an action when the user confirms a new data file
document.getElementById("confirm").addEventListener("click", async () => {
  verifyFile();
});

// To verify the new data file
async function verifyFile() {
  
  // Restore all defaults
  regReady = false;
  document.getElementById("load-tick").style.display = "none";
  document.getElementById("train-options").style.display = "none";
  document.getElementById("train-tick").style.display = "none";
  document.getElementById("train-img").style.display = "none";
  document.getElementById("input-r").style.display = "none";
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("perform-tick").style.display = "none";
  document.getElementById("regPrediction").value = "";
  document.getElementById("regComparison").value = "";

  let requiredheaders =
    "longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity";
  var csvFile = document.getElementById("csv-file").files[0];
  var data = await csvFile.text();
  var rows = data.split("\n");
  let headers = rows[0].split(",");

  // Removing spaces from headers to obtain a comparable string value
  if (
    headers.join(",").replace(/\s/g, "") ===
    requiredheaders.split(",").join(",").replace(/\s/g, "")
  ) {
    document.getElementById("load-status").style.display = "none";
    document.getElementById("load-tick").style.display = "inline";
    document.getElementById("train-options").style.display = "inline";
    document.getElementById("train-status").innerText = "";
  } else {
    document.getElementById("load-status").style.display = "inline";
  }
}

// To confirm Epoch selection
document.getElementById("r1").addEventListener("click", async () => {
  confirmTraining(5);
});
document.getElementById("r2").addEventListener("click", async () => {
  confirmTraining(50);
});
document.getElementById("r3").addEventListener("click", async () => {
  confirmTraining(300);
});
document.getElementById("r4").addEventListener("click", async () => {
  confirmTraining(500);
});
document.getElementById("r5").addEventListener("click", async () => {
  confirmTraining(1000);
});

// [ T R A I N ]----------------------------------------------------------------------------------------------------

// Return a trained model
async function trainModel() {
  
  // Prevent the user from confirming another file during training
  document.getElementById("confirm").style.display = "none";

  var csvFile = document.getElementById("csv-file").files[0];
  var data = await csvFile.text();
  rows = data.split("\n");
  var headers = rows[0].split(",");
  var features = [];
  var labels = [];
  for (let i = 1; i < rows.length; i++) {
    var rowcol = rows[i].split(",");
    
    // Get the first 8 columns
    var featureValues = rowcol.slice(0, 8).map(parseFloat);

    // Encode the 9th column
    var oceantypes = 5;
    try {
      if (rowcol[9].trim() == "NEAR BAY") {
        featureValues.push(0, 0, 0, 0, 1);
      } else if (rowcol[9].trim() == "INLAND") {
        featureValues.push(0, 0, 0, 1, 0);
      } else if (rowcol[9].trim() == "<1H OCEAN") {
        featureValues.push(0, 0, 1, 0, 0);
      } else if (rowcol[9].trim() == "ISLAND") {
        featureValues.push(0, 1, 0, 0, 0);
      } else if (rowcol[9].trim() == "NEAR OCEAN") {
        featureValues.push(1, 0, 0, 0, 0);
      } else {
        throw new Error();
      }
    } catch(error) {
      continue;
    }
    
    // Push to features and labels
    var labelValue = parseFloat(rowcol[8]);
    if (!featureValues.includes(NaN) && !isNaN(labelValue)) {
      features.push(featureValues);
      labels.push(labelValue);
    }
  }
  
  // Split data into train and test sets
  var trainNumber = Math.floor(features.length * splitpercent);
  trainLabels = tf.tensor1d(labels.slice(0, trainNumber));
  testLabels = tf.tensor1d(labels.slice(trainNumber));
  trainFeatures = tf.tensor2d(features.slice(0, trainNumber));
  testFeatures = tf.tensor2d(features.slice(trainNumber));

  // Prepare the model
  var model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 26,
      inputShape: [8 + oceantypes],
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 100,
      activation: "relu",
    })
  );
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));

  // Compile the model
  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
    metrics: ["mse"],
  });

  // Train the model
  var history = await model.fit(trainFeatures, trainLabels, {
    epochs: epochvalue,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.getElementById("train-status").innerText =
          "\n • EPOCH: " +
          (epoch + 1) +
          "/" +
          epochvalue +
          "\n\n • RMSE (Training): " +
          Math.round(Math.sqrt(logs.mse.toFixed(4)));
      },
    },
  });

  return model;
}

// Run the model and enable other options when complete
async function confirmTraining(type) {
  epochvalue = type;

  document.getElementById("train-img").style.display = "inline";
  document.getElementById("train-options").style.display = "none";

  regModel = await trainModel();
  regReady = true;

  // Wait until the model is trained and then update the UI
  document.getElementById("input-r").style.display = "inline";
  document.getElementById("result-panel").style.display = "inline";
  document.getElementById("train-tick").style.display = "inline";
  document.getElementById("train-img").style.display = "none";
  document.getElementById("confirm").style.display = "inline";
  plotChart();

  // Evaluate the model on test data
  var testEval = regModel.evaluate(testFeatures, testLabels);

  document.getElementById("train-status").innerText =
    document.getElementById("train-status").innerText +
    "\n\n • RMSE (Testing): " +
    Math.round(Math.sqrt(testEval[1].dataSync()));
}

// [ P E F O R M ]----------------------------------------------------------------------------------------------------

document.getElementById("calculate-r").addEventListener("click", async () => {
  regPredict();
});

document.getElementById("randomize").addEventListener("click", async () => {
  randomizer();
});

document.getElementById("compare").addEventListener("click", async () => {
  comparer();
});

document.getElementById("regForm").addEventListener("input", async () => {
  formChanged();
});

async function regPredict() {
  if (regReady) {
    document.getElementById("perform-tick").style.display = "inline";
    document.getElementById("regPrediction").style.color = "#16C028";

    var oceanvalue = document.getElementById("ocean-value").value;
    var predictforthese = [
      document.getElementById("input_long").value,
      document.getElementById("input_lat").value,
      document.getElementById("input_hma").value,
      document.getElementById("input_rooms").value,
      document.getElementById("input_bedrooms").value,
      document.getElementById("input_pop").value,
      document.getElementById("input_house").value,
      document.getElementById("input_medinc").value,
    ];

    if (oceanvalue == 1) {
      predictforthese.push(0, 0, 0, 0, 1);
    } else if (oceanvalue == 2) {
      predictforthese.push(0, 0, 0, 1, 0);
    } else if (oceanvalue == 3) {
      predictforthese.push(0, 0, 1, 0, 0);
    } else if (oceanvalue == 4) {
      predictforthese.push(0, 1, 0, 0, 0);
    } else if (oceanvalue == 5) {
      predictforthese.push(1, 0, 0, 0, 0);
    }

    predictforthese = predictforthese.map(parseFloat);

    var testerFeatures = tf.tensor2d([predictforthese]);
    var predictions = regModel.predict(testerFeatures);

    var output = Math.round(predictions.dataSync());

    if (output) {
      document.getElementById("regPrediction").value = "$ " + output;
    } else {
      document.getElementById("regPrediction").value = "";
    }
  } else {
    console.log("Model not ready");
  }
}

// Used to generate random values based on a predefined range
async function randomizer() {
  formChanged();
  document.getElementById("input_long").value = Math.floor(
    Math.random() * (-114 - -125 + 1) + -125
  );
  document.getElementById("input_lat").value = Math.floor(
    Math.random() * (42 - 33 + 1) + 33
  );
  document.getElementById("input_hma").value = Math.floor(
    Math.random() * (52 - 1 + 1) + 1
  );
  document.getElementById("input_rooms").value = Math.floor(
    Math.random() * (39320 - 2 + 1) + 2
  );
  document.getElementById("input_bedrooms").value = Math.floor(
    Math.random() * (6445 - 1 + 1) + 1
  );
  document.getElementById("input_pop").value = Math.floor(
    Math.random() * (35682 - 3 + 1) + 3
  );
  document.getElementById("input_house").value = Math.floor(
    Math.random() * (6082 - 1 + 1) + 1
  );
  document.getElementById("input_medinc").value = Math.floor(
    Math.random() * (15 - 1 + 1) + 1
  );
  document.getElementById("ocean-value").value = Math.floor(
    Math.random() * (5 - 1 + 1) + 1
  );
}

// Used to compare predications with the actual value for a given row
async function comparer() {
  var okayrow = false;
  var predictforthese;
  var oceanvalue;
  var labelValue;

  // If the row contains no NaN values
  while (!okayrow) {
    var i = Math.floor(Math.random() * rows.length);

    var rowcol = rows[i].split(",");
    predictforthese = rowcol.slice(0, 8).map(parseFloat);

    if (rowcol[9].trim() == "NEAR BAY") {
      predictforthese.push(0, 0, 0, 0, 1);
      oceanvalue = 1;
    } else if (rowcol[9].trim() == "INLAND") {
      predictforthese.push(0, 0, 0, 1, 0);
      oceanvalue = 2;
    } else if (rowcol[9].trim() == "<1H OCEAN") {
      predictforthese.push(0, 0, 1, 0, 0);
      oceanvalue = 3;
    } else if (rowcol[9].trim() == "ISLAND") {
      predictforthese.push(0, 1, 0, 0, 0);
      oceanvalue = 4;
    } else if (rowcol[9].trim() == "NEAR OCEAN") {
      predictforthese.push(1, 0, 0, 0, 0);
      oceanvalue = 5;
    } else {
      oceanvalue = null;
    }

    labelValue = parseFloat(rowcol[8]);

    if (
      !predictforthese.includes(NaN) &&
      !isNaN(labelValue) & !isNaN(oceanvalue)
    ) {
      okayrow = true;
    }
  }

  // Update the values in the UI before getting the prediction
  document.getElementById("input_long").value = predictforthese[0];
  document.getElementById("input_lat").value = predictforthese[1];
  document.getElementById("input_hma").value = predictforthese[2];
  document.getElementById("input_rooms").value = predictforthese[3];
  document.getElementById("input_bedrooms").value = predictforthese[4];
  document.getElementById("input_pop").value = predictforthese[5];
  document.getElementById("input_house").value = predictforthese[6];
  document.getElementById("input_medinc").value = predictforthese[7];
  document.getElementById("ocean-value").value = oceanvalue;

  var testerFeatures = tf.tensor2d([predictforthese]);
  var predictions = regModel.predict(testerFeatures);
  var output = Math.round(predictions.dataSync());

  // Update the values in the Result UI after getting the prediction
  document.getElementById("perform-tick").style.display = "inline";
  document.getElementById("regPrediction").style.color = "#16C028";
  document.getElementById("regPrediction").value = "$ " + output;

  document.getElementById("regComparison").value = "$ " + labelValue;
}

// If the form was changed, update the UI
function formChanged() {
  document.getElementById("regComparison").value = "";
  document.getElementById("perform-tick").style.display = "none";
  document.getElementById("regPrediction").style.color = "gray";
}

// [ R E S U L T ]----------------------------------------------------------------------------------------------------

// Intializing an empty chart
var chart = new Chart("TestChart", {type: "scatter", data: {}});

// Generate a chart to show learning based on test data
async function plotChart() {
  
  // Delete the previous chart if any
  chart.destroy();
  
  var xyValues = [];

  if (regReady) {
    for (var i = 0; i < testFeatures.shape[0]; i++) {
      var F = testFeatures.arraySync()[i].slice(0, 13);
      var L = testLabels.dataSync()[i];

      var testerFeatures = tf.tensor2d([F], [1, 13]);
      var predictions = regModel.predict(testerFeatures);
      var output = Math.round(predictions.dataSync());

      xyValues.push({ x: output, y: L });
    }
  } else {
    console.log("Model not ready");
  }

  chart = new Chart("TestChart", {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Median house value - Predicted vs Actual",
          pointRadius: 4,
          pointBackgroundColor: "#58bccc",
          color: "lightgray",
          data: xyValues,
        },
      ],
    },
  });
}
