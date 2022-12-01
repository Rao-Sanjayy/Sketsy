const IMAGE_SIZE = 784;
var CLASSES = ['apple','banana','basket','book','bucket','butterfly','cake','candle','car','circle','eye',
                'face','fan','fish','flower','grapes','hand','ladder','snake','square'];
const k = 3; // for top k predictions
let model;
let cnv;

// variables required for the scoring and prompt system
score = 0;
random_no = Math.floor((Math.random()*CLASSES.length)+1); // generates a random index of the class
Element_of_array = CLASSES[random_no]; // random class
select('#sketch_to_be_drawn').html('Sketch to be Drawn: ', Element_of_array);
document.getElementById("sketch_to_be_drawn").innerHTML = "Riya Yoyal "+ Element_of_array;
console.log(Element_of_array);

async function loadMyModel() {
  model = await tf.loadLayersModel('model/model.json'); 
  model.summary();
}

function setup() {
  loadMyModel();

  cnv = createCanvas(400, 400);
  background(255);
  cnv.parent('canvasContainer');

  let guessButton = select('#guess');
  // guessButton.mousePressed(guess);
  cnv.mouseReleased(guess);
  
  let clearButton = select('#clear');
  clearButton.mousePressed(() => {
    background(255);
    select('#res').html('');
  });
}

function guess() {
  // Get input image from the canvas
  const inputs = getInputImage();
  console.log(tf.tensor([inputs]));
  
  // Predict
  let guess = model.predict(tf.tensor([inputs])).squeeze();
  
  // Format res to an array
  const rawProb = Array.from(guess.dataSync());
  
  // Get top K res with index and probability
  const rawProbWIndex = rawProb.map((probability, index) => {
    return {
      index,
      probability
    }
  });

  const sortProb = rawProbWIndex.sort((a, b) => b.probability - a.probability);
  const topKClassWIndex = sortProb.slice(0, k);
  const topKRes = topKClassWIndex.map(i => `<br>${CLASSES[i.index]} (${(i.probability.toFixed(2) * 100)}%)`);
  console.log(topKRes.toString());
  select('#res').html(`I see: ${topKRes.toString()}`);
}

function getInputImage() {
  let inputs = [];
  // P5 function, get image from the canvas
  let img = get();
  img.resize(28, 28);
  img.loadPixels();

  // Group data into [[[i00] [i01], [i02], [i03], ..., [i027]], .... [[i270], [i271], ... , [i2727]]]]
  let oneRow = [];
  for (let i = 0; i < IMAGE_SIZE; i++) {
    let bright = img.pixels[i * 4];
    let onePix = [parseFloat((255 - bright) / 255)];
    oneRow.push(onePix);
    if (oneRow.length === 28) {
      inputs.push(oneRow);
      oneRow = [];
    }
  }

  return inputs;
}

function draw() {
  strokeWeight(15);
  stroke(5);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}