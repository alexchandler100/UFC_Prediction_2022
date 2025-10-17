theta = {};
intercept = {};
fighter_data = {}
ufcfightscrap = {}
vegas_odds = {}
prediction_history = {}
card_info = {}

$.getJSON('src/content/data/external/card_info.json', function (data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function (i, f) {
    card_info[i] = f
  });
});

$.getJSON('src/content/data/external/theta.json', function (data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function (i, f) {
    theta[i] = f.toFixed(2)
  });
});

$.getJSON('src/content/data/external/intercept.json', function (data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function (i, f) {
    intercept[i] = f.toFixed(2)
  });
});

$(function () { // building object fighter_data from fighter_data.json file
  $.getJSON('src/content/data/external/fighter_stats.json', function (data) {//for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
    $.each(data, function (i, f) {//create entry in local object
      const select = document.getElementById('fighters')
      select.insertAdjacentHTML('beforeend', `<option value="${i}">${i}</option>`)
      fighter_data[i] = f
    });
  });
});

$(function () { // building object ufcfightscrap from ufcfightscrap.json file
  $.getJSON('src/content/data/external/ufc_fight_data_for_website.json', function (data) {//for each input (i,f), i is the key (a number) and f is the value (all the data of the fight)
    $.each(data, function (i, f) {
      ufcfightscrap[i] = f
    });
  });
});

$(function () { // building object vegas_odds from vegas_odds.json file
  $.getJSON('src/content/data/external/vegas_odds.json', function (data) {//for each input (i,f), i is the key a column name like fighter name and f is the value (an object with keys being integers and values being strings (odds or names))
    $.each(data, function (i, f) {
      vegas_odds[i] = f
    });
  });
});

const years = document.getElementById('years')
for (let i = 0; i < 30; i++) {
  year = 2022 - i
  years.insertAdjacentHTML('beforeend', `
<option value="${year}">${year}</option>
`)
}

//set initial table values
setTimeout(() => {
  ufc_wins_list = []
  console.log(vegas_odds)
  for (const fight in ufcfightscrap) {
    if (ufcfightscrap[fight]['result'] == "W") {
      let fighter = ufcfightscrap[fight]['fighter']
      let opponent = ufcfightscrap[fight]['opponent']
      let date = ufcfightscrap[fight]['date']
      ufc_wins_list.push([fighter, opponent, date])
    }
  }

  //giving prediction_history correct keys
  $(function () {
    //var people = [];
    $.getJSON('src/content/data/external/prediction_history.json', function (data) {
      //for each input (i,f), i is the key a column name like fighter name and f is the value (an object with keys being integers and values being strings (odds or names))
      $.each(data, function (i, f) {
        //create entry in local object
        prediction_history[i] = f
      });
    });
  });
}, 500) // originally 250

function americanToImpliedProb(odds) {
  if (odds > 0) {
    return 100 / (odds + 100);
  } else {
    return -odds / (-odds + 100);
  }
}

function betPayout(amount, odds) {
  if (odds > 0) {
    return amount * (odds / 100);
  } else {
    return amount * (100 / -odds) ;
  }
}

function computeKelly() {
  const fighterPred = parseFloat(document.getElementById('fighterPred').value);
  const opponentPred = parseFloat(document.getElementById('opponentPred').value);
  const fighterVegas = parseFloat(document.getElementById('fighterVegas').value);
  const opponentVegas = parseFloat(document.getElementById('opponentVegas').value);

  if (
    isNaN(fighterPred) || isNaN(opponentPred) ||
    isNaN(fighterVegas) || isNaN(opponentVegas)
  ) {
    alert("Please enter all values correctly.");
    return;
  }

  const pFighter = americanToImpliedProb(fighterPred);
  const pOpponent = americanToImpliedProb(opponentPred);

  const qFighter = 1 - pFighter;
  const qOpponent = 1 - pOpponent;

  if (fighterVegas > 0){
    bFighter = fighterVegas / 100;
  } else {
    bFighter = -100 / fighterVegas;
  }
  if (opponentVegas > 0){
    bOpponent = opponentVegas / 100;
  } else {
    bOpponent = -100 / opponentVegas;
  }

  const kellyFighter = (bFighter * pFighter - qFighter) / bFighter;
  const kellyOpponent = (bOpponent * pOpponent - qOpponent) / bOpponent;

  document.getElementById("fighter kelly %").textContent =
    kellyFighter > 0 ? (kellyFighter * 100).toFixed(2) + '%' : '0%';

  document.getElementById("opponent kelly %").textContent =
    kellyOpponent > 0 ? (kellyOpponent * 100).toFixed(2) + '%' : '0%';
}

const levenshteinDistance = (str1 = '', str2 = '') => {
  const track = Array(str2.length + 1).fill(null).map(() =>
    Array(str1.length + 1).fill(null));
  for (let i = 0; i <= str1.length; i += 1) {
    track[0][i] = i;
  }
  for (let j = 0; j <= str2.length; j += 1) {
    track[j][0] = j;
  }
  for (let j = 1; j <= str2.length; j += 1) {
    for (let i = 1; i <= str1.length; i += 1) {
      const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
      track[j][i] = Math.min(
        track[j][i - 1] + 1, // deletion
        track[j - 1][i] + 1, // insertion
        track[j - 1][i - 1] + indicator, // substitution
      );
    }
  }
  return track[str2.length][str1.length];
};

function same_name(str1, str2) {
  str1 = str1.toLowerCase().replace("st.", 'saint').replace(" st ", ' saint ').replace(".", '').replace("-", ' ')
  str2 = str2.toLowerCase().replace("st.", 'saint').replace(" st ", ' saint ').replace(".", '').replace("-", ' ')
  let str1List = str1.split(" ")
  let str1Set = new Set(str1List)
  let str2List = str2.split(" ")
  let str2Set = new Set(str2List)
  if (str1 === str2) {
    return true
  } else if (eqSet(str1Set, str2Set)) {
    return true
  } else if (levenshteinDistance(str1, str2) < 3) {
    return true
  } else {
    return false
  }
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max) + 1;
}

function checkFileExist(urlToFile) {
  var xhr = new XMLHttpRequest();
  xhr.open('HEAD', urlToFile, false);
  xhr.send();

  if (xhr.status == "404") {
    return false;
  } else {
    return true;
  }
}

let picIndex = 0;
//set picIndex to be a random number between 0 and 4
picIndex = getRandomInt(4);

function selectFighter(id) { // id is 'rc' or 'bc' for red corner or blue corner
  var selectElement = document.querySelector('#' + id);
  var fighterNameText = selectElement.value;
  //document.querySelector('.' + out).textContent = output;
  picIndex += 1
  let j = (picIndex) % 4 + 1
  var name_encoded = encodeURIComponent(fighterNameText)
  var name_decoded = decodeURIComponent(name_encoded)
  name_decoded = decodeURIComponent(name_decoded)
  name_decoded = name_decoded.replace(new RegExp(' ', 'g'), '');

  // set the path to check if gif file exists (otherwise use pictures)
  if (checkFileExist("src/content/gifs/postCNNGIFs/" + name_decoded + ".gif")) {
    document.getElementById(`${id}FighterPic`).src = "src/content/gifs/postCNNGIFs/" + name_decoded + ".gif" //sets the image
  } else if (checkFileExist("src/content/images2/" + j + name_decoded + ".jpg")) {
    document.getElementById(`${id}FighterPic`).src = "src/content/images2/" + j + name_decoded + ".jpg" //sets the image
  } else {
    document.getElementById(`${id}FighterPic`).src = "src/content/images/" + j + name_decoded + ".jpg" //sets the image
  }
  populateTaleOfTheTape(fighterNameText, id)
  populateLast5Fights(fighterNameText, id)
}

// function selectDate(id) {
//   selectMonth = document.querySelector(`selectMonth_${id}`);
//   output = selectMonth.value;
//   document.querySelector(`.selectMonth_${id}`).textContent = output;
//   selectYear = document.querySelector(`.year_${id}`);
//   output2 = selectYear.value;
//   document.querySelector(`.selectYear_${id}`).textContent = output2;
// }

function selectFighterAndDate(id) {
  selectFighter(id)
  // selectDate(id)
}

function fighter_age(fighter, yearSelected) {
  //finding the correct name (could be entered differently in the fighter_data dataset)
  let fighterName = ''
  for (const name in fighter_data) {
    if (same_name(fighter, name)) {
      fighterName = name;
      break;
    }
  }
  let yearBorn = fighter_data[fighterName]['dob'].slice(-4)
  return parseInt(yearSelected) - parseInt(yearBorn)
}

function fighter_reach(fighter, yearSelected) {
  let reach = fighter_data[fighter]['reach'].slice(0, -1) //this removes the last character "
  return parseInt(reach)
}

function l5y_wins(fighter, year) {
  wins = 0
  for (const fight in ufcfightscrap) {
    let name = ufcfightscrap[fight]['fighter']
    let yearDiff = parseInt(year) - ufcfightscrap[fight]['date'].slice(-4)
    let result = ufcfightscrap[fight]['result']
    if (yearDiff >= 6) {
      return wins
      break;
    }
    if (same_name(name, fighter) && yearDiff < 6 && yearDiff >= 0 && result == 'W') {
      wins += 1
    }
  }
  return wins
}


function l2y_wins(fighter, year) {
  wins = 0
  for (const fight in ufcfightscrap) {
    let name = ufcfightscrap[fight]['fighter']
    let yearDiff = parseInt(year) - ufcfightscrap[fight]['date'].slice(-4)
    let result = ufcfightscrap[fight]['result']
    if (yearDiff >= 3) {
      return wins
      break;
    }
    if (same_name(name, fighter) && yearDiff < 3 && yearDiff >= 0 && result == 'W') {
      wins += 1
    }
  }
  return wins
}

function l5y_ko_losses(fighter, year) {
  ko_losses = 0
  for (const fight in ufcfightscrap) {
    let name = ufcfightscrap[fight]['fighter']
    let yearDiff = parseInt(year) - ufcfightscrap[fight]['date'].slice(-4)
    let result = ufcfightscrap[fight]['result']
    let method = ufcfightscrap[fight]['method']
    if (yearDiff >= 6) {
      return ko_losses
      break;
    }
    if (same_name(name, fighter) && yearDiff < 6 && yearDiff >= 0 && result == 'L' && method == "KO/TKO") {
      ko_losses += 1
    }
  }
  return ko_losses
}

function l5y_sub_wins(fighter, year) {
  sub_wins = 0
  for (const fight in ufcfightscrap) {
    let name = ufcfightscrap[fight]['fighter']
    let yearDiff = parseInt(year) - ufcfightscrap[fight]['date'].slice(-4)
    let result = ufcfightscrap[fight]['result']
    let method = ufcfightscrap[fight]['method']
    if (yearDiff >= 6) {
      return sub_wins
      break;
    }
    if (same_name(name, fighter) && yearDiff < 6 && yearDiff >= 0 && result == 'W' && method == "SUB") {
      sub_wins += 1
    }
  }
  return sub_wins
}

function l5y_losses(fighter, year) {
  losses = 0
  for (const fight in ufcfightscrap) {
    let name = ufcfightscrap[fight]['fighter']
    let yearDiff = parseInt(year) - ufcfightscrap[fight]['date'].slice(-4)
    let result = ufcfightscrap[fight]['result']
    if (yearDiff >= 6) {
      return losses;
      break;
    }
    if (same_name(name, fighter) && yearDiff < 6 && yearDiff >= 0 && result == 'L') {
      losses += 1
    }
  }
  return losses
}

function avg_count(stat, fighter, inf_abs, year) { // e.g. avg_count('total_strikes_landed',fighter1,'abs',day1)
  let summ = 0
  let time_in_octagon = 0
  let person;
  if (inf_abs == 'inf') {
    person = 'fighter'
  } else {
    person = 'opponent'
  }
  for (const fight in ufcfightscrap) {
    let name = ufcfightscrap[fight][person]
    let yearDiff = parseInt(year) - ufcfightscrap[fight]['date'].slice(-4)
    if (same_name(name, fighter) && yearDiff >= 0) {
      summ += parseInt(ufcfightscrap[fight][stat])
      let round = parseInt(ufcfightscrap[fight]['round'])
      let minutes = parseInt(ufcfightscrap[fight]['time'][0])
      let seconds = parseInt(ufcfightscrap[fight]['time'].slice(2))
      time_in_octagon += (round - 1) * 5 + minutes + seconds / 60
    }
  }
  return summ / time_in_octagon
}

function onlyUnique(value, index, self) {
  return self.indexOf(value) === index;
}

function wins_wins(fighter, year, years) {
  let relevant_fights = []
  for (let i = 0; i < ufc_wins_list.length; i++) {
    let yearDiff = parseInt(year) - ufc_wins_list[i][2].slice(-4)
    if (yearDiff > years) {
      break
    } else {
      relevant_fights.push(ufc_wins_list[i])
    }
  }
  let fighter_wins = []
  for (let i = 0; i < relevant_fights.length; i++) {
    if (same_name(relevant_fights[i][0], fighter)) {
      fighter_wins.push(relevant_fights[i][1])
    }
  }
  let fighter_wins_wins = []
  for (let i = 0; i < relevant_fights.length; i++) {
    //the same_name function should be used here but that requires refactoring
    if (fighter_wins.includes(relevant_fights[i][0])) {
      fighter_wins_wins.push(relevant_fights[i][1])
    }
  }
  let relevant_wins = fighter_wins.concat(fighter_wins_wins);
  relevant_wins = relevant_wins.filter(onlyUnique);
  return relevant_wins
}

function losses_losses(fighter, year, years) {
  let relevant_fights = []
  for (let i = 0; i < ufc_wins_list.length; i++) {
    let yearDiff = parseInt(year) - ufc_wins_list[i][2].slice(-4)
    if (yearDiff > years) {
      break
    } else {
      relevant_fights.push(ufc_wins_list[i])
    }
  }
  let fighter_losses = []
  for (let i = 0; i < relevant_fights.length; i++) {
    if (same_name(relevant_fights[i][1], fighter)) {
      fighter_losses.push(relevant_fights[i][0])
    }
  }
  let fighter_losses_losses = []
  for (let i = 0; i < relevant_fights.length; i++) {
    //same_name function should be used here but that requires refactoring (i.e. make a custom function to check if name is in list up to small changes)
    if (fighter_losses.includes(relevant_fights[i][1])) {
      fighter_losses_losses.push(relevant_fights[i][0])
    }
  }
  let relevant_losses = fighter_losses.concat(fighter_losses_losses);
  relevant_losses = relevant_losses.filter(onlyUnique);
  return relevant_losses
}

//this does not incorporate year for both fighters correctly...
function fight_math(fighter, opponent, year, years) {
  let relevant_fights = []
  for (let i = 0; i < ufc_wins_list.length; i++) {
    let yearDiff = parseInt(year) - ufc_wins_list[i][2].slice(-4)
    if (yearDiff > years) {
      break
    } else {
      relevant_fights.push(ufc_wins_list[i])
    }
  }
  let relevant_wins = wins_wins(fighter, year, years)
  relevant_wins.push(fighter)
  let fight_math_wins = []
  for (let i = 0; i < relevant_fights.length; i++) {
    if (relevant_wins.includes(relevant_fights[i][0]) && relevant_fights[i][1] == opponent) {
      fight_math_wins.push(relevant_fights[i])
    }
  }
  return fight_math_wins.length
}

function fight_math_diff(fighter, opponent, year1, year2, years) {
  return fight_math(fighter, opponent, year1, years) - fight_math(opponent, fighter, year2, years)
}

function fighter_score(fighter, year, years) {
  let relevant_wins = wins_wins(fighter, year, years)
  let relevant_losses = losses_losses(fighter, year, years)
  return relevant_wins.length - relevant_losses.length
}

function fighter_score_diff(fighter, opponent, year1, year2, years) {
  return fighter_score(fighter, year1, years) - fighter_score(opponent, year2, years)
}

function avg_count_diff(stat, fighter, opponent, inf_abs, year) {
  if (isNaN(avg_count(stat, fighter, inf_abs, year)) || isNaN(avg_count(stat, opponent, inf_abs, year))) {
    return 0
  }
  return avg_count(stat, fighter, inf_abs, year) - avg_count(stat, opponent, inf_abs, year)
}

//the input to this function looks like strings ("Mike Perry", "Conor McGregor", "June", "2022", "June", "2022")
function predictionTupleAbsolute(fighter1, fighter2, month1, year1, month2, year2) {
  let result;
  let guy1 = fighter1
  let guy2 = fighter2
  let mon1 = month1
  let mon2 = month2
  let yr1 = year1
  let yr2 = year2
  let fighter_score_diff_4 = fighter_score_diff(guy1, guy2, yr1, yr2, 4).toFixed(2)
  let fighter_score_diff_9 = fighter_score_diff(guy1, guy2, yr1, yr2, 9).toFixed(2)
  let fighter_score_diff_15 = fighter_score_diff(guy1, guy2, yr1, yr2, 15).toFixed(2)
  let fight_math_1 = fight_math_diff(guy1, guy2, yr1, yr2, 1).toFixed(2)
  let fight_math_6 = fight_math_diff(guy1, guy2, yr1, yr2, 6).toFixed(2)
  let l5y_sub_wins_diff = l5y_sub_wins(guy1, yr1).toFixed(2) - l5y_sub_wins(guy2, yr2).toFixed(2)
  let l5y_losses_diff = l5y_losses(guy1, yr1).toFixed(2) - l5y_losses(guy2, yr2).toFixed(2)
  let l5y_ko_losses_diff = l5y_ko_losses(guy1, yr1).toFixed(2) - l5y_ko_losses(guy2, yr2).toFixed(2)
  let age_diff = fighter_age(guy1, yr1).toFixed(2) - fighter_age(guy2, yr2).toFixed(2)
  let av_total_strikes_diff = avg_count_diff('total_strikes_landed', guy1, guy2, 'abs', yr1).toFixed(2)
  let av_abs_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'abs', yr1).toFixed(2)
  let av_inf_gr_strikes = avg_count_diff('ground_strikes_landed', guy1, guy2, 'inf', yr1).toFixed(2)
  let av_tk_atmps_diff = avg_count_diff('takedowns_attempts', guy1, guy2, 'inf', yr1).toFixed(2)
  let av_inf_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'inf', yr1).toFixed(2)
  result = [fighter_score_diff_4, fighter_score_diff_9, fighter_score_diff_15, fight_math_1, fight_math_6,
    l5y_sub_wins_diff, l5y_losses_diff, l5y_ko_losses_diff, age_diff, av_total_strikes_diff, av_abs_head_strikes_diff,
    av_inf_gr_strikes, av_tk_atmps_diff, av_inf_head_strikes_diff
  ]
  return result;
}

//It might make sense to scale the output by something between 1 and 2
function presigmoid_valueAbsolute(fighter1, fighter2, month1, year1, month2, year2) {
  let value = 0
  tup = predictionTupleAbsolute(fighter1, fighter2, month1, year1, month2, year2);
  for (let i = 0; i < tup.length; i++) {
    value += tup[i] * theta[i]
  }
  //return value + intercept[0]
  //return parseFloat(value) + parseFloat(intercept[0])
  return parseFloat(value)
}

function probabilityAbsolute(fighter1, fighter2, month1, year1, month2, year2) {
  return sigmoid(presigmoid_valueAbsolute(fighter1, fighter2, month1, year1, month2, year2))
}

function betting_oddsAbsolute(fighter1, fighter2, month1, year1, month2, year2) {
  p = probabilityAbsolute(fighter1, fighter2, month1, year1, month2, year2)
  let fighterOdds;
  let opponentOdds;
  if (p < .5) {
    fighterOdds = Math.round(100 / p - 100)
    opponentOdds = Math.round(1 / (1 / (1 - p) - 1) * 100)
    return [`+${fighterOdds}`, `-${opponentOdds}`]
  } else if (p >= .5) {
    fighterOdds = Math.round(1 / (1 / p - 1) * 100)
    opponentOdds = Math.round(100 / (1 - p) - 100)
    return [`-${fighterOdds}`, `+${opponentOdds}`]
  }
}

//this takes as input certain html locations holding this data... not strings
function predictionTuple(fighter1, fighter2, month1, year1, month2, year2) {
  let result;
  // Note # selects by id and . selects by class
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  mon1 = document.querySelector('#' + month1).value;
  mon2 = document.querySelector('#' + month2).value;
  yr1 = document.querySelector('#' + year1).value;
  yr2 = document.querySelector('#' + year2).value;
  let fighter_score_diff_4 = fighter_score_diff(guy1, guy2, yr1, yr2, 4).toFixed(2)
  let fighter_score_diff_9 = fighter_score_diff(guy1, guy2, yr1, yr2, 9).toFixed(2)
  let fighter_score_diff_15 = fighter_score_diff(guy1, guy2, yr1, yr2, 15).toFixed(2)
  let fight_math_1 = fight_math_diff(guy1, guy2, yr1, yr2, 1).toFixed(2)
  let fight_math_6 = fight_math_diff(guy1, guy2, yr1, yr2, 6).toFixed(2)
  let l5y_sub_wins_diff = l5y_sub_wins(guy1, yr1).toFixed(2) - l5y_sub_wins(guy2, yr2).toFixed(2)
  let l5y_losses_diff = l5y_losses(guy1, yr1).toFixed(2) - l5y_losses(guy2, yr2).toFixed(2)
  let l5y_ko_losses_diff = l5y_ko_losses(guy1, yr1).toFixed(2) - l5y_ko_losses(guy2, yr2).toFixed(2)
  let age_diff = fighter_age(guy1, yr1).toFixed(2) - fighter_age(guy2, yr2).toFixed(2)
  let av_total_strikes_diff = avg_count_diff('total_strikes_landed', guy1, guy2, 'abs', yr1).toFixed(2)
  let av_abs_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'abs', yr1).toFixed(2)
  let av_inf_gr_strikes = avg_count_diff('ground_strikes_landed', guy1, guy2, 'inf', yr1).toFixed(2)
  let av_tk_atmps_diff = avg_count_diff('takedowns_attempts', guy1, guy2, 'inf', yr1).toFixed(2)
  let av_inf_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'inf', yr1).toFixed(2)
  result = [fighter_score_diff_4, fighter_score_diff_9, fighter_score_diff_15, fight_math_1, fight_math_6,
    l5y_sub_wins_diff, l5y_losses_diff, l5y_ko_losses_diff, age_diff, av_total_strikes_diff, av_abs_head_strikes_diff,
    av_inf_gr_strikes, av_tk_atmps_diff, av_inf_head_strikes_diff
  ]
  return result;
}

//It might make sense to scale the output by something between 1 and 2 to adjust probabilities
function presigmoid_value(fighter1, fighter2, month1, year1, month2, year2) {
  let value = 0
  tup = predictionTuple(fighter1, fighter2, month1, year1, month2, year2);
  for (let i = 0; i < tup.length; i++) {
    value += tup[i] * theta[i]
  }
  //return value + intercept[0]
  //return parseFloat(value) + parseFloat(intercept[0])
  return parseFloat(value)
}


function sigmoid(x) {
  return 1 / (1 + Math.exp(-x))
}

function probability(fighter1, fighter2, month1, year1, month2, year2) {
  return sigmoid(presigmoid_value(fighter1, fighter2, month1, year1, month2, year2))
}

function betting_odds(fighter1, fighter2, month1, year1, month2, year2) {
  p = probability(fighter1, fighter2, month1, year1, month2, year2)
  let fighterOdds;
  let opponentOdds;
  if (p < .5) {
    fighterOdds = Math.round(100 / p - 100)
    opponentOdds = Math.round(1 / (1 / (1 - p) - 1) * 100)
    return [`+${fighterOdds}`, `-${opponentOdds}`]
  } else if (p >= .5) {
    fighterOdds = Math.round(1 / (1 / p - 1) * 100)
    opponentOdds = Math.round(100 / (1 - p) - 100)
    return [`-${fighterOdds}`, `+${opponentOdds}`]
  }
}

function get_vegas_odds(fighter1, fighter2, month1, year1, month2, year2) {
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  let f_names = Object.values(vegas_odds['fighter name'])
  let o_names = Object.values(vegas_odds['opponent name'])
  let vegas_odds_dict = {}
  for (let i = 0; i < f_names.length; i++) {
    if ((same_name(f_names[i], guy1) && same_name(o_names[i], guy2)) || (same_name(f_names[i], guy2) && same_name(o_names[i], guy1))) {
      for (let j = 0; j < Object.keys(vegas_odds).length; j++) {
        let key = Object.keys(vegas_odds)[j]
        let value = vegas_odds[key][i]
        vegas_odds_dict[key] = value
      }
    }
  }
  return vegas_odds_dict
}

function predict(fighter1, fighter2, month1, year1, month2, year2) {
  vegas_odds_dict = get_vegas_odds(fighter1, fighter2, month1, year1, month2, year2)
  let tup = predictionTuple(fighter1, fighter2, month1, year1, month2, year2);
  let value = presigmoid_value(fighter1, fighter2, month1, year1, month2, year2)
  let prob = sigmoid(value)
  let winner;
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  if (value >= 0) {
    winner = guy1
  } else {
    winner = guy2
  }
  let abs_value = (Math.abs(prob - .5))

  let resulting_text;
  if (abs_value >= 0 && abs_value <= .04) {
    resulting_text = winner + " wins a little over 5 out of 10 times."
  } else if (abs_value >= .04 && abs_value <= .15) {
    resulting_text = (winner + " wins 6 out of 10 times.")
  } else if (abs_value >= .15 && abs_value <= .25) {
    resulting_text = (winner + " wins 7 out of 10 times.")
  } else if (abs_value >= .25 && abs_value <= .4) {
    resulting_text = (winner + " wins 9 out of 10 times.")
  } else if (abs_value >= .4) {
    resulting_text = (winner + " wins 10 out of 10 times.")
  } else {
    console.log(`something is wrong with the probability`)
  }
  if (guy1 != guy2 && tup[0] == 0.0 && tup[1] == 0.0 && tup[2] == 0.0 && tup[3] == 0.0 && tup[4] == 0.0) {
    resulting_text = 'Internal issue encountered. Please refresh page and try again.'
  }
  document.querySelector('.fightoutcome').textContent = resulting_text
  odds = betting_odds(fighter1, fighter2, month1, year1, month2, year2)

  //populate odds
  var myTab;
  myTab = document.getElementById("tableoutcome");

  // LOOP THROUGH EACH ROW OF THE TABLE AFTER HEADER.
  myTab.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
  myTab.rows.item(1).cells.item(0).style.backgroundColor = "#323232";
  myTab.rows.item(1).cells.item(1).style.backgroundColor = "#323232";
  myTab.rows.item(1).cells.item(2).style.backgroundColor = "#323232";

  myTab.rows.item(1).cells.item(1).innerHTML = `${guy1}: <span style="color:#00FF00";>${odds[0]}</span>`;
  myTab.rows.item(1).cells.item(2).innerHTML = `${guy2}: <span style="color:#00FF00";>${odds[1]}</span>`;
}

function populateTaleOfTheTape(fighter, corner) {
  var myTab;
  yr = document.querySelector(`#selectYear_${corner}`).value;
  myTab = document.getElementById(`table_${corner}`);

  myTab.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
  myTab.rows.item(0).cells.item(0).innerHTML = fighter;
  
  // LOOP THROUGH EACH ROW OF THE TABLE AFTER HEADER.
  myTab.rows.item(2).cells.item(0).innerHTML = fighter_data[fighter]['height'];
  myTab.rows.item(2).cells.item(1).innerHTML = fighter_data[fighter]['reach'];
  myTab.rows.item(2).cells.item(2).innerHTML = fighter_age(fighter, yr);
  myTab.rows.item(2).cells.item(3).innerHTML = fighter_data[fighter]['stance'];
}

function eqSet(as, bs) {
  if (as.size !== bs.size) return false;
  for (var a of as)
    if (!bs.has(a)) return false;
  return true;
}

//same_name function should be used here instead of eqSet (because same_name is stronger)
function populateLast5Fights(fighter, corner) {
  var myTab;
  yr = document.querySelector(`#selectYear_${corner}`).value;
  myTab = document.getElementById(`l5ytable_${corner}`);

  for (numb = 2; numb < 7; numb++) { //reset color of rows to white and empty text content
    for (let j = 0; j < 4; j++) {
      myTab.rows.item(numb).cells.item(j).innerHTML = ''
      myTab.rows.item(numb).cells.item(j).style.backgroundColor = "#ffffff";
    }
  }
  let fightNumber = 1
  for (const fight in ufcfightscrap) {
    let result;
    let opponent;
    let method;
    let yearDiff = parseInt(yr) - parseInt(ufcfightscrap[fight]['date'].slice(0,4));
    tableTitleCell = myTab.rows.item(0).cells.item(0);
    tableTitleCell.innerHTML = fighter
    tableTitleCell.style.backgroundColor = "#212121";
    // note I changed to checking set equality of the set {firstName, middleName, lastName} because different orderings are used in different databases
    if (same_name(ufcfightscrap[fight]['fighter'], fighter) && yearDiff >= 0) {

      result = ufcfightscrap[fight]['result']
      fighter = ufcfightscrap[fight]['fighter']
      opponent = ufcfightscrap[fight]['opponent']
      method = ufcfightscrap[fight]['method']
      date = ufcfightscrap[fight]['date']
      fightNumber += 1

      let opponentText = `<span class="clickable">${opponent}</span>`;
      myTab.rows.item(fightNumber).cells.item(0).innerHTML = opponentText

      let item = myTab.rows.item(fightNumber).cells.item(0);
      let clickable2 = item.querySelector('.clickable');
      if (clickable2 != null){
        clickable2.addEventListener('click', function(event) {
          let opponentName = item.innerText;
          // populate the active fighter and opponent names
          document.getElementById(corner).value = opponentName;
          selectFighter(corner)
        })
      }

      myTab.rows.item(fightNumber).cells.item(1).innerHTML = result
      myTab.rows.item(fightNumber).cells.item(2).innerHTML = method
      myTab.rows.item(fightNumber).cells.item(3).innerHTML = date

      if (result == "W") {
        myTab.rows.item(fightNumber).cells.item(1).style.backgroundColor = "#54ff6b";
      } else if (result == "L") {
        myTab.rows.item(fightNumber).cells.item(1).style.backgroundColor = "#ff5454";
      } else {
        myTab.rows.item(fightNumber).cells.item(1).style.backgroundColor = "#b3b3b3";
      }
    }
    if (fightNumber > 5) {
      break
    }
  }
  while (fightNumber < 6) {
    fightNumber += 1
    myTab.rows.item(fightNumber).cells.item(0).style.backgroundColor = "#dedede";
    myTab.rows.item(fightNumber).cells.item(1).style.backgroundColor = "#dedede";
    myTab.rows.item(fightNumber).cells.item(2).style.backgroundColor = "#dedede";
    myTab.rows.item(fightNumber).cells.item(3).style.backgroundColor = "#dedede";
  }
}

//Building upcoming predictions table

setTimeout(() => { //timeout because other data needs to load first (probably better to do with async)
  var upcomingFightsTable = document.getElementById('upcoming')
  for (const i in vegas_odds['fighter name']) {
    fighter = vegas_odds['fighter name'][i]
    opponent = vegas_odds['opponent name'][i]
    fighterOdds = vegas_odds['predicted fighter odds'][i]
    opponentOdds = vegas_odds['predicted opponent odds'][i]
    // grab columns which may or may not be present
    bestFighterBookieCol = vegas_odds['best fighter bookie'] || null; // default to null if not present
    bestOpponentBookieCol = vegas_odds['best opponent bookie'] || null;
    fighterBankrollPercentageCol = vegas_odds['fighter bet bankroll percentage'] || null; // default to null if not present
    opponentBankrollPercentageCol = vegas_odds['opponent bet bankroll percentage'] || null; // default to null if not present

    if (bestFighterBookieCol != null) {
      bestFighterBookie = bestFighterBookieCol[i];
    } else {
      bestFighterBookie = null; // default to null if not present
    }
    // get info on best bookie odds for fighter and opponent
    if (bestOpponentBookieCol != null) {
      bestOpponentBookie = bestOpponentBookieCol[i];
    } else {
      bestOpponentBookie = null; // default to null if not present
    }
    if (bestFighterBookie) {
      // check if `fighter ${bestFighterBookie}` is in vegas_odds keys
      if (!(`fighter ${bestFighterBookie}` in vegas_odds) || !(`opponent ${bestFighterBookie}` in vegas_odds)) {
        console.warn(`Best fighter bookie ${bestFighterBookie} not found in vegas_odds keys for fight between ${fighter} and ${opponent}. Setting odds to null.`);
        bestFighterBookieOddsOnFighter = null;
        bestFighterBookieOddsOnOpponent = null;
      } else {
        bestFighterBookieOddsOnFighter = vegas_odds[`fighter ${bestFighterBookie}`][i];
        bestFighterBookieOddsOnOpponent = vegas_odds[`opponent ${bestFighterBookie}`][i];
      }
    } else {
      bestFighterBookieOddsOnFighter = null; // default to null if not present
      bestFighterBookieOddsOnOpponent = null; // default to null if not present
    }
    if (bestOpponentBookie) {
      bestOpponentBookieOddsOnFighter = vegas_odds[`fighter ${bestOpponentBookie}`][i];
      bestOpponentBookieOddsOnOpponent = vegas_odds[`opponent ${bestOpponentBookie}`][i];
    } else {
      bestOpponentBookieOddsOnFighter = null; // default to null if not present
      bestOpponentBookieOddsOnOpponent = null; // default to null if not present
    }

    // get info on kelly criterion bankroll percentages
    if (fighterBankrollPercentageCol != null) {
      fighterBankrollPercentage = parseFloat(fighterBankrollPercentageCol[i]).toFixed(2);
    } else {
      fighterBankrollPercentage = null; // default to null if not present
    }
    if (opponentBankrollPercentageCol != null) {
      opponentBankrollPercentage = parseFloat(opponentBankrollPercentageCol[i]).toFixed(2);
    } else {
      opponentBankrollPercentage = null; // default to null if not present
    }

    // populate table with the data
    upcomingFightsTable.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
    var tbody = upcomingFightsTable.tBodies[0]
    var tr = tbody.insertRow(-1);
    var td1 = document.createElement('td'); // Fighter</th>
    var td2 = document.createElement('td'); // Opponent</th>
    var td3 = document.createElement('td'); // Predicted<br>Fighter<br>Odds</th>
    var td4 = document.createElement('td'); // Predicted<br>Opponent<br>Odds</th>
    var td5 = document.createElement('td'); // Best Bookie<br>For Fighter<br>Bet</th>
    var td6 = document.createElement('td'); // Kelly<br>bankroll%</th> 
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tr.appendChild(td4);
    tr.appendChild(td5);
    tr.appendChild(td6);
    // TODO use same_name function to match fighter names else we miss some
    fighter_stats = fighter_data[fighter]
    opponent_stats = fighter_data[opponent]
    if (!fighter_stats) {
      console.warn(`Fighter data not found for ${fighter}. Skipping row.`);
      fighterHtml = '#'
    } else {
      fighterHtml = fighter_stats['url']
    }
    if (!opponent_stats) {
      console.warn(`Fighter data not found for ${opponent}. Skipping row.`);
      opponentHtml = '#'
    } else {
      opponentHtml = opponent_stats['url']
    }

    if (fighterOdds == '' || fighterOdds == null) { //if no prediction was made
      tr.cells.item(0).innerHTML = `<a href=${fighterHtml} target="_blank" style = "color: white">${fighter}</a>`
      tr.cells.item(1).innerHTML = `<a href=${opponentHtml} target="_blank" style = "color: white">${opponent}</a>`
    }
    else if (fighterOdds[0] == '-') { // if fighter is predicted to win
      //TODO check if they have a wikipedia page and if not, link to their UFC profile https://www.ufc.com/athlete/${fighter.replace(" ", '_')#athlete-record
      tr.cells.item(0).innerHTML = `<a href=${fighterHtml} target="_blank" style = "color: gold">${fighter}</a>`
      tr.cells.item(1).innerHTML = `<a href=${opponentHtml} target="_blank" style = "color: white">${opponent}</a>`
    } else { // if opponent is predicted to win
      tr.cells.item(0).innerHTML = `<a href=${fighterHtml} target="_blank" style = "color: white">${fighter}</a>`
      tr.cells.item(1).innerHTML = `<a href=${opponentHtml} target="_blank" style = "color: gold">${opponent}</a>`
    }

    // convert fighter name by removing spaces for linking to bokeh plot
    // e.g. "Elves Brener" -> "elves_brener"
    let fighterLinkName = fighter.toLowerCase().replaceAll(" ", '_');
    let opponentLinkName = opponent.toLowerCase().replaceAll(" ", '_');
    // convert date to YYYY-MM-DD format for linking to bokeh plot
    // e.g. "August 2, 2025" -> "2025-08-02"
    let fightDate = vegas_odds['date'][i]; // e.g. "August 2, 2025"
    let dateObj = new Date(fightDate);
    let year = dateObj.getFullYear();
    let month = (dateObj.getMonth() + 1).toString().padStart(2, '0'); // Months are zero-based
    let day = dateObj.getDate().toString().padStart(2, '0');
    let fightDateFormatted = `${year}-${month}-${day}`; // e.g. "2025-08-02"
    oddsHtml = `src/content/bokehPlots/${fightDateFormatted}_${fighterLinkName}_vs_${opponentLinkName}_bokeh_barplot.html`
    tr.cells.item(2).innerHTML =  `<a href=${oddsHtml} target="_blank" style="color: white"; >${fighterOdds}</a>`;
    tr.cells.item(3).innerHTML = `<a href=${oddsHtml} target="_blank" style="color: white"; >${opponentOdds}</a>`;
    
    // make the fighter and opponent names bold and gold if they have higher expected value
    // than the opponent and fighter respectively
    // TODO indicate potential payout
    let evPickColor = '#85BB65'; // default color for expected value text (money green)
    let coloredBrText = '';
    let fav_color = '#6cddffff';
    let dog_color = '#e9b24cff';
    if (fighterBankrollPercentage && opponentBankrollPercentage) { // check if both expected values are defined
      if (fighterBankrollPercentage > opponentBankrollPercentage && fighterBankrollPercentage > 0) { //if fighter has higher expected value
        tr.cells.item(4).innerHTML = `${bestFighterBookie}<br>${bestFighterBookieOddsOnFighter}, ${bestFighterBookieOddsOnOpponent}`;
        if (parseInt(bestFighterBookieOddsOnFighter) < 0) { //if fighter is favorite
          fav_dog = 'fav'
          fav_dog_color = fav_color
        } else {
          fav_dog = 'dog'
          fav_dog_color = dog_color
        }
        payout = betPayout(fighterBankrollPercentage, parseInt(bestFighterBookieOddsOnFighter)).toFixed(2)
        coloredBrText = `<span class="clickable">${fighterBankrollPercentage}</span> -> <span style="color:${evPickColor}">${payout}</span><br>${fighter} (<span style="color:${fav_dog_color}">${fav_dog}</span>)`;
      } else if (opponentBankrollPercentage > fighterBankrollPercentage && opponentBankrollPercentage > 0) { //if opponent has higher expected value
        tr.cells.item(4).innerHTML = `${bestOpponentBookie}<br>${bestOpponentBookieOddsOnFighter}, ${bestOpponentBookieOddsOnOpponent}`;
        if (parseInt(bestOpponentBookieOddsOnOpponent) < 0) { //if opponent is favorite
          fav_dog = 'fav'
          fav_dog_color = fav_color
        } else {
          fav_dog = 'dog'
          fav_dog_color = dog_color
        }
        payout = betPayout(opponentBankrollPercentage, parseInt(bestOpponentBookieOddsOnOpponent)).toFixed(2)
        coloredBrText = `<span class="clickable">${opponentBankrollPercentage}</span> -> <span style="color:${evPickColor}">${payout}</span><br>${opponent} (<span style="color:${fav_dog_color}">${fav_dog}</span>)`;

      } else if (fighterBankrollPercentage == opponentBankrollPercentage && fighterBankrollPercentage > 0) { //if both have same expected value
        tr.cells.item(4).innerHTML = `${bestFighterBookie}<br>${bestFighterBookieOddsOnFighter}, ${bestFighterBookieOddsOnOpponent}`;
        if (parseInt(bestFighterBookieOddsOnFighter) < 0) { //if fighter is favorite
          fav_dog = 'fav'
          fav_dog_color = fav_color
        } else {
          fav_dog = 'dog'
          fav_dog_color = dog_color
        }
        payout = betPayout(fighterBankrollPercentage, parseInt(bestFighterBookieOddsOnFighter)).toFixed(2)
        coloredBrText = `<span class="clickable">${fighterBankrollPercentage}</span> -> <span style="color:${evPickColor}">${payout}</span><br>${fighter} (<span style="color:${fav_dog_color}">${fav_dog}</span>)`;

      } else { //if both have negative expected value
        tr.cells.item(4).innerHTML = `${bestFighterBookie}<br>${bestFighterBookieOddsOnFighter}, ${bestFighterBookieOddsOnOpponent}`;
        coloredBrText = `<span class="clickable">${0.0}</span>`;
      }
    }

    tr.cells.item(5).innerHTML = coloredBrText;

    var item = tr.cells.item(5);
    const clickable = item.querySelector('.clickable');
    if (clickable != null){
      clickable.addEventListener('click', function(event) {
        clickedSpan = event.target;

        // get the parent <td> of the span
        const cell = clickedSpan.closest('td');
        const row = cell.closest('tr');
        const cells = row.querySelectorAll('td');
        const fighterName = cells[0].innerText;
        const opponentName = cells[1].innerText;
        // populate the active fighter and opponent names
        const d = new Date();
        let month = d.getMonth();
        let year = d.getFullYear();
        var months = ["January", "February", 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', "November", 'December']
        document.getElementById('rc').value = fighterName;
        document.getElementById('bc').value = opponentName;
        document.getElementById('selectMonth_rc').value = months[month]
        document.getElementById('selectYear_rc').value = year
        document.getElementById('selectMonth_bc').value = months[month]
        document.getElementById('selectYear_bc').value = year
        selectFighterAndDate('rc')
        selectFighterAndDate('bc')
      })
    }

    tr.cells.item(0).style.backgroundColor = "#323232";
    tr.cells.item(1).style.backgroundColor = "#323232";
    tr.cells.item(2).style.backgroundColor = "#323232";
    tr.cells.item(3).style.backgroundColor = "#323232";
    tr.cells.item(4).style.backgroundColor = "#323232";
    tr.cells.item(5).style.backgroundColor = "#323232";
    tr.cells.item(0).style.color = "#ffffff";
    tr.cells.item(1).style.color = "#ffffff";
    tr.cells.item(2).style.color = "#ffffff";
    tr.cells.item(3).style.color = "#ffffff";
    tr.cells.item(4).style.color = "#ffffff";
    tr.cells.item(5).style.color = "#ffffff";
  }
}, 1000) //originally 350



setTimeout(() => { //this builds a table for the history of predictions which is built in python in the jupyter notebook UFC_Prediction_Model
  var numberModelCorrect = 0
  var numberTotal = 0
  var numTotalWithBookieOdds = 0
  var numBookieCorrect = 0
  for (const i in prediction_history['fighter name']) { //iterating over rows of prediction_history
    fighter = prediction_history['fighter name'][i]
    opponent = prediction_history['opponent name'][i]
    fighterOdds = String(prediction_history['predicted fighter odds'][i])
    opponentOdds = String(prediction_history['predicted opponent odds'][i])

    bestFighterBookieCol = prediction_history['best fighter bookie'] || null; // default to null if not present
    if (bestFighterBookieCol != null) {
      bestFighterBookie = bestFighterBookieCol[i];
    } else {
      bestFighterBookie = null; // default to null if not present
    }
    // get info on best bookie odds for fighter and opponent
    bestOpponentBookieCol = prediction_history['best opponent bookie'] || null;
    if (bestOpponentBookieCol != null) {
      bestOpponentBookie = bestOpponentBookieCol[i];
    } else {
      bestOpponentBookie = null; // default to null if not present
    }
    if (bestFighterBookie) {
        numTotalWithBookieOdds += 1; // count only if we have bookie odds
        // console.log(`bestFighterBookie is ${bestFighterBookie}`);
        bestFighterBookieOddsOnFighter = prediction_history[`fighter ${bestFighterBookie}`][i];
        bestFighterBookieOddsOnOpponent = prediction_history[`opponent ${bestFighterBookie}`][i];
    } else {
      bestFighterBookieOddsOnFighter = null; // default to null if not
      bestFighterBookieOddsOnOpponent = null; // default to null if not present
    }
    if (bestOpponentBookie) {
        bestOpponentBookieOddsOnFighter = prediction_history[`fighter ${bestOpponentBookie}`][i];
        bestOpponentBookieOddsOnOpponent = prediction_history[`opponent ${bestOpponentBookie}`][i];
    } else {  
      bestOpponentBookieOddsOnFighter = null; // default to null if not present
      bestOpponentBookieOddsOnOpponent = null; // default to null if not present
    }

    fighterBankrollPercentageCol = prediction_history['fighter bet bankroll percentage'] || null; // default to null if not present
    if (fighterBankrollPercentageCol != null) {
      fighterBankrollPercentage = parseFloat(fighterBankrollPercentageCol[i]).toFixed(2);
    } else {
      fighterBankrollPercentage = 0; // or some default value
    }

    fighterBetCol = prediction_history['fighter bet'] || null; // default to null if not present
    if (fighterBetCol != null) {
      fighterBet = parseFloat(fighterBetCol[i]).toFixed(2);
    } else {
      fighterBet = 0; // or some default value
    }

    opponentBankrollPercentageCol = prediction_history['opponent bet bankroll percentage'] || null; // default to null if not present
    if (opponentBankrollPercentageCol != null) {
      opponentBankrollPercentage = parseFloat(opponentBankrollPercentageCol[i]).toFixed(2);
    } else {
      opponentBankrollPercentage = 0; // or some default value
    }

    opponentBetCol = prediction_history['opponent bet'] || null; // default to null if not present
    if (opponentBetCol != null) {
      opponentBet = parseFloat(opponentBetCol[i]).toFixed(2);
    } else {  
      opponentBet = 0; // or some default value
    }


    bankrollCol = prediction_history['current bankroll after'] || null; // default to null if not present
    if (bankrollCol != null) {    
      currentBankroll = parseFloat(bankrollCol[i]).toFixed(2);
    } else {
      currentBankroll = 0; // or some default value
    }

    // TODO does not take into account if we bet on both the fighter and the opponent (maybe TODO)
    betResultCol = prediction_history['bet result'] || null; // default to null if not present
    if (betResultCol == null) {
      bankrollColor = 'white'; // default color if bet result is not present
    } else {
      betResult = betResultCol[i]; // default to null if not present
      if (betResult == 'W') {
        // console.log(`1 bet result is W for ${fighter} vs ${opponent}`);
        bankrollColor = 'green'; // color based on bankroll difference
      } else if (betResult == 'L') {
        // console.log(`2 bet result is L for ${fighter} vs ${opponent}`);
        bankrollColor = 'red'; // color based on bankroll difference
      } else {
        // console.log(`3 bet result is not W or L for ${fighter} vs ${opponent}`);
        bankrollColor = 'white'; // color based on bankroll difference
      }
    }

    numberTotal += 1;
    var fightHistoryTable = document.getElementById('tablehistory')
    fightHistoryTable.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
    var tbody = fightHistoryTable.tBodies[0]
    var tr = tbody.insertRow(-1);
    var td1 = document.createElement('td'); // Fighter vs Opponent
    var td2 = document.createElement('td'); // Predicted Odds
    var td3 = document.createElement('td'); // best fighter bookie odds
    var td4 = document.createElement('td'); // bankroll percentage
    var td5 = document.createElement('td'); // current bankroll after
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tr.appendChild(td4);
    tr.appendChild(td5);

    tr.cells.item(0).style.backgroundColor = "#323232";
    tr.cells.item(1).style.backgroundColor = "#323232";
    tr.cells.item(2).style.backgroundColor = "#323232";
    tr.cells.item(3).style.backgroundColor = "#323232";
    tr.cells.item(4).style.backgroundColor = "#323232";
    tr.cells.item(0).style.color = "#ffffff";
    tr.cells.item(1).style.color = "#ffffff";
    tr.cells.item(2).style.color = "#ffffff";
    tr.cells.item(3).style.color = "#ffffff";
    tr.cells.item(4).style.color = "#ffffff";

    tr.cells.item(0).innerHTML = `${fighter} vs ${opponent}`;
    tr.cells.item(1).innerHTML = `${fighterOdds}, ${opponentOdds}`;

    bestBookie = '';
    bestBookieOddsOnFighter = 0;
    bestBookieOddsOnOpponent = 0;
    bankrollPercentage = 0;
    bet = 0;
    bettingOn = '';
    if (fighterBankrollPercentage > 0){
      bestBookieOddsOnFighter = bestFighterBookieOddsOnFighter;
      bestBookieOddsOnOpponent = bestFighterBookieOddsOnOpponent;
      bestBookie = bestFighterBookie;
      bankrollPercentage = fighterBankrollPercentage;
      bet = fighterBet;
      bettingOn = fighter;
    }
    if (opponentBankrollPercentage > fighterBankrollPercentage){
      bestBookieOddsOnFighter = bestOpponentBookieOddsOnFighter;
      bestBookieOddsOnOpponent = bestOpponentBookieOddsOnOpponent;
      bestBookie = bestOpponentBookie;
      bankrollPercentage = opponentBankrollPercentage;
      bet = opponentBet;
      bettingOn = opponent;
    }
    tr.cells.item(2).innerHTML = `${bestBookie}<br>${bestBookieOddsOnFighter}, ${bestBookieOddsOnOpponent}`;
    tr.cells.item(3).innerHTML = `${bankrollPercentage}% = ${bet}$<br>${bettingOn}`;

    // color the current bankroll based on the result of the bet (green = won, red = lost)
    currentBankrollText = `<span style="color:${bankrollColor}">${currentBankroll}</span>`;
    tr.cells.item(4).innerHTML = currentBankrollText;

    color = 'gold'; //winner color
    fighterWon = false;
    opponentWon = false;
    if (prediction_history['correct?'][i] == 1) {
      tr.cells.item(1).style.backgroundColor = "#00ff00";
      numberModelCorrect += 1
      if (parseInt(fighterOdds) < parseInt(opponentOdds)) { //if fighter is predicted to win
        coloredFightText = `<span style="color:${color}">${fighter}</span> | vs | <span>${opponent}</span>`;
        fighterWon = true;
      } else {
        coloredFightText = `<span>${fighter}</span> | vs | <span style="color:${color}">${opponent}</span>`;
        opponentWon = true;
      }
    } else if (prediction_history['correct?'][i] == 0) {
      tr.cells.item(1).style.backgroundColor = "#ff0000";
      if (parseInt(fighterOdds) < 0) {
        coloredFightText = `<span>${fighter}</span> | vs | <span style="color:${color}">${opponent}</span>`;
        opponentWon = true;
      } else {
        coloredFightText = `<span style="color:${color}">${fighter}</span> | vs | <span>${opponent}</span>`;
        fighterWon = true;
      }
    } else if (prediction_history['correct?'][i] == 'N/A') {
      tr.cells.item(1).style.backgroundColor = "#b3b3b3";
      coloredFightText = `<span>${fighter}</span> | vs | <span>${opponent}</span>`;
    } else {
      console.log(`something is wrong with the prediction history data for ${fighter} vs ${opponent}`);
      coloredFightText = `<span>${fighter}</span> | vs | <span>${opponent}</span>`;
    }

    tr.cells.item(0).innerHTML = coloredFightText;

    // color bookie bet columns to indicate if they picked correctly
    if (bestFighterBookie != null) {
      if (fighterWon){
        if (parseInt(bestBookieOddsOnFighter) < parseInt(bestBookieOddsOnOpponent)) {
          numBookieCorrect += 1;
          tr.cells.item(2).style.backgroundColor = "#00ff00";
        } else {
          tr.cells.item(2).style.backgroundColor = "#ff0000";
        }
      } else if (opponentWon) {
        if (parseInt(bestBookieOddsOnOpponent) < parseInt(bestBookieOddsOnFighter)) {
          numBookieCorrect += 1;
          tr.cells.item(2).style.backgroundColor = "#00ff00";
        } else {
          tr.cells.item(2).style.backgroundColor = "#ff0000";
        }
      }
    }
  }
  var acc = numberModelCorrect / numberTotal;
  var bookieAcc = numBookieCorrect / numTotalWithBookieOdds;
  var accuracy = document.getElementById("myaccuracy")
  // round accuracy to 2 decimal places
  acc = (Math.round(acc * 10000) / 100).toFixed(2);
  bookieAcc = (Math.round(bookieAcc * 10000) / 100).toFixed(2);
  console.log(`Bookie accuracy: ${bookieAcc}`)
  // set accuracy text
  accuracy.innerText = `Accuracy: ${acc}%`;
  var bookieAccuracy = document.getElementById("bookieaccuracy")
  bookieAccuracy.innerText = `Bookie Accuracy: ${bookieAcc}%`;
}, 1500) //originally 450

//set initial table values and display fight
setTimeout(() => {
  var upcomingFightsTable = document.getElementById('upcoming')
  const d = new Date();
  let month = d.getMonth();
  let year = d.getFullYear();
  var months = ["January", "February", 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', "November", 'December']
  document.getElementById('rc').value = upcomingFightsTable.rows[2].cells[0].textContent
  document.getElementById('bc').value = upcomingFightsTable.rows[2].cells[1].textContent
  document.getElementById('selectMonth_rc').value = months[month]
  document.getElementById('selectYear_rc').value = year
  document.getElementById('selectMonth_bc').value = months[month]
  document.getElementById('selectYear_bc').value = year
  selectFighterAndDate('rc')
  selectFighterAndDate('bc')
  var myTab;
  myTab = document.getElementById("tableoutcome");
  // LOOP THROUGH EACH ROW OF THE TABLE AFTER HEADER.
  myTab.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
  myTab.rows.item(1).cells.item(0).style.backgroundColor = "#323232";
  myTab.rows.item(1).cells.item(1).style.backgroundColor = "#323232";
  myTab.rows.item(1).cells.item(2).style.backgroundColor = "#323232";
  /*
  for (let i = 3; i < 3 + 11; i++) {
    for (let j = 0; j < 3; j++) {
      myTab.rows.item(i).cells.item(j).style.backgroundColor = "#323232";
    }
  }
  */

}, 2000) //originally 500

console.log(document.getElementById("loader"))

//make a loading screen
setTimeout(() => {
  //set text of futurefights to "loading..."
  var card_info_text = card_info['title'] + "     " + card_info['date'];
  //for debugging purposes
  //card_info_text = "UFC FIGHT NIGHT: SANDHAGEN VS. NURMAGOMEDOV      December 15, 2023"; // example of very long title
  //card_info_text = "UFC 292: ALJAMAIN STERLING VS O'MALLEY      August 19th"; // example of long title
  //card_info_text = "BELLATOR VS RIZIN LETS GO MANWO      October 31st"; // example of medium title
  //card_info_text = "UFC 294 Conor vs Khabib    May 23, 2023"; // example of very short title
  //card_info_text = "UFC 294      August 23, 2023"; // example of very short title
  // find the length of the string card_info_text
  var card_info_text_length = card_info_text.length;
  console.log(`card info text length ${card_info_text_length}`)
  if (card_info_text_length > 58) {
    document.getElementById("card-title-style").style.fontSize = "15px";
    console.log('card title and date is very long. Case A.')
  } else if (card_info_text_length > 50) {
    document.getElementById("card-title-style").style.fontSize = "20px";
    console.log('card title and date is long. Case B.')
  } else if (card_info_text_length > 40) {
    document.getElementById("card-title-style").style.fontSize = "25px";
    console.log('card title and date is medium. Case C.')
  } else if (card_info_text_length > 30) {
    document.getElementById("card-title-style").style.fontSize = "30px";
    console.log('card title and date is short. Case D.')
  } else {
    document.getElementById("card-title-style").style.fontSize = "40px";
    console.log('card title and date is very short. Case E.')
  }
  document.getElementById("card title and date").textContent = card_info_text;
  document.getElementById("loader").
    style.display = "none";
}, 2500) //originally 600

