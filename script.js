theta = {};
intercept = {};
fighter_data = {}
ufcfightscrap = {}
vegas_odds = {}
prediction_history = {}
card_info = {}

$.getJSON('src/models/buildingMLModel/data/external/card_info.json', function (data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function (i, f) {
    card_info[i] = f
  });
});

$.getJSON('src/models/buildingMLModel/data/external/theta.json', function (data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function (i, f) {
    theta[i] = f.toFixed(2)
  });
});

$.getJSON('src/models/buildingMLModel/data/external/intercept.json', function (data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function (i, f) {
    intercept[i] = f.toFixed(2)
  });
});

$(function () { // building object fighter_data from fighter_data.json file
  $.getJSON('src/models/buildingMLModel/data/external/fighter_stats.json', function (data) {//for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
    $.each(data, function (i, f) {//create entry in local object
      const select = document.getElementById('fighters')
      select.insertAdjacentHTML('beforeend', `<option value="${i}">${i}</option>`)
      fighter_data[i] = f
    });
  });
});

$(function () { // building object ufcfightscrap from ufcfightscrap.json file
  $.getJSON('src/models/buildingMLModel/data/external/ufcfightscrap.json', function (data) {//for each input (i,f), i is the key (a number) and f is the value (all the data of the fight)
    $.each(data, function (i, f) {
      ufcfightscrap[i] = f
    });
  });
});

$(function () { // building object vegas_odds from vegas_odds.json file
  $.getJSON('src/models/buildingMLModel/data/external/vegas_odds.json', function (data) {//for each input (i,f), i is the key a column name like fighter name and f is the value (an object with keys being integers and values being strings (odds or names))
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
  for (let j = 0; j < Object.keys(vegas_odds).length; j++) {
    prediction_history[Object.keys(vegas_odds)[j]] = {}
    prediction_history['predicted fighter odds'] = {}
    prediction_history['predicted opponent odds'] = {}
    prediction_history['winner'] = {}
    prediction_history['correct'] = {}
  }
  //filling in any previous data into prediction_history
  $(function () {
    //var people = [];
    $.getJSON('src/models/buildingMLModel/data/external/prediction_history.json', function (data) {
      //for each input (i,f), i is the key a column name like fighter name and f is the value (an object with keys being integers and values being strings (odds or names))
      $.each(data, function (i, f) {
        //create entry in local object
        prediction_history[i] = f
      });
    });
  });
}, 500) // originally 250

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

function selectFighter(id, out) {
  selectElement = document.querySelector('#' + id);
  output = selectElement.value;
  //document.querySelector('.' + out).textContent = output;
  let i = id[6]
  picIndex += 1
  let j = (picIndex) % 4 + 1
  name = selectElement.value;
  var name_encoded = encodeURIComponent(name)
  var name_decoded = decodeURIComponent(name_encoded)
  name_decoded = decodeURIComponent(name_decoded)
  name_decoded = name_decoded.replace(new RegExp(' ', 'g'), '');

  // set the path to check if gif file exists (otherwise use pictures)
  if (checkFileExist("src/models/buildingMLModel/gifs/postCNNGIFs/" + name_decoded + ".gif")) {
    document.getElementById("fighter" + i + "pic").src = "src/models/buildingMLModel/gifs/postCNNGIFs/" + name_decoded + ".gif" //sets the image
  } else if (checkFileExist("src/models/buildingMLModel/images2/" + j + name_decoded + ".jpg")) {
    document.getElementById("fighter" + i + "pic").src = "src/models/buildingMLModel/images2/" + j + name_decoded + ".jpg" //sets the image
  } else {
    document.getElementById("fighter" + i + "pic").src = "src/models/buildingMLModel/images/" + j + name_decoded + ".jpg" //sets the image
  }
  if (i == '1') {
    populateTaleOfTheTape(output, 'rc')
    populateLast5Fights(output, 'rc')
  } else {
    populateTaleOfTheTape(output, 'bc')
    populateLast5Fights(output, 'bc')
  }
}

function selectDate(monthid, monthout, yearid, yearout) {
  //sets the image to be the image of the fighter
  selectMonth = document.querySelector('#' + monthid);
  output = selectMonth.value;
  document.querySelector('.' + monthout).textContent = output;
  selectYear = document.querySelector('#' + yearid);
  output2 = selectYear.value;
  document.querySelector('.' + yearout).textContent = output2;
}

function selectFighterAndDate(id, out, monthid, monthout, yearid, yearout) {
  selectFighter(id, out)
  selectDate(monthid, monthout, yearid, yearout)
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
  if (corner == 'rc') {
    yr = document.querySelector('#' + 'f1selectyear').value;
    myTab = document.getElementById("table1");
  } else if (corner == 'bc') {
    yr = document.querySelector('#' + 'f2selectyear').value;
    myTab = document.getElementById("table2");
  }
  // LOOP THROUGH EACH ROW OF THE TABLE AFTER HEADER.
  myTab.rows.item(0).cells.item(0).style.backgroundColor = "#212121";

  myTab.rows.item(0).cells.item(0).innerHTML = fighter;

  myTab.rows.item(2).cells.item(0).innerHTML = fighter_data[fighter]['height'];
  myTab.rows.item(2).cells.item(1).innerHTML = fighter_data[fighter]['reach'];
  myTab.rows.item(2).cells.item(2).innerHTML = fighter_age(fighter, yr);
  myTab.rows.item(2).cells.item(3).innerHTML = fighter_data[fighter]['stance'];
  //document.querySelector('.tableEntry').textContent = fighter
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
  if (corner == 'rc') {
    yr = document.querySelector('#' + 'f1selectyear').value;
    myTab = document.getElementById("l5ytable1");
  } else if (corner == 'bc') {
    yr = document.querySelector('#' + 'f2selectyear').value;
    myTab = document.getElementById("l5ytable2");
  }
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
    let yearDiff = parseInt(yr) - ufcfightscrap[fight]['date'].slice(-4)
    // note I changed to checking set equality of the set {firstName, middleName, lastName} because different orderings are used in different databases
    if (same_name(ufcfightscrap[fight]['fighter'], fighter) && yearDiff >= 0) {
      result = ufcfightscrap[fight]['result']
      fighter = ufcfightscrap[fight]['fighter']
      opponent = ufcfightscrap[fight]['opponent']
      method = ufcfightscrap[fight]['method']
      date = ufcfightscrap[fight]['date']
      fightNumber += 1
      myTab.rows.item(0).cells.item(0).innerHTML = fighter
      myTab.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
      myTab.rows.item(fightNumber).cells.item(0).innerHTML = opponent
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


function myFunction1() {
  document.getElementById("myDropdown1").classList.toggle("show");
}

function myFunction2() {
  document.getElementById("myDropdown2").classList.toggle("show");
}

function filterFunction1() {
  var input, filter, ul, li, a, i;
  input = document.getElementById("myInput1");
  filter = input.value.toUpperCase();
  div = document.getElementById("myDropdown1");
  a = div.getElementsByTagName("a");
  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}

function filterFunction2() {
  var input, filter, ul, li, a, i;
  input = document.getElementById("myInput2");
  filter = input.value.toUpperCase();
  div = document.getElementById("myDropdown2");
  a = div.getElementsByTagName("a");
  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}

/* #here we were copying new fights from vegas odds to prediction history... now we do this in the file update_and_rebuild_model.py
setTimeout(() => {
  //iterates over rows of vegas_odds (each row is a collection of bookie odds data for a specific fight)
  for (let i = 0; i < Object.values(vegas_odds['fighter name']).length; i++) {
    //checks if both fighters are in the ufc
    let fighter_in_ufc = Object.keys(fighter_data).some(function(name) {
      return same_name(name, vegas_odds['fighter name'][i])
    });
    let opponent_in_ufc = Object.keys(fighter_data).some(function(name) {
      return same_name(name, vegas_odds['opponent name'][i])
    });
    if (fighter_in_ufc && opponent_in_ufc) {
      let bookie_scores = []
      for (let j = 0; j < Object.keys(vegas_odds).length; j++) {
        bookie_scores.push(vegas_odds[Object.keys(vegas_odds)[j]][i])
      }
      let all_bookie_scores = bookie_scores.every(function(value) {
        return value.length > 0
      });
      if (all_bookie_scores) { //makes sure that all bookies have scores in the books (to avoid fights in smaller organizations or fantasy fights)
        //copies row i of vegas_odds to prediction_history
        for (let j = 0; j < Object.keys(vegas_odds).length; j++) {
          prediction_history[Object.keys(vegas_odds)[j]][i] = vegas_odds[Object.keys(vegas_odds)[j]][i]
        }
      }
    }
  }
}, 350)
*/

//the following is now done in the file update_and_rebuild_model.py but we'll keep this here if needed
/*
setTimeout(()=>{
  //making predictions for fights on the books
  for (const i in prediction_history['fighter name']){ //iterating over rows of prediction_history
    if (!prediction_history['predicted fighter odds'][i]){
      fighter = prediction_history['fighter name'][i]
      opponent = prediction_history['opponent name'][i]
      month1=document.getElementById('f1selectmonth').value
      month2=document.getElementById('f1selectmonth').value
      year1=document.getElementById('f1selectyear').value
      year2=document.getElementById('f1selectyear').value
      odds=betting_oddsAbsolute(fighter, opponent, month1, year1, month2, year2)
      prediction_history['predicted fighter odds'][i]=odds[0]
      prediction_history['predicted opponent odds'][i]=odds[1]
    }
  }
}, 750)
*/

//Building upcoming predictions table

setTimeout(() => { //timeout because other data needs to load first (probably better to do with async)
  var upcomingFightsTable = document.getElementById('upcoming')
  for (const i in vegas_odds['fighter name']) {
    fighter = vegas_odds['fighter name'][i]
    opponent = vegas_odds['opponent name'][i]
    fighterOdds = vegas_odds['predicted fighter odds'][i]
    opponentOdds = vegas_odds['predicted opponent odds'][i]
    avBookieOdds = vegas_odds['average bookie odds'][i]
    upcomingFightsTable.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
    var tbody = upcomingFightsTable.tBodies[0]
    var tr = tbody.insertRow(-1);
    var td1 = document.createElement('td');
    var td2 = document.createElement('td');
    var td3 = document.createElement('td');
    var td4 = document.createElement('td');
    var td5 = document.createElement('td');
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tr.appendChild(td4);
    tr.appendChild(td5);
    if (fighterOdds == '') { //if no prediction was made
      tr.cells.item(0).innerHTML = `<a href=https://en.wikipedia.org/wiki/${fighter.replace(" ", '_')}#Mixed_martial_arts_record target="_blank" style = "color: white">${fighter}</a>`
      tr.cells.item(1).innerHTML = `<a href=https://en.wikipedia.org/wiki/${opponent.replace(" ", '_')}#Mixed_martial_arts_record target="_blank" style = "color: white">${opponent}</a>`
    }
    else if (fighterOdds[0] == '-') { // if fighter is predicted to win
      //TODO check if they have a wikipedia page and if not, link to their UFC profile https://www.ufc.com/athlete/${fighter.replace(" ", '_')#athlete-record
      tr.cells.item(0).innerHTML = `<a href=https://en.wikipedia.org/wiki/${fighter.replace(" ", '_')}#Mixed_martial_arts_record target="_blank" style = "color: gold">${fighter}</a>`
      tr.cells.item(1).innerHTML = `<a href=https://en.wikipedia.org/wiki/${opponent.replace(" ", '_')}#Mixed_martial_arts_record target="_blank" style = "color: white">${opponent}</a>`
    } else { // if opponent is predicted to win
      tr.cells.item(0).innerHTML = `<a href=https://en.wikipedia.org/wiki/${fighter.replace(" ", '_')}#Mixed_martial_arts_record target="_blank" style = "color: white">${fighter}</a>`
      tr.cells.item(1).innerHTML = `<a href=https://en.wikipedia.org/wiki/${opponent.replace(" ", '_')}#Mixed_martial_arts_record target="_blank" style = "color: gold">${opponent}</a>`
    }
    tr.cells.item(2).innerHTML = fighterOdds;
    tr.cells.item(3).innerHTML = opponentOdds;
    tr.cells.item(4).innerHTML = avBookieOdds;
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
  }
}, 1000) //originally 350



setTimeout(() => { //this builds a table for the history of predictions which is built in python in the jupyter notebook UFC_Prediction_Model
  var numberModelCorrect = 0
  var numberVegasCorrect = 0
  var numberTotal = 0
  for (const i in prediction_history['fighter name']) { //iterating over rows of prediction_history
    fighter = prediction_history['fighter name'][i]
    opponent = prediction_history['opponent name'][i]
    fighterOdds = String(prediction_history['predicted fighter odds'][i])
    opponentOdds = String(prediction_history['predicted opponent odds'][i])
    avBookieOdds = prediction_history['average bookie odds'][i]
    numberTotal += 1;
    var upcomingFightsTable = document.getElementById('tablehistory')
    upcomingFightsTable.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
    var tbody = upcomingFightsTable.tBodies[0]
    var tr = tbody.insertRow(-1);
    var td1 = document.createElement('td');
    var td2 = document.createElement('td');
    var td3 = document.createElement('td');
    var td4 = document.createElement('td');
    var td5 = document.createElement('td');
    var td6 = document.createElement('td');
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tr.appendChild(td4);
    tr.appendChild(td5);
    tr.appendChild(td6);
    tr.cells.item(0).innerHTML = fighter;
    tr.cells.item(1).innerHTML = opponent;
    tr.cells.item(2).innerHTML = fighterOdds;
    tr.cells.item(3).innerHTML = opponentOdds;
    tr.cells.item(5).innerHTML = avBookieOdds;
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
    tr.cells.item(5).style.color = "#ffffff";
    if (prediction_history['correct?'][i] == 1) {
      tr.cells.item(4).innerHTML = 'yes'
      tr.cells.item(4).style.backgroundColor = "#00ff00";
      numberModelCorrect += 1
      if (parseInt(fighterOdds) < 0) {
        tr.cells.item(0).style.fontWeight = "bold";
        tr.cells.item(0).style.fontSize = "15px";
        tr.cells.item(0).style.color = "#ffd700";
      } else {
        tr.cells.item(1).style.fontWeight = "bold";
        tr.cells.item(1).style.fontSize = "15px";
        tr.cells.item(1).style.color = "#ffd700";
      }
    } else {
      tr.cells.item(4).innerHTML = 'no'
      tr.cells.item(4).style.backgroundColor = "#ff0000";
      if (parseInt(fighterOdds) < 0) {
        tr.cells.item(1).style.fontWeight = "bold";
        tr.cells.item(1).style.fontSize = "15px";
        tr.cells.item(1).style.color = "#ffd700";
      } else {
        tr.cells.item(0).style.fontWeight = "bold";
        tr.cells.item(0).style.fontSize = "15px";
        tr.cells.item(0).style.color = "#ffd700";
      }
    }
  }
  var acc = numberModelCorrect / numberTotal;
  var accuracy = document.getElementById("myaccuracy")
  accuracy.innerText = `Accuracy: ${acc}`;
}, 1500) //originally 450

//set initial table values and display fight
setTimeout(() => {
  var upcomingFightsTable = document.getElementById('upcoming')
  const d = new Date();
  let month = d.getMonth();
  let year = d.getFullYear();
  var months = ["January", "February", 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', "November", 'December']
  document.getElementById('select1').value = upcomingFightsTable.rows[2].cells[0].textContent
  document.getElementById('select2').value = upcomingFightsTable.rows[2].cells[1].textContent
  document.getElementById('f1selectmonth').value = months[month]
  document.getElementById('f1selectyear').value = year
  document.getElementById('f2selectmonth').value = months[month]
  document.getElementById('f2selectyear').value = year
  selectFighterAndDate('select1', 'name1', 'f1selectmonth', 'month1', 'f1selectyear', 'year1')
  selectFighterAndDate('select2', 'name2', 'f2selectmonth', 'month2', 'f2selectyear', 'year2')
  var myTab;
  myTab = document.getElementById("tableoutcome");
  // LOOP THROUGH EACH ROW OF THE TABLE AFTER HEADER.
  myTab.rows.item(0).cells.item(0).style.backgroundColor = "#212121";
  myTab.rows.item(1).cells.item(0).style.backgroundColor = "#323232";
  myTab.rows.item(1).cells.item(1).style.backgroundColor = "#323232";
  myTab.rows.item(1).cells.item(2).style.backgroundColor = "#323232";
  myTab.rows.item(2).cells.item(0).style.backgroundColor = "#212121";
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

