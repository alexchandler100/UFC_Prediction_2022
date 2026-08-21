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
    // The weekly card may have rendered before this larger file arrived. Render
    // again so fighter profile links are added without relying on a timer.
    if (vegas_odds['fighter name']) {
      renderUpcomingPredictions();
    }
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
    renderUpcomingPredictions();
  }).fail(function () {
    renderUpcomingPredictions('Weekly forecast data could not be loaded.');
  });
});

const years = document.getElementById('years')
const currentYear = new Date().getFullYear()
for (let i = 0; i < 30; i++) {
  const year = currentYear - i
  years.insertAdjacentHTML('beforeend', `
<option value="${year}">${year}</option>
`)
}
document.getElementById('selectYear_rc').value = currentYear
document.getElementById('selectYear_bc').value = currentYear

// Website fight data is stored as ISO YYYY-MM-DD today, while older snapshots
// used human-readable dates.  Never infer a year from the last four characters
// of an unknown format ("2026-05-09" would otherwise become "5-09").
function yearFromDate(dateValue) {
  const text = String(dateValue || '').trim()
  const isoMatch = text.match(/^(\d{4})-\d{2}-\d{2}$/)
  if (isoMatch) {
    return parseInt(isoMatch[1], 10)
  }
  const yearMatch = text.match(/\b(19|20)\d{2}\b/)
  return yearMatch ? parseInt(yearMatch[0], 10) : NaN
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

// function that converts probability to american odds
function probToAmerican(prob) {
  if (prob >= 0.5) {
    return Math.round(-100 * prob / (1 - prob));
  } else {
    return Math.round(100 * (1 - prob) / prob);
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

function selectFighter(name, id) { // id is 'rc' or 'bc' for red corner or blue corner
  populateListWithName(id, name);
  document.getElementById(id).value = name;
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

function selectDate(id) {
  var monthList = document.getElementById(`selectMonth_${id}`);
  var selectedMonth = monthList.value;
  monthList.textContent = selectedMonth;
  var yearList = document.getElementById(`selectYear_${id}`);
  var selectedYear = yearList.value;
  yearList.textContent = selectedYear;
}

function populateListWithName(id, name) {
  var selectElement = document.getElementById(id + 'List');
  selectElement.value = name;
}

function getnamefromlist(id) {
  // grab current value from list <input onfocus=this.value='' id=bc type="text" list="fighters" value=""><br><br>
  // var selectElement = document.querySelector('#' + id);
  var selectElement = document.getElementById(id + 'List');
  fighterNameText = selectElement.value;
  // fighterNameText = selectElement.options[selectElement.selectedIndex].text;
  return fighterNameText
}


function selectFighterHighlightedInList(id) {
  name = getnamefromlist(id)
  selectFighter(name, id)
  selectDate(id)  
}

function selectFighterAndDate(name, id) {
  selectFighter(name, id)
  selectDate(id)
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
  let yearBorn = yearFromDate(fighter_data[fighterName]['dob'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufcfightscrap[fight]['date'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufcfightscrap[fight]['date'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufcfightscrap[fight]['date'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufcfightscrap[fight]['date'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufcfightscrap[fight]['date'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufcfightscrap[fight]['date'])
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
    let yearDiff = parseInt(year) - yearFromDate(ufc_wins_list[i][2])
    if (yearDiff < 0) {
      continue
    }
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
    let yearDiff = parseInt(year) - yearFromDate(ufc_wins_list[i][2])
    if (yearDiff < 0) {
      continue
    }
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
    let yearDiff = parseInt(year) - yearFromDate(ufc_wins_list[i][2])
    if (yearDiff < 0) {
      continue
    }
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

function avg_count_diff(stat, fighter, opponent, inf_abs, fighterYear, opponentYear) {
  if (isNaN(avg_count(stat, fighter, inf_abs, fighterYear)) || isNaN(avg_count(stat, opponent, inf_abs, opponentYear))) {
    return 0
  }
  return avg_count(stat, fighter, inf_abs, fighterYear) - avg_count(stat, opponent, inf_abs, opponentYear)
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
  let av_total_strikes_diff = avg_count_diff('total_strikes_landed', guy1, guy2, 'abs', yr1, yr2).toFixed(2)
  let av_abs_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'abs', yr1, yr2).toFixed(2)
  let av_inf_gr_strikes = avg_count_diff('ground_strikes_landed', guy1, guy2, 'inf', yr1, yr2).toFixed(2)
  let av_tk_atmps_diff = avg_count_diff('takedowns_attempts', guy1, guy2, 'inf', yr1, yr2).toFixed(2)
  let av_inf_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'inf', yr1, yr2).toFixed(2)
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
  let av_total_strikes_diff = avg_count_diff('total_strikes_landed', guy1, guy2, 'abs', yr1, yr2).toFixed(2)
  let av_abs_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'abs', yr1, yr2).toFixed(2)
  let av_inf_gr_strikes = avg_count_diff('ground_strikes_landed', guy1, guy2, 'inf', yr1, yr2).toFixed(2)
  let av_tk_atmps_diff = avg_count_diff('takedowns_attempts', guy1, guy2, 'inf', yr1, yr2).toFixed(2)
  let av_inf_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'inf', yr1, yr2).toFixed(2)
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
          selectFighter(opponentName, corner)
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

// Build the upcoming forecast table from the published JSON contract. The
// weekly point-in-time model is deliberately not connected to the legacy
// browser calculator or its old Bokeh explanations.
function tableValue(table, column, index) {
  const values = table && table[column];
  if (!values || !Object.prototype.hasOwnProperty.call(values, index)) {
    return null;
  }
  return values[index];
}

function hasDisplayValue(value) {
  if (value === null || value === undefined) {
    return false;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value);
  }
  const text = String(value).trim().toLowerCase();
  return text !== '' && text !== 'nan' && text !== 'none' &&
    text !== 'null' && text !== 'undefined';
}

function firstDisplayValue() {
  for (const value of arguments) {
    if (hasDisplayValue(value)) {
      return value;
    }
  }
  return null;
}

function finiteNumber(value) {
  if (!hasDisplayValue(value)) {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatProbability(value) {
  const probability = finiteNumber(value);
  if (probability === null || probability < 0 || probability > 1) {
    return '—';
  }
  return (probability * 100).toFixed(1) + '%';
}

function formatOdds(value) {
  if (!hasDisplayValue(value)) {
    return '—';
  }
  const text = String(value).trim();
  const number = Number(text);
  if (!Number.isFinite(number)) {
    return text;
  }
  return number > 0 && text[0] !== '+' ? '+' + text : text;
}

function formatHistoryCount(value) {
  const count = finiteNumber(value);
  return count === null || count < 0 ? '—' : String(Math.round(count));
}

function humanizeForecastLabel(value) {
  if (!hasDisplayValue(value)) {
    return '';
  }
  const key = String(value).trim().toLowerCase();
  const labels = {
    'model': 'Model available',
    'low_history': 'Low history — interpret cautiously',
    'abstain_unresolved_identity': 'Model unavailable — unresolved fighter identity',
    'market_no_vig': 'Market no-vig consensus',
    'market_no_vig_consensus': 'Market no-vig consensus',
    'point_in_time_model': 'Point-in-time statistics model',
    'stats_model': 'Point-in-time statistics model',
    'stats_model_low_history': 'Point-in-time model (low history)',
    'weekly_forecast': 'Weekly forecast',
    'legacy_weekly_prediction': 'Legacy weekly prediction',
    'disabled_pending_market_relative_validation':
      'Betting disabled — prospective return/CLV validation active'
  };
  if (labels[key]) {
    return labels[key];
  }
  const text = key.replaceAll('_', ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function setCellLines(cell, lines) {
  cell.textContent = '';
  const visibleLines = lines.filter(hasDisplayValue);
  if (visibleLines.length === 0) {
    cell.textContent = '—';
    return;
  }
  visibleLines.forEach(function (line, lineIndex) {
    if (lineIndex > 0) {
      cell.appendChild(document.createElement('br'));
    }
    cell.appendChild(document.createTextNode(String(line)));
  });
}

function setFighterCell(cell, fighterName, isFavored) {
  cell.textContent = '';
  const name = hasDisplayValue(fighterName) ? String(fighterName) : 'Unknown fighter';
  const profile = fighter_data[name];
  const profileUrl = profile && hasDisplayValue(profile.url) ? profile.url : null;
  const label = profileUrl ? document.createElement('a') : document.createElement('span');
  label.textContent = name;
  label.style.color = isFavored ? 'gold' : 'white';
  if (profileUrl) {
    label.href = profileUrl;
    label.target = '_blank';
    label.rel = 'noopener noreferrer';
  }
  cell.appendChild(label);
}

function americanOddsProbability(value) {
  const odds = finiteNumber(value);
  if (odds === null || odds === 0) {
    return null;
  }
  return odds > 0 ? 100 / (odds + 100) : -odds / (-odds + 100);
}

function favoredForecastSide(probabilityValue, fighterOdds, opponentOdds) {
  const probability = finiteNumber(probabilityValue);
  if (probability !== null && probability >= 0 && probability <= 1) {
    if (probability > 0.5) {
      return 'fighter';
    }
    if (probability < 0.5) {
      return 'opponent';
    }
    return null;
  }
  const fighterProbability = americanOddsProbability(fighterOdds);
  const opponentProbability = americanOddsProbability(opponentOdds);
  if (fighterProbability === null || opponentProbability === null ||
      fighterProbability === opponentProbability) {
    return null;
  }
  return fighterProbability > opponentProbability ? 'fighter' : 'opponent';
}

function renderWeeklyModelProvenance(indices, loadError) {
  const provenance = document.getElementById('weekly-model-provenance');
  if (!provenance) {
    return;
  }
  if (loadError) {
    provenance.textContent = loadError;
    return;
  }
  let provenanceIndex = null;
  for (const index of indices) {
    if (hasDisplayValue(tableValue(vegas_odds, 'model id', index))) {
      provenanceIndex = index;
      break;
    }
  }
  if (provenanceIndex === null) {
    provenance.textContent =
      'Legacy weekly forecast loaded; point-in-time model provenance is not available for this card.';
    return;
  }
  const modelVersion = tableValue(vegas_odds, 'model version', provenanceIndex);
  const modelId = tableValue(vegas_odds, 'model id', provenanceIndex);
  const trainedThrough = tableValue(vegas_odds, 'model trained through', provenanceIndex);
  const details = ['Point-in-time model'];
  if (hasDisplayValue(modelVersion)) {
    details.push('version ' + modelVersion);
  }
  details.push('ID ' + modelId);
  if (hasDisplayValue(trainedThrough)) {
    details.push('trained through ' + trainedThrough);
  }
  provenance.textContent = details.join(' · ');
}

function renderUpcomingPredictions(loadError) {
  const upcomingFightsTable = document.getElementById('upcoming');
  if (!upcomingFightsTable || !upcomingFightsTable.tBodies.length) {
    return;
  }
  const tbody = upcomingFightsTable.tBodies[0];
  tbody.textContent = '';
  const fighterNames = vegas_odds && vegas_odds['fighter name'];
  const indices = fighterNames ? Object.keys(fighterNames) : [];
  renderWeeklyModelProvenance(indices, loadError);

  if (loadError || indices.length === 0) {
    const row = tbody.insertRow(-1);
    const cell = row.insertCell(-1);
    cell.colSpan = 6;
    cell.style.backgroundColor = '#323232';
    cell.style.color = '#ffffff';
    cell.textContent = loadError || 'No upcoming weekly forecasts are available.';
    return;
  }

  upcomingFightsTable.rows.item(0).cells.item(0).style.backgroundColor = '#212121';
  for (const index of indices) {
    const fighter = tableValue(vegas_odds, 'fighter name', index);
    const opponent = tableValue(vegas_odds, 'opponent name', index);
    const forecastFighterOdds = firstDisplayValue(
      tableValue(vegas_odds, 'forecast fighter odds', index),
      tableValue(vegas_odds, 'predicted fighter odds', index)
    );
    const forecastOpponentOdds = firstDisplayValue(
      tableValue(vegas_odds, 'forecast opponent odds', index),
      tableValue(vegas_odds, 'predicted opponent odds', index)
    );
    const modelProbability = tableValue(vegas_odds, 'model probability', index);
    const forecastProbability = firstDisplayValue(
      tableValue(vegas_odds, 'forecast probability', index),
      modelProbability
    );
    const marketProbability = tableValue(
      vegas_odds, 'market no-vig fighter probability', index
    );
    const oddsObservedAt = tableValue(vegas_odds, 'odds observed at', index);
    const oddsSource = tableValue(vegas_odds, 'odds source', index);
    const modelStatus = tableValue(vegas_odds, 'model status', index);
    const modelId = tableValue(vegas_odds, 'model id', index);
    let forecastSource = tableValue(vegas_odds, 'forecast source', index);
    if (!hasDisplayValue(forecastSource)) {
      forecastSource = hasDisplayValue(modelId) || hasDisplayValue(modelProbability)
        ? 'point_in_time_model'
        : 'legacy_weekly_prediction';
    }

    const row = tbody.insertRow(-1);
    row.dataset.matchup = 'true';
    for (let columnIndex = 0; columnIndex < 6; columnIndex += 1) {
      row.appendChild(document.createElement('td'));
    }

    const favoredSide = favoredForecastSide(
      forecastProbability, forecastFighterOdds, forecastOpponentOdds
    );
    setFighterCell(row.cells.item(0), fighter, favoredSide === 'fighter');
    setFighterCell(row.cells.item(1), opponent, favoredSide === 'opponent');

    const probabilityNumber = finiteNumber(forecastProbability);
    setCellLines(row.cells.item(2), [
      formatOdds(forecastFighterOdds),
      probabilityNumber === null ? null : formatProbability(probabilityNumber)
    ]);
    setCellLines(row.cells.item(3), [
      formatOdds(forecastOpponentOdds),
      probabilityNumber === null ? null : formatProbability(1 - probabilityNumber)
    ]);

    const marketNumber = finiteNumber(marketProbability);
    setCellLines(row.cells.item(4), marketNumber === null ? ['Not available'] : [
      'Fighter ' + formatProbability(marketNumber),
      'Opponent ' + formatProbability(1 - marketNumber)
    ]);

    const fighterHistory = formatHistoryCount(
      tableValue(vegas_odds, 'fighter prior fights', index)
    );
    const opponentHistory = formatHistoryCount(
      tableValue(vegas_odds, 'opponent prior fights', index)
    );
    const statusLines = [
      'Forecast: ' + humanizeForecastLabel(forecastSource)
    ];
    if (hasDisplayValue(modelStatus)) {
      statusLines.push(humanizeForecastLabel(modelStatus));
    }
    if (hasDisplayValue(modelProbability)) {
      statusLines.push('Independent model: ' + formatProbability(modelProbability));
    }
    if (fighterHistory !== '—' || opponentHistory !== '—') {
      statusLines.push('Prior UFC fights: ' + fighterHistory + ' / ' + opponentHistory);
    }
    if (hasDisplayValue(oddsObservedAt)) {
      statusLines.push(
        'Market observed: ' + String(oddsObservedAt).replace('T', ' ').replace('+00:00', ' UTC')
      );
    }
    if (hasDisplayValue(oddsSource)) {
      statusLines.push('Market source: ' + String(oddsSource));
    }
    statusLines.push('Betting disabled; paper tracking active');
    setCellLines(row.cells.item(5), statusLines);

    for (const cell of row.cells) {
      cell.style.backgroundColor = '#323232';
      cell.style.color = '#ffffff';
    }
    row.cells.item(5).style.fontSize = '12px';
    if (String(modelStatus).toLowerCase() === 'low_history' ||
        String(modelStatus).toLowerCase().startsWith('abstain')) {
      row.cells.item(5).style.color = '#e9b24c';
    }
  }
}



setTimeout(() => { //this builds a table for the history of predictions which is built in python in the jupyter notebook UFC_Prediction_Model
  var numberModelCorrect = 0
  var numberTotal = 0
  var numTotalWithBookieOdds = 0
  var numBookieCorrect = 0
  const historyFighters = prediction_history['fighter name'] || {}
  for (const i in historyFighters) { //iterating over rows of prediction_history
    fighter = prediction_history['fighter name'][i]
    opponent = prediction_history['opponent name'][i]
    fighterOdds = formatOdds(firstDisplayValue(
      tableValue(prediction_history, 'forecast fighter odds', i),
      tableValue(prediction_history, 'predicted fighter odds', i)
    ))
    opponentOdds = formatOdds(firstDisplayValue(
      tableValue(prediction_history, 'forecast opponent odds', i),
      tableValue(prediction_history, 'predicted opponent odds', i)
    ))
    const bettingStatus = tableValue(prediction_history, 'betting status', i)
    const bettingDisabled = hasDisplayValue(bettingStatus) &&
      String(bettingStatus).toLowerCase().startsWith('disabled')

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
      fighterBankrollPercentage = finiteNumber(fighterBankrollPercentageCol[i]) || 0;
    } else {
      fighterBankrollPercentage = 0; // or some default value
    }

    fighterBetCol = prediction_history['fighter bet'] || null; // default to null if not present
    if (fighterBetCol != null) {
      fighterBet = finiteNumber(fighterBetCol[i]) || 0;
    } else {
      fighterBet = 0; // or some default value
    }

    opponentBankrollPercentageCol = prediction_history['opponent bet bankroll percentage'] || null; // default to null if not present
    if (opponentBankrollPercentageCol != null) {
      opponentBankrollPercentage = finiteNumber(opponentBankrollPercentageCol[i]) || 0;
    } else {
      opponentBankrollPercentage = 0; // or some default value
    }

    opponentBetCol = prediction_history['opponent bet'] || null; // default to null if not present
    if (opponentBetCol != null) {
      opponentBet = finiteNumber(opponentBetCol[i]) || 0;
    } else {  
      opponentBet = 0; // or some default value
    }


    bankrollCol = prediction_history['current bankroll after'] || null; // default to null if not present
    const currentBankrollNumber = bankrollCol == null ? null : finiteNumber(bankrollCol[i])
    if (currentBankrollNumber !== null) {
      currentBankroll = currentBankrollNumber.toFixed(2);
    } else {
      currentBankroll = '—';
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

    tr.cells.item(0).textContent = `${fighter} vs ${opponent}`;
    tr.cells.item(1).textContent = `${fighterOdds}, ${opponentOdds}`;

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
    if (bettingDisabled) {
      tr.cells.item(2).textContent = 'No wager';
      tr.cells.item(3).textContent = humanizeForecastLabel(bettingStatus);
    } else {
      setCellLines(tr.cells.item(2), [
        bestBookie || 'No book recorded',
        bestBookie ? `${formatOdds(bestBookieOddsOnFighter)}, ${formatOdds(bestBookieOddsOnOpponent)}` : null
      ]);
      setCellLines(tr.cells.item(3), [
        `${Number(bankrollPercentage).toFixed(2)}% = ${Number(bet).toFixed(2)}$`,
        bettingOn || null
      ]);
    }

    // color the current bankroll based on the result of the bet (green = won, red = lost)
    const currentBankrollText = document.createElement('span');
    currentBankrollText.style.color = bankrollColor;
    currentBankrollText.textContent = currentBankroll;
    tr.cells.item(4).textContent = '';
    tr.cells.item(4).appendChild(currentBankrollText);

    color = 'gold'; //winner color
    fighterWon = false;
    opponentWon = false;
    if (prediction_history['correct?'][i] == 1) {
      numberTotal += 1;
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
      numberTotal += 1;
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
    if (bestFighterBookie && !bettingDisabled) {
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
  var acc = numberTotal > 0 ? numberModelCorrect / numberTotal : null;
  var bookieAcc = numTotalWithBookieOdds > 0 ? numBookieCorrect / numTotalWithBookieOdds : null;
  var accuracy = document.getElementById("myaccuracy")
  // round accuracy to 2 decimal places
  acc = acc === null ? 'N/A' : (Math.round(acc * 10000) / 100).toFixed(2) + '%';
  bookieAcc = bookieAcc === null ? 'N/A' : (Math.round(bookieAcc * 10000) / 100).toFixed(2) + '%';
  console.log(`Bookie accuracy: ${bookieAcc}`)
  // set accuracy text
  accuracy.innerText = `Forecast Accuracy: ${acc}`;
  var bookieAccuracy = document.getElementById("bookieaccuracy")
  bookieAccuracy.innerText = `Bookie Accuracy: ${bookieAcc}`;
}, 1500) //originally 450

//set initial table values and display fight
setTimeout(() => {
  var upcomingFightsTable = document.getElementById('upcoming')
  // TODO get date working again
  const d = new Date();
  let month = d.getMonth();
  let year = d.getFullYear();
  var months = ["January", "February", 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', "November", 'December']
  const firstMatchupRow = upcomingFightsTable.querySelector('tbody tr[data-matchup="true"]')
  document.getElementById('selectMonth_rc').value = months[month]
  document.getElementById('selectYear_rc').value = year
  document.getElementById('selectMonth_bc').value = months[month]
  document.getElementById('selectYear_bc').value = year
  if (firstMatchupRow) {
    const fighterName = firstMatchupRow.cells[0].textContent
    const opponentName = firstMatchupRow.cells[1].textContent
    // Low-history/debut rows may not yet exist in the legacy browser dataset.
    // Do not let that prevent the weekly forecast table from rendering.
    if (fighter_data[fighterName] && fighter_data[opponentName]) {
      selectFighterAndDate(fighterName, 'rc')
      selectFighterAndDate(opponentName, 'bc')
    }
  }
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

