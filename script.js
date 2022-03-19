/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
//your script here.

fighter_data = {}
ufcfightscrap = {}

$(function() {
  //var people = [];
  $.getJSON('buildingMLModel/fighter_stats.json', function(data) {
    //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
    $.each(data, function(i, f) {
      //create entry in local object
      const select = document.getElementById('fighters')
      select.insertAdjacentHTML('beforeend', `
  <option value="${i}">${i}</option>
`)
      fighter_data[i] = f
    });
  });
});

$(function() {
  //var people = [];
  $.getJSON('buildingMLModel/ufcfightscrap.json', function(data) {
    //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
    $.each(data, function(i, f) {
      //create entry in local object
      ufcfightscrap[i] = f
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

function pickRand(a, b) {
  let x = Math.random();
  if (x < .5) {
    return a
  } else {
    return b
  }
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max)+1;
}

function selectFighter(id, out) {
  selectElement = document.querySelector('#' + id);
  output = selectElement.value;
  document.querySelector('.' + out).textContent = output;
  let i = id[6]
  let j=getRandomInt(4).toString()
  name = selectElement.value;
  //console.log(fighter_data[name])
  name = name.replace(" ", "")
  console.log("buildingMLModel/images/" + j + name + ".jpg")
  //sets the image to be the image of the fighter
  document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images/" + j + name + ".jpg"
}

// i='1' or '2' (1 for fighter1 2 for fighter2)
function setFighterImage(i) {
  selectElement = document.querySelector('#' + i);
  name = selectElement.value;
  name = name.replace(" ", "")
  //sets the image to be the image of the fighter
  document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images/" + i + name + ".jpg"
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

function fighter_age(fighter, yearSelected) {
  let yearBorn = fighter_data[fighter]['dob'].slice(-4)
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
    if (name == fighter && yearDiff < 6 && yearDiff >= 0 && result == 'W') {
      wins += 1
      //console.log(ufcfightscrap[fight]['fighter'],ufcfightscrap[fight]['opponent'], ufcfightscrap[fight]['date'], result)
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
    if (name == fighter && yearDiff < 6 && yearDiff >= 0 && result == 'L' && method == "KO/TKO") {
      ko_losses += 1
      //console.log(ufcfightscrap[fight]['fighter'],ufcfightscrap[fight]['opponent'], ufcfightscrap[fight]['date'], result, method)
    }
  }
  return ko_losses
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
    if (name == fighter && yearDiff < 6 && yearDiff >= 0 && result == 'L') {
      losses += 1
      //console.log(ufcfightscrap[fight]['fighter'], ufcfightscrap[fight]['opponent'], ufcfightscrap[fight]['date'], result)
    }
  }
  return losses
}

//avg_count('total_strikes_landed',fighter1,'abs',day1)
function avg_count(stat, fighter, inf_abs, year) {
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
    if (name == fighter && yearDiff >= 0) {
      //console.log(`adding: ${ufcfightscrap[fight][stat]}`)
      summ += parseInt(ufcfightscrap[fight][stat])
      let round = parseInt(ufcfightscrap[fight]['round'])
      let minutes = parseInt(ufcfightscrap[fight]['time'][0])
      let seconds = parseInt(ufcfightscrap[fight]['time'].slice(2))
      time_in_octagon += (round - 1) * 5 + minutes + seconds / 60
      //console.log(t)
    }
  }
  //console.log(stat,fighter,inf_abs,year,summ,time_in_octagon)
  return summ / time_in_octagon
}



function predictionTuple(fighter1, fighter2, month1, year1, month2, year2) {
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  mon1 = document.querySelector('#' + month1).value;
  mon2 = document.querySelector('#' + month2).value;
  yr1 = document.querySelector('#' + year1).value;
  yr2 = document.querySelector('#' + year2).value;
  let age1 = fighter_age(guy1, yr1)
  let age2 = fighter_age(guy2, yr2)
  let reachdiff = (fighter_reach(guy1) - fighter_reach(guy2)) * 2.54
  let l5y_ko_losses_diff = l5y_ko_losses(guy1, yr1) - l5y_ko_losses(guy2, yr2)
  let l5y_wins_diff = l5y_wins(guy1, yr1) - l5y_wins(guy2, yr2)
  let l5y_losses_diff = l5y_losses(guy1, yr1) - l5y_losses(guy2, yr2)
  let av_total_strikes_diff = avg_count('total_strikes_landed', guy1, 'abs', yr1) - avg_count('total_strikes_landed', guy2, 'abs', yr2)
  let av_tk_atmps_diff = avg_count('takedowns_attempts', guy1, 'inf', yr1) - avg_count('takedowns_attempts', guy2, 'inf', yr2)
  let av_gr_strikes = avg_count('ground_strikes_landed', guy1, 'abs', yr1) - avg_count('ground_strikes_landed', guy2, 'abs', yr2)
  return [age1, age2, reachdiff, l5y_ko_losses_diff, l5y_wins_diff, l5y_losses_diff, av_total_strikes_diff, av_tk_atmps_diff, av_gr_strikes]
}

let theta = [-0.03429425, -0.03429425, 0.0186947, -0.10857499, 0.10907987, -0.04910427,
  -0.07615219, 0.48450442, -0.00705265
]

let intercept = 2.06996272

function presigmoid_value(fighter1, fighter2, month1, year1, month2, year2) {
  let value = 0
  tup = predictionTuple(fighter1, fighter2, month1, year1, month2, year2);
  for (let i = 0; i < tup.length; i++) {
    value += tup[i] * theta[i]
  }
  return value + intercept
}

function predict(fighter1, fighter2, month1, year1, month2, year2) {
  let value = presigmoid_value(fighter1, fighter2, month1, year1, month2, year2)
  let value2 = presigmoid_value(fighter2, fighter1, month2, year2, month1, year1)
  console.log(value, value2)
  let winner;
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  if (value - value2 >= 0) {
    winner = guy1
  } else {
    winner = guy2
  }
  let abs_value = (Math.abs(value) + Math.abs(value2)) / 2
  let resulting_text;
  if (abs_value >= 0 && abs_value <= .2) {
    resulting_text = winner + " wins a little over 5 out of 10 times."
  } else if (abs_value >= .2 && abs_value <= .4) {
    resulting_text = (winner + " wins 6 out of 10 times.")
  } else if (abs_value >= .4 && abs_value <= .6) {
    resulting_text = (winner + " wins 7 out of 10 times.")
  } else if (abs_value >= .6 && abs_value <= .8) {
    resulting_text = (winner + " wins 9 out of 10 times.")
  } else if (abs_value >= .8) {
    resulting_text = (winner + " wins 10 out of 10 times.")
  }
  document.querySelector('.fightoutcome').textContent = resulting_text
}


function myFunction1() {
  document.getElementById("myDropdown1").classList.toggle("show");
}

function myFunction2() {
  document.getElementById("myDropdown2").classList.toggle("show");
}

function filterFunction1() {
  //console.log('calling filter function')
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
  //console.log('calling filter function')
  var input, filter, ul, li, a, i;
  input = document.getElementById("myInput2");
  filter = input.value.toUpperCase();
  div = document.getElementById("myDropdown2");
  //console.log(div)
  a = div.getElementsByTagName("a");
  //console.log(a[i].textContent)
  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}
