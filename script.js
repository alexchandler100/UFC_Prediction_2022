/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
//your script here.

window.onload = fadeIn;

        function fadeIn() {
            var fade = document.getElementById("body");
            var opacity = 0;
            var intervalID = setInterval(function() {

                if (opacity < 1) {
                    opacity = opacity + 0.1
                    fade.style.opacity = opacity;
                } else {
                    clearInterval(intervalID);
                }
            }, 10);
        }


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

function selectFighter(id, out) {
  selectElement = document.querySelector('#' + id);
  output = selectElement.value;
  document.querySelector('.' + out).textContent = output;
  let i = id[6]
  let j = getRandomInt(4).toString()
  name = selectElement.value;
  var name_encoded = encodeURIComponent(name)
  var name_decoded = decodeURIComponent(name_encoded)
  name_decoded = decodeURIComponent(name_decoded)
  name_decoded = name_decoded.replace(new RegExp(' ', 'g'), '');

  // Calling function
  // set the path to check
  if (checkFileExist("buildingMLModel/images2/" + j + name_decoded + ".jpg")){
    document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images2/" + j + name_decoded + ".jpg" //sets the image
    console.log(j+name_decoded)
  } else {
    document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images/" + j + name_decoded + ".jpg" //sets the image
    console.log(j+name_decoded)
  }
/*
  try {
    console.log(j+name_decoded)
    document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images2/" + j + name_decoded + ".jpg" //sets the image
} catch (error) {
  console.error(error)
  console.log(j+name_decoded)
  document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images/" + j + name_decoded + ".jpg" //sets the image
  // expected output: ReferenceError: nonExistentFunction is not defined
  // Note - error messages will vary depending on browser
}
*/
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
    if (name == fighter && yearDiff < 3 && yearDiff >= 0 && result == 'W') {
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
    if (name == fighter && yearDiff < 6 && yearDiff >= 0 && result == 'L' && method == "KO/TKO") {
      ko_losses += 1
      //console.log(ufcfightscrap[fight]['fighter'],ufcfightscrap[fight]['opponent'], ufcfightscrap[fight]['date'], result, method)
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
    if (name == fighter && yearDiff < 6 && yearDiff >= 0 && result == 'W' && method == "SUB") {
      sub_wins += 1
      //console.log(ufcfightscrap[fight]['fighter'],ufcfightscrap[fight]['opponent'], ufcfightscrap[fight]['date'], result, method)
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
      summ += parseInt(ufcfightscrap[fight][stat])
      let round = parseInt(ufcfightscrap[fight]['round'])
      let minutes = parseInt(ufcfightscrap[fight]['time'][0])
      let seconds = parseInt(ufcfightscrap[fight]['time'].slice(2))
      time_in_octagon += (round - 1) * 5 + minutes + seconds / 60
    }
  }
  //console.log(stat,fighter,inf_abs,year,summ,time_in_octagon)
  return summ / time_in_octagon
}

function onlyUnique(value, index, self) {
  return self.indexOf(value) === index;
}

function wins_wins(fighter,year,years){
  let relevant_fights = []
  for (let i=0;i<ufc_wins_list.length;i++){
    let yearDiff = parseInt(year) - ufc_wins_list[i][2].slice(-4)
    if (yearDiff>years){
      break
    } else {
      relevant_fights.push(ufc_wins_list[i])
    }
  }
  let fighter_wins = []
  for (let i=0;i<relevant_fights.length;i++){
    if (relevant_fights[i][0]==fighter){
      fighter_wins.push(relevant_fights[i][1])
    }
  }
  let fighter_wins_wins=[]
  for (let i=0;i<relevant_fights.length;i++){
    if (fighter_wins.includes(relevant_fights[i][0])){
      fighter_wins_wins.push(relevant_fights[i][1])
    }
  }
  let relevant_wins = fighter_wins.concat(fighter_wins_wins);
  relevant_wins = relevant_wins.filter(onlyUnique);
  //console.log(`${fighter} wins: ${relevant_wins}`)
  return relevant_wins
}

function losses_losses(fighter,year,years){
  let relevant_fights = []
  for (let i=0;i<ufc_wins_list.length;i++){
    let yearDiff = parseInt(year) - ufc_wins_list[i][2].slice(-4)
    if (yearDiff>years){
      break
    } else {
      relevant_fights.push(ufc_wins_list[i])
    }
  }
  let fighter_losses = []
  for (let i=0;i<relevant_fights.length;i++){
    if (relevant_fights[i][1]==fighter){
      fighter_losses.push(relevant_fights[i][0])
    }
  }
  let fighter_losses_losses=[]
  for (let i=0;i<relevant_fights.length;i++){
    if (fighter_losses.includes(relevant_fights[i][1])){
      fighter_losses_losses.push(relevant_fights[i][0])
    }
  }
  let relevant_losses = fighter_losses.concat(fighter_losses_losses);
  relevant_losses = relevant_losses.filter(onlyUnique);
  //console.log(`${fighter} losses: ${relevant_losses}`)
  return relevant_losses
}

//this does not incorporate year for both fighters correctly...
function fight_math(fighter,opponent,year,years){
  let relevant_fights = []
  for (let i=0;i<ufc_wins_list.length;i++){
    let yearDiff = parseInt(year) - ufc_wins_list[i][2].slice(-4)
    if (yearDiff>years){
      break
    } else {
      relevant_fights.push(ufc_wins_list[i])
    }
  }
  let relevant_wins=wins_wins(fighter,year,years)
  relevant_wins.push(fighter)
  let fight_math_wins = []
  for (let i=0;i<relevant_fights.length;i++){
    if (relevant_wins.includes(relevant_fights[i][0]) && relevant_fights[i][1]==opponent){
      fight_math_wins.push(relevant_fights[i])
    }
  }
  return fight_math_wins.length
}

function fight_math_diff(fighter,opponent,year1, year2,years){
  return fight_math(fighter,opponent,year1,years)-fight_math(opponent,fighter,year2,years)
}

function fighter_score(fighter, year, years){
  let relevant_wins=wins_wins(fighter,year,years)
  let relevant_losses=losses_losses(fighter,year,years)
  return relevant_wins.length - relevant_losses.length
}

function fighter_score_diff(fighter,opponent,year1,year2,years){
  return fighter_score(fighter, year1, years) - fighter_score(opponent, year2, years)
}

function avg_count_diff(stat, fighter, opponent, inf_abs, year){
  if (isNaN(avg_count(stat, fighter, inf_abs, year)) || isNaN(avg_count(stat, opponent, inf_abs, year)) ){
    return 0
  }
  return avg_count(stat, fighter, inf_abs, year)-avg_count(stat, opponent, inf_abs, year)
}

function predictionTuple(fighter1, fighter2, month1, year1, month2, year2) {
  let result;
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  mon1 = document.querySelector('#' + month1).value;
  mon2 = document.querySelector('#' + month2).value;
  yr1 = document.querySelector('#' + year1).value;
  yr2 = document.querySelector('#' + year2).value;
  let fighter_score_diff_4 = fighter_score_diff(guy1,guy2, yr1,yr2,4).toFixed(2)
  let fighter_score_diff_9 = fighter_score_diff(guy1,guy2, yr1,yr2,9).toFixed(2)
  let fighter_score_diff_15 = fighter_score_diff(guy1,guy2, yr1,yr2,15).toFixed(2)
  let fight_math_1 = fight_math_diff(guy1,guy2, yr1,yr2,1).toFixed(2)
  let fight_math_6 = fight_math_diff(guy1,guy2, yr1,yr2,6).toFixed(2)
  let l5y_sub_wins_diff = l5y_sub_wins(guy1, yr1).toFixed(2) - l5y_sub_wins(guy2, yr2).toFixed(2)
  let l5y_losses_diff = l5y_losses(guy1, yr1).toFixed(2) - l5y_losses(guy2, yr2).toFixed(2)
  let l5y_ko_losses_diff = l5y_ko_losses(guy1, yr1).toFixed(2) - l5y_ko_losses(guy2, yr2).toFixed(2)
  let age_diff = fighter_age(guy1, yr1).toFixed(2) - fighter_age(guy2, yr2).toFixed(2)
  let av_total_strikes_diff = avg_count_diff('total_strikes_landed', guy1,guy2, 'abs', yr1).toFixed(2)
  let av_abs_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'abs', yr1).toFixed(2)
  let av_inf_gr_strikes = avg_count_diff('ground_strikes_landed', guy1, guy2, 'inf', yr1).toFixed(2)
  let av_tk_atmps_diff = avg_count_diff('takedowns_attempts', guy1, guy2, 'inf', yr1).toFixed(2)
  let av_inf_head_strikes_diff = avg_count_diff('head_strikes_landed', guy1, guy2, 'inf', yr1).toFixed(2)
  result = [fighter_score_diff_4,fighter_score_diff_9,fighter_score_diff_15,fight_math_1,fight_math_6,
    l5y_sub_wins_diff, l5y_losses_diff, l5y_ko_losses_diff, age_diff, av_total_strikes_diff, av_abs_head_strikes_diff,
    av_inf_gr_strikes, av_tk_atmps_diff, av_inf_head_strikes_diff]
  console.log(`prediction tuple: ${result}`)
  console.log(`coefficients: ${Object.values(theta)}`)
  console.log(`intercept: ${Object.values(intercept)}`)


  return result;
}

theta = {};
intercept = {};

$.getJSON('buildingMLModel/theta.json', function(data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function(i, f) {
    theta[i] = f.toFixed(2)
    //console.log(theta[i])
  });
});

$.getJSON('buildingMLModel/intercept.json', function(data) {
  //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
  $.each(data, function(i, f) {
    intercept[i] = f.toFixed(2)
    //console.log(intercept[i])
  });
});


//console.log(theta[0])
//console.log(intercept[0])

function presigmoid_value(fighter1, fighter2, month1, year1, month2, year2) {
  let value = 0
  tup = predictionTuple(fighter1, fighter2, month1, year1, month2, year2);
  for (let i = 0; i < tup.length; i++) {
    value += tup[i] * theta[i]
  }
  console.log(`value and intercept: ${value} ${intercept[0]}`)
  console.log(`presigmoid_value: ${value + intercept[0]}`)
  //return value + intercept[0]
  return value
}

function predict(fighter1, fighter2, month1, year1, month2, year2) {
  let value = presigmoid_value(fighter1, fighter2, month1, year1, month2, year2)
  if (value>0){
    console.log('first guy wins')
  } else {
    console.log('second guy wins')
  }
  let winner;
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  if (value >= 0) {
    winner = guy1
  } else {
    winner = guy2
  }
  console.log(winner)
  console.log(`value: ${value}`)
  console.log(`before abs`)
  let abs_value = (Math.abs(value))
  console.log(`the value is ${abs_value}`)
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
  console.log(resulting_text)
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
  myTab.rows.item(1).cells.item(0).innerHTML = fighter_data[fighter]['height'];
  myTab.rows.item(1).cells.item(1).innerHTML = fighter_data[fighter]['reach'];
  myTab.rows.item(1).cells.item(2).innerHTML = fighter_age(fighter, yr)
  myTab.rows.item(1).cells.item(3).innerHTML = fighter_data[fighter]['stance'];
  //document.querySelector('.tableEntry').textContent = fighter
}

function populateLast5Fights(fighter, corner) {
  var myTab;
  if (corner == 'rc') {
    yr = document.querySelector('#' + 'f1selectyear').value;
    myTab = document.getElementById("l5ytable1");
  } else if (corner == 'bc') {
    yr = document.querySelector('#' + 'f2selectyear').value;
    myTab = document.getElementById("l5ytable2");
  }
  for (numb=1;numb<6;numb++) {
      myTab.rows.item(numb).cells.item(0).innerHTML = ''
      myTab.rows.item(numb).cells.item(1).innerHTML = ''
      myTab.rows.item(numb).cells.item(2).innerHTML = ''
      myTab.rows.item(numb).cells.item(3).innerHTML = ''
      myTab.rows.item(numb).cells.item(1).style.backgroundColor = "#ffffff";
      }
  let fightNumber = 0
  for (const fight in ufcfightscrap) {
    let result;
    let opponent;
    let method;
    let yearDiff = parseInt(yr) - ufcfightscrap[fight]['date'].slice(-4)
    if (ufcfightscrap[fight]['fighter'] == fighter && yearDiff >= 0) {
      result = ufcfightscrap[fight]['result']
      opponent = ufcfightscrap[fight]['opponent']
      method = ufcfightscrap[fight]['method']
      date = ufcfightscrap[fight]['date']
      fightNumber += 1
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
    if (fightNumber > 4) {
      break
    }
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

//set initial table values
setTimeout(() => {
  ufc_wins_list=[]
  for (const fight in ufcfightscrap) {
    if (ufcfightscrap[fight]['result']=="W") {
      let fighter = ufcfightscrap[fight]['fighter']
      let opponent = ufcfightscrap[fight]['opponent']
      let date = ufcfightscrap[fight]['date']
      ufc_wins_list.push([fighter,opponent,date])
    }
  }
  document.getElementById('select1').value = "Khamzat Chimaev"
  document.getElementById('select2').value = "Gilbert Burns"
  document.getElementById('f1selectmonth').value = "April"
  document.getElementById('f1selectyear').value = "2022"
  document.getElementById('f2selectmonth').value = "April"
  document.getElementById('f2selectyear').value = "2022"
  selectFighterAndDate('select1','name1','f1selectmonth','month1','f1selectyear','year1')
  selectFighterAndDate('select2','name2','f2selectmonth','month2','f2selectyear','year2')
  fighter_score('Petr Yan', '2022', 4)
  fighter_score('Aljamain Sterling', '2022', 4)
  //populateTaleOfTheTape('Khabib Nurmagomedov', 'rc')
  //populateTaleOfTheTape('Colby Covington', 'bc')
  //populateLast5Fights('Khabib Nurmagomedov', 'rc')
  //populateLast5Fights('Colby Covington', 'bc')
}, 300)
