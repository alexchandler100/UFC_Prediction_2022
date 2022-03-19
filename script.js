/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
//your script here.

fighter_data = {}
fight_history = {}

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

function selectFighter(id, out) {
  selectElement = document.querySelector('#' + id);
  output = selectElement.value;
  document.querySelector('.' + out).textContent = output;
  i = id[6]
  name = selectElement.value;
  console.log(fighter_data[name])
  name = name.replace(" ", "")
  //sets the image to be the image of the fighter
  document.getElementById("fighter" + i + "pic").src = "buildingMLModel/images/" + i + name + ".jpg"
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

function fighter_age(fighter,yearSelected) {
  let yearBorn = fighter_data[fighter]['dob'].slice(-4)
  return parseInt(yearSelected)-parseInt(yearBorn)
}

function fighter_reach(fighter,yearSelected) {
  let reach = fighter_data[fighter]['reach'].slice(0, -1) //this removes the last character "
  return parseInt(reach)
}

function predictionTuple(fighter1, fighter2, month1, year1, month2, year2) {
  guy1 = document.querySelector('#' + fighter1).value;
  guy2 = document.querySelector('#' + fighter2).value;
  mon1 = document.querySelector('#' + month1).value;
  mon2 = document.querySelector('#' + month2).value;
  yr1 = document.querySelector('#' + year1).value;
  yr2 = document.querySelector('#' + year2).value;
  let age1=fighter_age(guy1,yr1)
  let age2=fighter_age(guy2,yr2)
  let reachdiff = fighter_reach(guy1)-fighter_reach(guy2)
  //day1 = `${month1} 1, ${year1}`
  //day2 = `${month2} 1, ${year2}`
  /*
  return [fighter_age(fighter1,day1),
            fighter_age(fighter2,day2),
            fighter_reach(fighter1)-fighter_reach(fighter2),
            L5Y_ko_losses(fighter1,day1)-L5Y_ko_losses(fighter2,day2),
            L5Y_wins(fighter1,day1)-L5Y_wins(fighter2,day2),
            L5Y_losses(fighter1,day1)-L5Y_losses(fighter2,day2),
            avg_count('total_strikes_landed',fighter1,'abs',day1)-avg_count('total_strikes_landed',fighter2,'abs',day2),
            avg_count('takedowns_attempts',fighter1,'inf',day1)-avg_count('takedowns_attempts',fighter2,'inf',day2),
            avg_count('ground_strikes_landed',fighter1,'abs',day1)-avg_count('ground_strikes_landed',fighter2,'abs',day2)
        ]
        */
  //return fighter_data[name1]
  return [age1,age2,reachdiff]
}

function predict(fighter1, fighter2, month1, year1, month2, year2) {
  tup = predictionTuple(fighter1, fighter2, month1, year1, month2, year2)
  console.log(tup)
  return tup
}


function myFunction1() {
  document.getElementById("myDropdown1").classList.toggle("show");
}

function myFunction2() {
  document.getElementById("myDropdown2").classList.toggle("show");
}

function filterFunction1() {
  console.log('calling filter function')
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
  console.log('calling filter function')
  var input, filter, ul, li, a, i;
  input = document.getElementById("myInput2");
  filter = input.value.toUpperCase();
  div = document.getElementById("myDropdown2");
  console.log(div)
  a = div.getElementsByTagName("a");
  console.log(a[i].textContent)
  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}
