/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
//your script here.

$(function() {
  //var people = [];
  $.getJSON('buildingMLModel/fighter_stats.json', function(data) {
    //for each input (i,f), i is the key (a fighter's name) and f is the value (all their data)
    $.each(data, function(i, f) {
      //create entry in local object
      const select = document.getElementById('fighters')
      const value = i
      const label = i

      select.insertAdjacentHTML('beforeend', `
  <option value="${value}">${label}</option>
`)
    });
  });
});

function pickRand(a, b) {
  let x = Math.random();
  if (x < .5) {
    return a
  } else {
    return b
  }
}

function getOption(id,out) {
  selectElement = document.querySelector('#'+id);
  output = selectElement.value;
  document.querySelector('.'+out).textContent = output;
  i=id[6]
  console.log(i)
  name = selectElement.value;
  name = name.replace(" ","")
  console.log("buildingMLModel/images/"+i+name+".jpg")
  //sets the image to be the image of the fighter
  document.getElementById("fighter"+i+"pic").src="buildingMLModel/images/"+i+name+".jpg"
}

// i='1' or '2' (1 for fighter1 2 for fighter2)
function setFighterImage(i){
  selectElement = document.querySelector('#'+i);
  name = selectElement.value;
  name = name.replace(" ","")
  //sets the image to be the image of the fighter
  document.getElementById("fighter"+i+"pic").src="buildingMLModel/images/"+i+name+".jpg"
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
