
//////////////////////////////// Helper Functions ////////////////////////////////

// test url: https://kyblab.tuebingen.mpg.de/seq/simonsays/?PROLIFIC_PID={{AGES}}&STUDY_ID={{TEST}}&SESSION_ID={{AAA}}


function prolificGetParam(name) {
    console.log('trying to get prolific parameter')
    var regexS = "[\?&]" + name + "=([^&#]*)";
    var regex = new RegExp(regexS);
    // var tmpURL = fullurl;
    var tmpURL = document.location.href;

    var results = regex.exec(tmpURL);
    console.log('result is : ', results);
    if (results == null) {
        return "";
    } else {
        return results[1];
    }

}

// function prolificGetParam(variable)
// {
//     var query = window.location.search.substring(1);
//     var vars = query.split("&");
//     for (var i=0;i<vars.length;i++) {
//         var pair = vars[i].split("=");
//         if(pair[0] == variable){return pair[1];}
//     }
//     return(false);
// }






//function to draw the letter boxes into the HTML
function drawletter(l) {
    console.log(l)
    change('letter', l);
}

function drawkey(k){
// l = letter + k + pspecs + border;
// drawletter(l);
    ['S','D','F','J','K','L','E'].forEach(x => document.getElementById(x).style.display = 'none');
    document.getElementById(k).style.display = 'block';
    window.scrollTo(0, 0);
}


function change(x, y) //changes inner HTML of div with ID=x to y
{
    document.getElementById(x).innerHTML = y;
}


function loadkey(k){
    document.getElementById('letter').innerHTML = '<input type="image" src="letters/' + k + '.png"  width="800" height="400" border="0">';
}


function showkey(k){
    document.getElementById(hide).style.display = 'none';
    document.getElementById(show).style.display = 'block';
    window.scrollTo(0, 0);

}

function clickStart(hide, show) // hide one html div and show another by changing the display style
{
    document.getElementById(hide).style.display = 'none';
    document.getElementById(show).style.display = 'block';
    window.scrollTo(0, 0);
}

function hide(x) //Hides id=x
{
    document.getElementById(x).style.display = 'none';
}

function show(x) //shows div with id=x
{
    document.getElementById(x).style.display = 'block';
    window.scrollTo(0, 0);
}

function randomNum(min, max) {//this function is inclusive of its both boundaries.
    return Math.floor(Math.random() * (max - min + 1) + min)
}


function permute(array) {
    //permute a list
    var currentIndex = array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;
        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }
    return array;
}




