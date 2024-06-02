//TODO: tell people how long the sequence is going to be? A count down in the sequence being shown
//local url for the purpose of local testing: 
//file:///Users/swu/Desktop/research/motif_learning/Experiments/simonsays/index.html?PROLIFIC_PID={{AGES}}&STUDY_ID={{TEST}}&SESSION_ID={{AAA}}

//data recording
//reward calculation + score calculation recheck
//chi's position changed
//Performance and Reward does not work show up
//Instruction Page
//Cues for chunk of four
//Flickering, try adjusting html value
//XYZ instead of XY
//Test Trial
//Style Adjustment
//Shorten Trial Size
//combine page 1 and 2 into the same page 



n_train_trial = 40;
n_test_trial = 8//8;
seqlength= 12;// sequence length
var train_data,test_data,block_structure, condition_structure;
var inst; // instruction sequence. 
block_structure = ['train', 'test', 'test','test'];

var blockcounter = 0; 
var inst_counter = 1; 
var trial_counter = 1;
var returnpressed = 0;

conditioncollect = [];// same length as n_trial
blockcollect = [];
correctcollect = [];
timecollect = [];
recallcollect = [];// [[A,B,C,C,B,A], [AA]]
trialcollect = [];//[1,2,...] trial number
instructioncollect = [];


var trialinstruction = [];
var trialrecall = [];
var trialrt = [];
var block, condi;

var timeInMs = 0;

var age;
var gender;  
var bonus_max = 4;
maxtrialbonus = bonus_max/(n_train_trial + n_test_trial*3);

var overallbonus = 0;
var reward = 0;
base_pay = 4;
var presenttotal,money;
var workerID,studyID;

lightuptime = 800;
lightdowntime = 400;
lightresponsetime = 150;
//letter configuration
letter = '<input type="image" src="./letters/';
pspecs = '.png"  width="fit-content" height="fit-content"';
var border = 'border="0" align="center">';

speaking = '<input type="image" src="say.png" width="300" height="200" align="center">';//  
listening = '<input type="image" src="listen.png" width="230" height="300" align = "center">';
noclub = '<p><span style="font-size:200%;color:black;"></span></p>';
oneclub = '<p><span style="font-size:200%;color:black;"> &#128062;</span></p>';
twoclubs = '<p><span style="font-size:200%;color:black;"> &#128062; &#128062; </span></p>';
threeclubs = '<p><span style="font-size:200%;color:black;"> &#128062; &#128062; &#128062;</span></p>';
demo_trial = ['J','K','K','J', 'E', 'K','J','J','K','E','J','K','K','K'];
demo_trial_compact = ['J','K','K','J','K','J','J','K','J','K','K','K'];
var trialrecalldemo = [];
var demo_counter = -1;

//////////////////////////////// Experiment Relevant Functions ////////////////////////////////

function checkconfigure() {
    // get subject ID
    if (window.location.search.indexOf('PROLIFIC_PID') > -1) {
      workerID = prolificGetParam('PROLIFIC_PID');
    }
    // If no ID is present, generate one using random numbers - this is useful for testing
    else {
      workerID = 'test-' + Math.floor(Math.random() * (2000000 - 0 + 1)) + 0; 
    }

    // STUDY ID
    if (window.location.search.indexOf('STUDY_ID') > -1) {
        studyID = prolificGetParam('STUDY_ID');
    }
    else 
    { studyID = 'data';}

    console.log('workerID', workerID);
    console.log('studyID', studyID);

    saveDataArray = {
        'prolificID': workerID,
        'studyID': studyID,
    };
    //PHPsuccess = Momchil_datasave(saveDataArray);
    clickStart('page1', 'page2');

}


function Momchil_datasave(thisdata) {
    file_name = '../data/' + 'prolificID= ' + workerID + ' studyID=' + studyID + ' reward=' + reward + '.txt';
    $.post("results_data.php", {
        postresult: JSON.stringify(thisdata),
        postfile: file_name
    });
    return true;
}



//randomly assign test block structure for each participant 
function assigntestblock(){
teststructure = permute(['ind','m1','m2']);
	return teststructure;
}

function assigncondition(){
condition = ['ind','m2'][randomNum(0,1)];

//condition = ['ind','m1','m2'][randomNum(0,2)];
return condition
}

function assigndata(condition, teststructure){
	//also need to mark the condition and the block structure
	test_data = [];
	train_data =[];
	condition_structure = [];
	idx = randomNum(0, ind_train.length-1);

	// assign the data of the train block and the test blocks 
	if (condition == 'ind'){
		train_data = ind_train[idx].slice(0,n_train_trial);
		condition_structure.push('ind');
	}
	if (condition == 'm1'){
		train_data = m1_train[idx].slice(0,n_train_trial);
		condition_structure.push('m1');
	}
	if (condition == 'm2'){
		train_data = m2_train[idx].slice(0,n_train_trial);
		condition_structure.push('m2');
	}
	for(var i = 0; i < 3; i++){
	// assign the data of the train block and the test blocks 
	condition_structure.push(teststructure[i]);
	if (teststructure[i] == 'ind'){
		test_data.push(ind_test[idx].slice(0,n_test_trial));}
	if (teststructure[i] == 'm1'){
		test_data.push(m1_test[idx].slice(0,n_test_trial));}
	if (teststructure[i] == 'm2'){
		test_data.push(m2_test[idx].slice(0,n_test_trial));}}
}

//Retrieve worker and assignment id from URL header, and then assigns them a scenario
function beginExperiment() {
	condition = assigncondition();
	teststructure = assigntestblock();
	assigndata(condition, teststructure);
	block = 'train';
	condi = condition[0];

    keyassignment = permute(['S', 'D', 'F', 'J', 'K', 'L']);
    keyassignment = ['E'].concat(keyassignment);

    alert('Great, you have answered all of the questions correctly. The study will now start.');
    clickStart('page5', 'pagenormal');

	returnpressed = 2;
	document.addEventListener('keydown', function (event) {
	  if (event.key === ' ' && returnpressed == 2) {
    	returnpressed = 1;
	    console.log('check sum ')
    	nextblock();
	  }
	});

}

//////// show sequence of keys /////////f

function showempty(){//shows dark for 500ms
	drawkey('E');
    setTimeout(function() {
    	begintrial();
	}, lightdowntime);
}
function get_next_key(inst_counter) {
    keyindex = inst[trial_counter-1][inst_counter-1];// 0 is the subject number inst_counter is the trial number 
    this_key = keyassignment[keyindex];
    return this_key}

//display the letter one after another, if there is an empty letter, then it is translated to a pause. 
// insert before: triallength = train_data[subj][trial_counter-1].length
function begintrial(){
	console.log(inst_counter)
    if (inst_counter > inst[trial_counter-1].length){
    	console.log('end of sequence display, into recalltrial');
    	change('cue', noclub);
    	change('Chi', listening);
	    change('remain', "What did the kitty say? ");
		recalltrial();} // exit here
	else{
	    key = get_next_key(inst_counter);
	    if (key!='E'){
	    trialinstruction.push(key);//record to instruction sequence
	   	}
	    drawkey(key);
		setTimeout(function() {
			if (inst_counter == 6){change('cue', twoclubs);}
			if (inst_counter == 11){change('cue', threeclubs);}
			showempty()//go to showempty after 1s sp
		}, lightuptime);
		inst_counter++;}
}

function nexttrial(){
	change('cue', noclub);
	drawkey('E');

	clickStart('pagetrialfeedback','keypresspage');	
    setTimeout(function() {
    	change('cue', oneclub);
        setTimeout(function() {
        	console.log('initiate begintrial')
            begintrial(); }, lightdowntime);
	}, lightresponsetime);
}

function tryout(){
	change('Chi', speaking);
	change('remain', "The kitty is showing you a sequence. ");
	drawkey('E');
	clickStart('page4b','keypresspage');
    setTimeout(function() {
    	change('cue', oneclub);
        setTimeout(function() {
            begindemo();}, lightdowntime);
	}, lightresponsetime);
}

function begindemo(){
    if (demo_counter == demo_trial.length-1){
    	change('cue', noclub);
    	console.log('end of sequence display, into recalltrial');
    	change('Chi',listening);
	    change('remain', "What did the kitty say? ");
		recalldemo();} // exit here
	else{
	demo_counter++;
    key = demo_trial[demo_counter];
    drawkey(key);
	setTimeout(function() {
		if (demo_counter == 4){change('cue', twoclubs);}
		if (demo_counter == 9){change('cue', threeclubs);}
		drawkey('E');
	    setTimeout(function() {
	    	begindemo();
		}, lightdowntime);
	}, lightuptime);
}
}



//////// Recall /////////

function evaluate_accuracy(subject_response, displayed_instruction){
	total = 0; 
	// instruction_without_empty = [];
	// for(var i = 0; i < displayed_instruction.length; i++){
	// 	if (displayed_instruction[i]!='E'){
	// 		instruction_without_empty.push(displayed_instruction[i])
	// 	}
	// }

	for (var i = 0; i < subject_response.length; i ++){
		if (subject_response[i]==displayed_instruction[i]){
			total++;
		}
	}
	var trial_acc = total / subject_response.length;
	return trial_acc;}


function nextblock() {
	change('Chi', speaking);
	change('remain', "The kitty is showing you a sequence. ");
	change('cue', noclub);

	if (blockcounter == 4){// end of the third block 
		finishexperiment();}
	else{

    if (blockcounter == 0) {inst = train_data;}
    else{inst = test_data[blockcounter - 1];}

	block = block_structure[blockcounter];
	condi = condition_structure[blockcounter];
	inst_counter = 1; //instruction within a trial, innermost loop
	trial_counter = 1;//intermediate loop 
	if (blockcounter ==0){clickStart('pagenormal','keypresspage')}
	if (blockcounter >=1){clickStart('pageblockfeedback','keypresspage')}

	drawkey('E');
    setTimeout(function() {
    	change('cue', oneclub);
        setTimeout(function() {
            begintrial(); }, lightdowntime);
	}, lightresponsetime);

	}
}

function finishexperiment() {
	clickStart('pageblockfeedback', 'page7');
	alert("You have finished the experiment.");
}

function display_trial_acc(trialacc){
	blockcollect.push(block);//['train', 'test', ...]
	conditioncollect.push(condi);//['m1','m2',...]
	trialcollect.push(trial_counter);//[1,2,...] trial number
	correctcollect.push(trialacc);//[0.1,0.2,...]
	timecollect.push(trialrt);//*** 
	recallcollect.push(trialrecall);// [[A,B,C,C,B,A], [AA]]
	instructioncollect.push(trialinstruction);

	var trialspeed = trialrt.reduce((a,b)=>a+b, 0);
	var trialreward = calculate_trial_bonus(trialacc,trialspeed);
	overallbonus = overallbonus + trialreward;
	var score = trialacc*10; 

	var performance_fast = '<p> Your have got '  + Math.floor(trialacc*100) + ' percent correct. Your speed is ' + Math.floor(trialspeed) + ' ms. </p> <p> You have earned ' + Math.floor(trialreward*100) + ' pences in this trial. </p>';
	trialrecall = [];
	trialinstruction = [];
	trialrt = []; 
	inst_counter = 1;
	trial_counter++;
	console.log(trial_counter)
	if (trial_counter> inst.length){// switch to the next block
		blockcounter++; 
		change('feedbackblock', performance_fast);
		clickStart('keypresspage', 'pageblockfeedback');
		returnpressed = 3;
		document.addEventListener('keydown', function (event) {
		  if (event.key === ' ' && returnpressed == 3) {
	    	returnpressed = 1;
	    	nextblock();}});
	}

	else{//proceed to the next trial within the block
		change('feedbacktrial', performance_fast);
		clickStart('keypresspage', 'pagetrialfeedback');
		returnpressed = 4;
		document.addEventListener('keydown', function (event) {
		  if (event.key === ' ' && returnpressed == 4) {
	    	returnpressed = 1;
	    	nexttrial();}});

	 }
}

function recalltrial() //this function initializes a trial
{
	console.log('call recall trial');
	if (trialrecall.length == 0){change('cue', oneclub);}
	if (trialrecall.length == 4){change('cue', twoclubs);}
	if (trialrecall.length == 8){change('cue', threeclubs);}

timeInMs = Date.now();//initialize time count
returnpressed = 0;//get the pressed key
if (trialrecall.length >= seqlength){// detect end of trial    
	trialacc = evaluate_accuracy(trialrecall, trialinstruction);

	console.log(trialacc);
	display_trial_acc(trialacc);// display accuracy for one trial
}
else{

document.addEventListener('keydown', function (event) {
    if (event.key === 's' & returnpressed == 0) {
    	trialrecall.push('S');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
        drawkey('S');
    	setTimeout(function() {
    		drawkey('E');
    		recalltrial();}, lightresponsetime);
    }
    if (event.key == 'd' & returnpressed == 0) {
    	trialrecall.push('D');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('D');
    	setTimeout(function() {
    		drawkey('E');
    		recalltrial();}, lightresponsetime);
    }
    if (event.key == 'f' & returnpressed == 0) {
    	trialrecall.push('F');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('F');
    	setTimeout(function() {
    		drawkey('E');
    		recalltrial();}, lightresponsetime);
    }
    if (event.key == 'j' & returnpressed == 0) {
    	trialrecall.push('J');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('J');
    	setTimeout(function() {
    		drawkey('E');
    		recalltrial();}, lightresponsetime);
    }
    if (event.key == 'k' & returnpressed == 0) {
    	trialrecall.push('K');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('K');
    	setTimeout(function() {
    		drawkey('E');
    		recalltrial();}, lightresponsetime);
    }
    if (event.key == 'l' & returnpressed == 0) {
    	trialrecall.push('L');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
        drawkey('L');
    	setTimeout(function() {
    		drawkey('E');
    		recalltrial();}, lightresponsetime);
    }
	});

}}



function display_demo_acc(trialacc){
	var score = trialacc*10; 
	var performance_fast = "You have scored " + Math.floor(score).toString() + " points on this trial";
	change('feedbackdemo', performance_fast);
	clickStart('keypresspage', 'pagedemofeedback');
}


function recalldemo() //this function initializes a trial
{
	if (trialrecalldemo.length == 0){change('cue', oneclub);}
	if (trialrecalldemo.length == 4){change('cue', twoclubs);}
	if (trialrecalldemo.length == 8){change('cue', threeclubs);}
	timeInMs = Date.now();//initialize time count
	var returnpressed = 5;//get the pressed key
if (trialrecalldemo.length == seqlength){// detect end of trial    
    demoaccuracy = evaluate_accuracy(trialrecalldemo, demo_trial_compact);
    display_demo_acc(demoaccuracy);// display accuracy for one trial
    change('Chi', speaking);
    change('remain', "The kitty is showing you a sequence. ");

}
else{

document.addEventListener('keydown', function (event) {
    if (event.key === 's' & returnpressed == 5) {
    	trialrecalldemo.push('S');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
        drawkey('S');
    	setTimeout(function() {
    		drawkey('E');
    		recalldemo();}, lightresponsetime);
    }
    if (event.key == 'd' & returnpressed == 5) {
    	trialrecalldemo.push('D');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('D');
    	setTimeout(function() {
    		drawkey('E');
    		recalldemo();}, lightresponsetime);
    }
    if (event.key == 'f' & returnpressed == 5) {
    	trialrecalldemo.push('F');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('F');
    	setTimeout(function() {
    		drawkey('E');
    		recalldemo();}, lightresponsetime);
    }
    if (event.key == 'j' & returnpressed == 5) {
    	trialrecalldemo.push('J');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('J');
    	setTimeout(function() {
    		drawkey('E');
    		recalldemo();}, lightresponsetime);
    }
    if (event.key == 'k' & returnpressed == 5) {
    	trialrecalldemo.push('K');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
		drawkey('K');
    	setTimeout(function() {
    		drawkey('E');
    		recalldemo();}, lightresponsetime);
    }
    if (event.key == 'l' & returnpressed == 5) {
    	trialrecalldemo.push('L');
    	returnpressed = 1;
    	timeInMs = Date.now() - timeInMs;
    	trialrt.push(timeInMs);
        drawkey('L');
    	setTimeout(function() {
    		drawkey('E');
    		recalldemo();}, lightresponsetime);
    }
	});

}}


function calculate_trial_bonus(trialacc, trialrt){

	var bonusfast, bonusacc,trialbonus;
	//calculate bonus fast first
	if (trialrt < 2000){
		bonusfast = maxtrialbonus;
	}
	if (trialrt > 10000){
		bonusfast = 0;
	}
	if (trialrt>=2000 && trialrt <= 10000){
		bonusfast = bonus_max - (10000 - trialrt)/(10000-2000)*maxtrialbonus;
	}

	if (trialacc >=1){
		bonusacc = maxtrialbonus;
	}
	if (trialacc <= 0.5){
		bonusacc = 0;
	}
	if (trialacc > 0.5 && trialacc <1){
		bonusacc = bonus_max * (trialacc - 0.5)/(1-0.5);
	}

	trialbonus = 0.5*bonusfast + 0.5*bonusacc
	console.log(bonusfast, bonusacc, trialbonus, maxtrialbonus)

    if (trialbonus > maxtrialbonus) {
        trialbonus = maxtrialbonus;
    }
    if (trialbonus < 0) {
        trialbonus = 0;
    }

	return trialbonus
}



//Demographics & Finish
function setgender(x) {
    gender = x;
    return (gender)
}

//sets the selected age
function setage(x) {
    age = x;
    return (age)
}

function setrecontact(x) {
    recontact = x;
    return (recontact)
}

function instructioncheck() {
    //check if correct answers are provided
    if (document.getElementById('icheck1').checked) {
        var ch1 = 1
    }
    if (document.getElementById('icheck2').checked) {
        var ch2 = 1
    }
    if (document.getElementById('icheck3').checked) {
        var ch3 = 1
    }
    //are all of the correct
    var checksum = ch1 + ch2 + ch3;
    if (checksum === 3) {
        //if correct, continue
        beginExperiment();
        return;
    } else {
        //if one or more answers are wrong, raise alert box
        alert('You have answered some of the questions wrong. Please try again.');
        //go back to instructions
        clickStart('page5', 'page1');
        return;
    }
}

function mysubmit() {

    //calculate money earned
    money = 'You will be paid £ ' + Number.parseFloat(base_pay).toPrecision(1) + ' pounds for your participation, and a bonus of £ ' + Number.parseFloat(overallbonus).toPrecision(3) + ' based on your performance';
    //show score and money
    change('money', money);
    reward = overallbonus;

    processDescription = document.getElementById('processDescription').value;
    patternDescription = document.getElementById('patternDescription').value;

    saveDataArray = {
        'prolificID': workerID,
        'studyID': studyID,
        'bonus': reward,
        //
        'blockcollect': blockcollect,
        'keyassignment': keyassignment,//which of DFJK is mapped into 1234 
        'condition': conditioncollect,//experiment condition, 0 = independent, 1 = M1, 2 = M2 
        'timecollect': timecollect,//reaction time [[rt1, rt2, ...]]
        'recallcollect': recallcollect,
        'instructioncollect': instructioncollect,// instructed keys
        'trialcollect': trialcollect,
        'correctcollect': correctcollect,
        //
        'patternawareness': patternDescription,//user comments
        'feedback': processDescription,//user comments
        'age': age,
        'gender': gender,
    };

    PHPsuccess = Momchil_datasave(saveDataArray);

    // onButtonFinishPressed();
    clickStart('page7', 'page8');
}

