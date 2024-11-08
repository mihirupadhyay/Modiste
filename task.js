var timeline = []


/* init connection with pavlovia.org */
var pavlovia_init = {
	type: "pavlovia",
	command: "init"
};
timeline.push(pavlovia_init);

// capture info from Prolific
// help from: https://www.jspsych.org/overview/prolific/
var prolific_id = jsPsych.data.getURLVariable('PROLIFIC_PID');
var study_id = jsPsych.data.getURLVariable('STUDY_ID');
var session_id = jsPsych.data.getURLVariable('SESSION_ID');
// subj id help from: https://www.jspsych.org/7.0/overview/data/index.html
// generate a random subject ID with 15 characters
var subject_id = jsPsych.randomization.randomID(15);
jsPsych.data.addProperties({
	subject: subject_id,
	prolific_id: prolific_id,
	study_id: study_id,
	session_id: session_id
});

// function numberRange (start, end) {
	// from: https://stackoverflow.com/questions/3895478/does-javascript-have-a-method-like-range-to-generate-a-range-within-the-supp
	//return new Array(end - start).fill().map((d, i) => i + start);
//}

// modified snippet

// console.log(variant_batches[0])

function numberRange(start, end) {
   
    console.log(`start: ${start}, end: ${end}`);

	if (end === undefined) {
        end = start;
        start = 0;
    }
    
    if (typeof start !== 'number' || typeof end !== 'number') {
        console.error("TypeError: Start and end must be numbers.");
        throw new TypeError("Start and end must be numbers.");
    }

    if (end < start) {
        console.error("RangeError: End must be greater than or equal to start.");
        throw new RangeError("End must be greater than or equal to start.");
    }

    var length = end - start;
    console.log(`Calculated length: ${length}`);

    if (length < 0) {
        console.error("Length cannot be negative.");
        throw new RangeError("Invalid array length: length cannot be negative.");
    }
    
    if (length > 4294967295) {
        console.error("Length exceeds maximum array size.");
        throw new RangeError("Invalid array length: exceeds maximum array size.");
    }

    return new Array(length).fill().map((_, i) => i + start);
}

// some flags mainly for internal debugging
var official_run = true
var no_server = true
var simulate = false
var impoversh_fp = false // this was a tag for impovershing the foreign policy category -- very specific to our case, ignore here / delete!

// console.log(variant_batches.length)



var main_variant = "diff-topics"//"pilot-separable-human-impovershed"
// console.log(variant_batches[0])

var batches = variant_batches[0][main_variant]

console.log(batches.length)

// pick a random condition for the subject at the start of the experiment
// help from: https://www.jspsych.org/overview/prolific/
// based on our total number of batches <--- note: can subset if we need to run some a few more
var num_batches = 4

// conditions are like "batch indexes"
var conditions = numberRange(num_batches);
var condition_num = jsPsych.randomization.sampleWithoutReplacement(conditions, 1)[0];

// var condition_num = 18// 8

console.log("condition: ", condition_num)

// record the condition assignment
jsPsych.data.addProperties({
	condition: condition_num
});

// index into batched mixup images
// javascript loading help from: https://github.com/jspsych/jsPsych/discussions/705
var batchData = batches[String(condition_num)]

var variant = batchData[0]["variant"]

console.log("variant: ", variant)

if (variant.includes("alg")){
	no_server = false;
} else {
	no_server = true;
}

var numExamples = batchData.length
// the examples are ordered so that the last T are class-balanced
// shuffle the subsets
// for text task, we balance here by topics

// specific to MMLU -- we were class-balancing at the end
// you can likely ignore
var final_sample = 18
var first_idxs = jsPsych.randomization.shuffle(numberRange(0, numExamples - final_sample))
var final_idxs = jsPsych.randomization.shuffle(numberRange(numExamples - final_sample, numExamples))

var ordered_idxs = first_idxs.concat(final_idxs)

console.log(variant_batches[0])

if (impoversh_fp){
	main_variant = "impoversh_fp"
}

// record the condition assignment
jsPsych.data.addProperties({
	condition: condition_num
});


var options=["Yes", "No"]

var classNames = ['A', 'B', 'C', 'D']

var num_show = batchData.length //+ num_rerun

var num_pages_per_img = 1 // to handle num pages per img

var progress_bar_increase = 1 / (num_show * num_pages_per_img)

// consent form help from: https://gitlab.pavlovia.org/beckerla/language-analysis/blob/master/html/language-analysis.js
// sample function that might be used to check if a subject has given consent to participate.
var check_consent = function(elem) {
	if ($('#consent_checkbox').is(':checked') && $('#read_checkbox').is(':checked') && $('#age_checkbox').is(':checked')) {
		return true;
	}
	else {
		alert("If you wish to participate, you must check the boxes in the Consent Form.");
		return false;
	}
	return false;
};
var consent = {
	type:'external-html',
	url: "consent.html",
	cont_btn: "start",
	check_fn: check_consent
}

if (official_run){
	timeline.push(consent)
}

var total_time = 30;
var base_rate = 9;
var bonus_rate = 10;

var base_pay = base_rate * (total_time/60);
var bonus_pay = bonus_rate * (total_time/60) - (base_pay);

base_pay = parseFloat(base_pay).toFixed(2);
bonus_pay = parseFloat(bonus_pay).toFixed(2);

var model_use_instruction ='<p> The model\'s prediction will show up as yellow highlighting over that answer choice. If shown, you are free to use or ignore the information when selecting your answer however you wish.</p>'

var instruction_pages = ['<p> Welcome! </p> <p> We are conducting an experiment to understand how people make decisions with and without AI support. Your answers will be used to inform machine learning, cognitive science, and human-computer interaction research. </p>' +
'<p> This experiment should take at most <strong>' + total_time + ' minutes</strong>. </br></br> You will be compensated at a base rate of $'+ base_rate + '/hour for a total of <strong>$' + base_pay + '</strong>, which you will receive as long as you complete the study.</p>',
'<p> We take your compensation and time seriously! The email for the main experimenter is <strong>cambridge.mlg.studies@gmail.com</strong>. </br></br> Please write this down now, and email us with your Prolific ID and the subject line <i>Human experiment compensation</i> if you have problems submitting this task, or if it takes much more time than expected. </p>',
'<p> In this experiment, you will be seeing <i>multiple choice questions</i>, from various topics, such as those that you may find in school (e.g., biology, mathematics, foreign policy, computer science).</p>' +
'<p> Your task is to determine the <strong>most likely answer</strong> for each question. You can select this category by clicking on the radio button associated with your answer. </p>'];

if ((variant == "justM1") || (variant == "justM2")){

	instruction_pages.push('<p> During the tasks, you will also see the <strong>prediction of an AI-based model</strong>.</p>' +
	model_use_instruction) // +
}
else if (variant.includes("alg") || variant.includes("popStats") || variant.includes("fixedPop")){

	instruction_pages.push('<p> During the tasks, you may also see the <strong>prediction of an AI-based model</strong>.</p>' +
	model_use_instruction + '<p> On other trials, you will see no additional information. Don\'t be alarmed if you do not see the model prediction on any given example.</p>')
}

instruction_pages.push('<p> We encourage you to try to work through each problem. You will not be able to continue to the next question until at least <strong>10 seconds</strong> have passed. The SUBMIT button will change from grey to blue when you are able to click to move to the next page whenever you are ready to answer.</p>' +
'<p> Of course you can take longer than 10 seconds on any question if needed! It may be very challenging to determine the answer for some questions. Others may be easy. <strong>Please try your best</strong> regardless. </p>')

instruction_pages.push(
	'<p> You will receive a <strong>bonus</strong> of up to a rate of $' + bonus_rate + '/hour (+$' + bonus_pay + ') based on how many questions you correctly answer.</p>' +
	'<p> You will be informed whether or not you are correct after each trial. </p>');

instruction_pages.push(
	'<p> We realize that some topics may be outside of your expertise. If you don\'t know an answer, please give it your best guess! <strong>Please do not search on Google</strong>, or any other web-browser during the experiment.</p>' +
	'<p> There is room to note topics you are unfamiliar with in the comment section at the end of the survey. We will take this into account with the bonus and will help us inform the design of future studies. </p>'
)

instruction_pages.push('<p> You will see a total of <strong>' + num_show + ' questions</strong>.</p>');

instruction_pages.push('<p> When you are ready, please click <strong>\"Next\"</strong> to complete a quick comprehension check, before moving on to the experiment. </p>' +
'<p> Please make sure to window size is in full screen, or substantially large enough, to properly view the questions. </p>');

var instructions = {
	type: "instructions",
	pages: instruction_pages,
	show_clickable_nav: true,
};

var correct_task_description = "The answer to a mutliple choice question."

var correct_perspective_description = "During <i>some</i> of the trials."
var incorrect_perspective_description = "During <i>all</i> of the trials."
var incorrect_perspective_description2 = "During <i>none</i> of the trials."

var comprehension_check = {
	type: "survey-multi-choice",
	preamble: ["<p align='center'>Check your knowledge before you begin. If you don't know the answers, don't worry; we will show you the instructions again.</p>"],
	questions: [
		{
			prompt: "What will you be asked to determine in this task?",
			options: [correct_task_description, "The least likely answer to a multiple choice question.", "The most likely categories of an image.",],
			required: true
		},

		{
			prompt: "How will you select your answer?",
			options: ["Typing in a text box.", "Clicking on a radio button.", "Selecting from a dropdown menu."],
			required: true
		},


	],
	on_finish: function (data) {
		var responses = data.response;
		if (responses['Q0'] == correct_task_description &&  responses['Q1'] == "Clicking on a radio button." ){
			familiarization_check_correct = true;
		} else {
			familiarization_check_correct = false;
		}
	}
}

var familiarization_timeline = [instructions, comprehension_check]

var familiarization_loop = {
	timeline: familiarization_timeline,
	loop_function: function (data) {
		return !familiarization_check_correct;
	}
}

if (official_run){
	timeline.push(familiarization_loop)
}

var final_instructions = {
	type: "instructions",
	pages: ['<p> Now you are ready to begin! </p>' +
	'<p> Please click <strong>\"Next\"</strong> to start the experiment. Note, it may take a moment to load at the start.</p>' +
	'<p> Thank you for participating in our study! </p>'],
	show_clickable_nav: true
};
timeline.push(final_instructions)


// keep track of global info that's sent/received from server interactions
// update accordingly
var modelPredLabel = ""
var humanPredLabel = ""
var humanCorrect = false // store the most recent human correctness
var currentInterfaceType = "defer"
var modelEntropy=null
var score = 0
var tot = 0
var humanPred = 0
var friendPredLabel = ""
var friendPredDist = []
var modelPredDist = []

var question=""
var options=""


var getGrammaticalArticle = function(word){
	// return "a" or "an" depending on whether start of word is a vowel
	var firstChar = word[0]
	if (["a", "e", "i", "o", "u"].includes(firstChar)){
		return "an"
	} else {
		return "a"
	}
}

// start off w/ button not clicked -- adjust if so, and reset per trial
// global variable is hacky... might change later
var btnClicked=false
// global to measure button press time -- note, may not be super precise...
// help from: https://stackoverflow.com/questions/313893/how-to-measure-time-taken-by-a-function-to-execute
var trialStartTime = null;
var btnClickedTime = null

var onClickHandler = function(show_txt) {

	btnClickedTime = new Date().getTime();

	// alert(show_txt.value)
	btnClicked=true

	// can also maintain time information here and/or num clicks
	// modified from: https://github.com/jspsych/jsPsych/discussions/1931
	document.getElementById('modelInfo').style.cssText = document.getElementById('modelInfo').style.cssText.replace("hidden","visible");

};


var storeHumanSelect = function(selectIdx) {
	// help from: https://stackoverflow.com/questions/647282/is-there-an-onselect-event-or-equivalent-for-html-select
	humanPred = selectIdx
}

var main_page = {

	// help for various input forms from: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
	// and: https://www.jspsych.org/7.0/plugins/survey-html-form/
	type: 'survey-html-form',

	html: function() {
		// var img =jsPsych.timelineVariable('filename').split("imgs/")[1]


		var human_corrupted = jsPsych.timelineVariable('human_corrupted')

		if (human_corrupted == 1){
			var img =jsPsych.timelineVariable('corrupted_filename')

			console.log("img name: ", img)

		}else{
				var img =jsPsych.timelineVariable('clean_filename')
		}


		// var interfaceType = currentInterfaceType //jsPsych.timelineVariable('interfaceType')

		img = "imgs/" + img

		// help from: https://stackoverflow.com/questions/441018/replacing-spaces-with-underscores-in-javascript

		var topic = jsPsych.timelineVariable('display_topic') // .split('_').join(' ')

		btnClicked=false

		console.log("INTERFACE: ", currentInterfaceType)

		modelPredLabel = jsPsych.timelineVariable('llm_answer')
		friendPredLabel = jsPsych.timelineVariable('label')

		question = jsPsych.timelineVariable("question")
		options = jsPsych.timelineVariable("options")


		// // PATCH TO RUN W/O SERVER
		if (no_server){

			console.log("no server: ", no_server)

			currentInterfaceType=jsPsych.timelineVariable('action')

			console.log("REVISED INTERFACE: ", currentInterfaceType)

		}

		/* humanCorrect holds whether human was correct on last trial
		could be used to change color and/or add txt */


		    	//var currentScoreTxt = '<p class="feedback">You have classified ' + score + ' out of ' + tot + ' images correct (' + Math.round((score / tot) * 100) + '%).</p>'
		 		// NOTE: if we only want to show the score after the first trial, could use this instead
				if (tot != 0){
				//	var currentScoreTxt = '<div id="runningScore"><p class="score"><strong>Your Score: </strong></br>' + score + ' out of ' + tot + ' correct (<strong>' + Math.round((score / tot) * 100) + '%</strong>)</p></div></div>'
					var currentScoreTxt = '<div id="runningScore"><p class="score"><strong>Your Score: </strong></br>' + score + ' out of ' + tot + '</p></div></div>'
				}else{
					var currentScoreTxt = '<div id="runningScore"><p class="score"><strong>Your Score: </p><p style="text-align: center;"> - </strong></p></div></div>'
				}


				var interfaceHtmlTxt = `
								<div class='outer_card'>
										<div class='inner_card'>
										    <div class = "first_row">
										    	<div class="headerInst">`
										    //<div class="headerInst"><h2>Please answer the question.</h2>


				//interfaceHtmlTxt += '<p class="instructions"></p></div>'
				interfaceHtmlTxt += `<h2>Please answer the question about <strong>` + topic + `</strong> by selecting exactly one of the answers below. `
				// If an answer is marked in <span class="highlight">yellow</span>, it is the answer that the AI predicts to be correct.</h2></div>`

				aiInstructionTxt = `An AI model\'s predicted answer is marked in <span class="highlight">yellow</span>.`
				//
				// interfaceHtmlTxt+=topicTxt

				if (currentInterfaceType == "showPred"){
					console.log("adding instruction!!!", currentInterfaceType)
					interfaceHtmlTxt+= aiInstructionTxt
				}

				interfaceHtmlTxt += `</h2></div>`


				interfaceHtmlTxt += currentScoreTxt

	// 			var task_data = {'question': 'What is the term for a sub-optimal but acceptable outcome of negotiations between parties?',
 // 'options': ['Bargaining', 'Satisficing', 'Accepting', 'Compromising'],
 // 'answer': 'B'}

 			// var question = jsPsych.timelineVariable("question")
			// var options = jsPsych.timelineVariable("options")

				var questionTxt = '<div class="question">' + question


			  modelPredLabel = jsPsych.timelineVariable("llm_answer")
				var trueLabel = jsPsych.timelineVariable("label")

				if ((impoversh_fp) && (jsPsych.timelineVariable("topic").includes("policy"))){

					// sample from the definitely wrong
					// console.log(options)
					var possAnswers = new Set(classNames);
					// console.log(possAnswers, trueLabel)
					possAnswers.delete(trueLabel)
					// console.log(possAnswers)
					modelPredLabel = possAnswers.values().next().value;//jsPsych.randomization.sampleWithoutReplacement(new Array(possAnswers), 1)[0];

					// console.log("possWrong: ", possAnswers, " true: ", trueLabel, " model: ", modelPredLabel)



				}


				var optionsTxt = '<form action="">'
				for (var i = 0; i < options.length; i++){
					var option = options[i];

					if (!(official_run) && (simulate)){
						var optionHTML = '<p><input type="radio" id='+classNames[i]+' name=mcAnswer value='+classNames[i]+'>'
						+ '<label for='+classNames[i]+'>'
					} else{
						var optionHTML = '<p><input type="radio" id='+classNames[i]+' name=mcAnswer value='+classNames[i]+' required>'
						+ '<label for='+classNames[i]+'>'
					}
					// var optionHTML = '<p><input type="radio" id='+classNames[i]+' name=mcAnswer value='+classNames[i]+' required>'
					// + '<label for='+classNames[i]+'>'

					console.log("INTERFACE!!", (currentInterfaceType == "showPred"), (classNames[i] == modelPredLabel))

					if ((currentInterfaceType == "showPred") & (classNames[i] == modelPredLabel)){
						// highlight entry
						// help from: https://www.computerhope.com/issues/ch001391.htm
						//optionHTML += '<span class="highlight">' + classNames[i] + ') ' + option + '</span>'
						optionHTML += '<span class="highlight">' + option + '</span>'
					}
					else {
						optionHTML += option

					}
					optionHTML+='</label></p>'
					optionsTxt += optionHTML


				}
				optionsTxt += "</form></div>"


				// add image and button next to each other
				// interfaceHtmlTxt += '<div class="flex_columns">' + questionTxt + optionsTxt //<img src=' + img + '>'

				interfaceHtmlTxt += '<div>' + questionTxt + optionsTxt //<img src=' + img + '>'


				var debug_panel = "<div><strong>True Label:</strong> " + jsPsych.timelineVariable("label") + " <strong>Current Arm: </strong>" + currentInterfaceType + "</div>"

				if (!official_run){
					interfaceHtmlTxt += debug_panel
				}

				console.log("model pred label: ", modelPredLabel, " true: ", trueLabel)


				return interfaceHtmlTxt
	},
	button_label: 'SUBMIT',
	// trial_duration: 1500,
	on_load: function(){

		// help from: https://stackoverflow.com/questions/60214165/how-to-disable-button-for-period-and-repeat-that-again-javascript
		const btn = document.getElementById('jspsych-survey-html-form-next')

		btn.disabled=true
		btn.style.background = "#A8A8A8";

		if (official_run){
			var time_delay = 10000
		}else{
			var time_delay = 100
		}

		setTimeout(function(){
			btn.disabled = false;
			console.log('Button Activated')
			btn.style.background = "#2e436b";
      // document.querySelector('jspsych-survey-html-form-next').src = 'imgB.png'
    }, time_delay);

	},
	on_finish: function (data) {

		humanPred = data.response["mcAnswer"]

		console.log("human pred: ", humanPred)

		jsPsych.data.get().push(humanPred);

		// check whether person is "right" or wrong
		var trueCategory = jsPsych.timelineVariable("label");

		if (humanPred == trueCategory){
			/* the human is correct! */
			humanCorrect = true
			score += 1
			 // alert("You were correct!");
		}else{
			/* the human is incorrect :( */
			humanCorrect = false
			// alert("You were incorrect.");
		}
		tot+=1
		humanPredLabel=humanPred // store for later

		console.log("Human Pred: ", humanPred, " Correct? ", humanCorrect)

		console.log("CHECK: model pred label: ", modelPredLabel, " true: ", trueCategory, "SAVING: ", currentInterfaceType)

		if (modelPredLabel == trueCategory) {
			modelCorrect = true
		}else{
			modelCorrect = false
		}


		var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
		jsPsych.setProgressBar(curr_progress_bar_value + progress_bar_increase);


		if (btnClicked){
			timeToClick = btnClickedTime-trialStartTime
		}else{timeToClick = -1} // "imposs" value just for the case when no btn pressed (easy to identify in analysis)


		// help from: https://www.jspsych.org/7.0/reference/jspsych-data/index.html
		var data={"btn_clicked": btnClicked, "score": score, "timeToClick": timeToClick,
			"modelPred": modelPredLabel,
			 "humanCorrect": humanCorrect, "modelCorrect": modelCorrect,
			 "interfaceType": currentInterfaceType, "modelPredDist": modelPredDist}
		jsPsych.data.get().push(data);

		trialStartTime=null
		btnClickedTime=null

		humanPred = 0 // reset in case someone wants "Airplane" (or whatever the first category is)

	}
}

var current_filename = null
var unique_subj_id = null
var example_label = null
var data_split = null
var img_id = null

var stringToDist = function(distStr) {
	// convert a string represensentation of a distribution to vector form
	// e.g., 0.75_0.1_0.05_0_0.1 => [0.75, 0.1, 0.05, 0.0, 0.1]

	var strProbs = distStr.split("_")
	var strProbs = distStr.split("_")
	var dist = strProbs.map(x => parseFloat(x));
	return dist
}


var send_receive = {
		type: 'call-function',
		async: true,
		func: function(done){

			if (no_server){
				done()
			}
			else{

				// help from Kartik
				// and: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_client_applications
					const oReq = new XMLHttpRequest();
					// oReq.addEventListener("load", function(){serverReceiver(done)});
					// oReq.addEventListener("load", serverReceiver);

					oReq.onload = () => {

						var responseMsg = oReq.response

						// modelPredLabel, currentInterfaceType = responseMsg.split("-")

						var msgData = responseMsg.split("*")


						// modelPredLabel = msgData[0]
						currentInterfaceType = msgData[0]
						// // modelEntropy = msgData[3]
						// friendPredLabel = msgData[2]
						// friendPredDist = stringToDist(msgData[3])
						// modelPredDist = stringToDist(msgData[4])

						// currentInterfaceType = "clickOption"

						// currentInterfaceType="clickOption"
						done(oReq.response)
					}


					// console.log(jsPsych.timelineVariable('filename'))

					var send_example_idx = jsPsych.timelineVariable('example_idx')
					var send_oracle_label = jsPsych.timelineVariable('label')

					console.log("USER ID!!!!!!!!", subject_id)

					var msg = subject_id + "*" + send_example_idx + "*" + send_oracle_label + "*" + humanPredLabel + "*" + condition_num + "*" + variant
					oReq.open("GET", "https://interface_project.alanliu.dev/" + msg);
					// oReq.open("GET", "http://localhost:80/" + msg);
					// ^using the above will let you run a local server

					oReq.send();


			}
		}
}

	var feedback_page = {
	    type: "image-keyboard-response",
	    stimulus: function(){

				if (humanCorrect) {
					return "feedback_imgs/Correct.png"
				}else{
					return "feedback_imgs/Incorrect.png"
				}

			},
	    choices: "NO_KEYS",
	    prompt: function(){



				var feedback_txt = "<p class='feedback'>"

				if (humanCorrect) {
					feedback_txt += "Correct! </p>"
				}else{
					feedback_txt +="Incorrect. </p>"

				}

				if (currentInterfaceType == "showPred"){
					console.log("interface was show: ", currentInterfaceType)
					if (modelCorrect){
						feedback_txt += "<p class='feedback'> AI Model: <strong><span style='color: #0b661f'>CORRECT</span></strong></p>"
					} else{
						feedback_txt += "<p class='feedback'> AI Model: <strong><span style='color: #87230c'>INCORRECT</span></strong></p>"
					}
				}


				return feedback_txt

			},
	    trial_duration: 1500 // 1500
	};

	var rating_task = {
		timeline: [send_receive, main_page, feedback_page],
		timeline_variables: batchData,
		data: {
			question: jsPsych.timelineVariable('question'),
			options: jsPsych.timelineVariable('options'),
			llm_answer: jsPsych.timelineVariable('llm_answer'),
			task: 'modiste',
			subj_id: jsPsych.timelineVariable('id'),
			label: jsPsych.timelineVariable('label'),
			example_id: jsPsych.timelineVariable('example_idx'),
			variant: variant,//jsPsych.timelineVariable('variant'),
			main_variant: main_variant,
			topic: jsPsych.timelineVariable('topic'),
			prompt: jsPsych.timelineVariable('prompt'),
		},
		sample: {
			type: 'custom',
			fn: function (t) {
				// t = set of indices from 0 to n-1, where n = # of trials in stimuli variable
				// returns a set of indices for trials
				return ordered_idxs
			}
		},
		on_load: function(){
			trialStartTime=new Date().getTime();
			btnClickedTime=null
		},

	}

	timeline.push(rating_task);

	var comments_block = {
		type: "survey-text",
		preamble: "<p>Thank you for participating in our study!</p>" +
		"<p>Click <strong>\"Finish\"</strong> to complete the experiment and receive compensation. If you have any comments about the experiment, please let us know in the form below.</p>",
		questions: [
			{prompt: "Were the instructions clear? (On a scale of 1-10, with 10 being very clear)"},
			{prompt: "How challenging did you find the questions? (On a scale of 1-10, with 10 being very challenging)"},

			// {prompt: "How challenging was it to choose which category to select as most probable per image? (On a scale of 1-10, with 10 being very challenging)"},
			// {prompt: "How challenging was it to come up a category when you did not have access to the model prediction? (On a scale of 1-10, with 10 being very challenging)"},
			// {prompt: "Did you trust the predictions you received from the model? (On a scale of 1-10, with 10 being always trusted)"},
				{prompt: "Were there any question topics you struggled with?", rows: 2, columns: 50},
				{prompt: "Were there any question topics you were always very confident in?", rows: 2, columns: 50},
			// {prompt: "Were there any particular qualities of images you considered when coming up with your response?", rows:5,columns:50},
			{prompt: "Did you notice the delay before you were allowed to submit a response?", rows: 1, columns: 50},
			{prompt: "If you answered yes to noticing the delay, was it too long?", rows: 1, columns: 50},

			{prompt: "Do you have any additional comments to share with us?", rows: 5, columns: 50}],
			button_label: "Finish",
		};
		timeline.push(comments_block)

	/* finish connection with pavlovia.org */
	var pavlovia_finish = {
		type: "pavlovia",
		command: "finish",
	};
	timeline.push(pavlovia_finish);

	// todo: update w/ proper prolific link!!
	jsPsych.init({
		timeline: timeline,
		on_finish: function () {
			// send back to main prolific link
			// window.location = "https://www.google.com/"

			window.location = "https://app.prolific.co/submissions/complete?cc=2681CCA7"
		},
		show_progress_bar: true,
		auto_update_progress_bar: false,

	});
