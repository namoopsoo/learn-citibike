

/* Set the options for our chart */
var options = { segmentShowStroke : false,
								animateScale: true,
								percentageInnerCutout : 50,
                showToolTips: true,
                tooltipEvents: ["mousemove", "touchstart", "touchmove"],
                tooltipFontColor: "#fff",
								animationEasing : 'easeOutCirc'
              }

randomColor = function() {
	min = 1; max = 255;
	return Math.floor(Math.random() * (max - min + 1)) + min
}

randomRGB = function() {
	r = randomColor();
	g = randomColor();
	b = randomColor();
	return 'rgba(' + r.toString() + ', ' + g.toString() + ', ' + b.toString() + ', 0.2)';
}


updateDoughnut = function(chart_id, response) {

	console.log(response);
	/* Create the context for applying the chart to the HTML canvas */
	var ctx = $(chart_id).get(0).getContext("2d");

	/* Set the inital data */
	var data = [];

	for (var i = 0; i < response.length; i++) {
		data.push({
					value: response[i]['time-length'],
					// color: "#3498db",
					color: randomRGB(), // 'rgba(255, 99, 132, 0.2)',
					// highlight: "#2980b9",
					label: response[i]['project-identifier']
				})
	}
	console.log('data for doughnut: ');
	console.log(data);
	console.log('done updateDoughnut for chart_id ' + chart_id);

	// 
	graph = new Chart(ctx).Doughnut(data, options);

}

formatDateString = function(d) {
	var dd = d.getDate();
	var mm = d.getMonth()+1; //January is 0!
	var yyyy = d.getFullYear();

	if(dd<10) {
		dd='0'+dd
	} 

	if(mm<10) {
		mm='0'+mm
	} 

	return yyyy + '-' + mm + '-' + dd;
}


formatDateTimeString = function(d) {
	date_string = formatDateString(d);

	var hh = d.getHours();
	var mm = d.getMinutes(); 

	return date_string + ' ' + hh + ':' + mm + ':00';
}

copyDate = function(d) {
	x = new Date(JSON.parse(JSON.stringify(d)));
	return x
}

dateMinusDelta = function(d, delta) {
	x = copyDate(d);
	x.setDate(x.getDate() - delta);
	return x;
}

datePlusDelta = function(d, delta) {
	x = copyDate(d);
	x.setDate(x.getDate() + delta);
	return x;
}

changeDateDay = function(d, new_day) {
	x = copyDate(d);
	x.setDate(new_day);
	return x;
}

changeDateMonth = function(d, new_month) {
	x = copyDate(d);
	x.setMonth(new_month);
	return x;
}

dateFromDateString = function(date_str) {
	// Expecting date string like: "2017-12-01"
	x = new Date(date_str + "T12:00");
	return x;
}

getThisWeekDateRange = function(today) {
	// Yesterday
	yesterday = dateMinusDelta(today, 1);
	yesterday_string = formatDateString(yesterday);
	
	// last monday..
	// today.getDay()
	if (today.getDay() == 1) { // if today is Monday, nothing to do.
		return;
	} else if (today.getDay() == 0) {  // if today is Sunday , delta is 6.
		var delta = 6;
	} else {
		var delta = today.getDay() - 1;
	}

	last_monday = dateMinusDelta(today, delta);
	last_monday_string = formatDateString(last_monday);

	return [last_monday_string, yesterday_string];
}

getLastWeekDateRange = function(today) {
	// Get prior Sunday.
	if (today.getDay() == 0) { // Sunday
		var delta = 7;
	} else {
		var delta = today.getDay();
	}
	last_sunday = dateMinusDelta(today, delta);
	last_sunday_string = formatDateString(last_sunday);

	prior_monday = dateMinusDelta(last_sunday, 6);
	prior_monday_string = formatDateString(prior_monday);

	return [prior_monday_string, last_sunday_string];
}

getThisMonthDateRange = function(today) {
	// If the 1st, then nothing to do.
	if (today.getDate() == 1) {
		return;}
   
	yesterday = dateMinusDelta(today, 1);
	start_of_month = changeDateDay(today, 1);
	return [
		formatDateString(start_of_month),
		formatDateString(yesterday)];
}


getLastMonthDateRange = function(today) {
	last_month_end = dateMinusDelta(
			changeDateDay(today, 1),
			1);
	last_month_start = changeDateDay(last_month_end, 1);
	return [
		formatDateString(last_month_start),
		formatDateString(last_month_end)];
}

getThisYearDateRange = function(today) {
	yesterday = dateMinusDelta(today, 1);
	start_of_year = changeDateMonth(
			changeDateDay(today, 1), 0);

	return [
		formatDateString(start_of_year),
		formatDateString(yesterday)];
}

function assert(condition, message) {
    if (!condition) {
        throw message || "Assertion failed";
    }
}

assertRangesMatch = function(d, out, want) {
	assert(((out[0] == want[0]) && (out[1] == want[1])),
			'For ' + formatDateString(d) + ' got ' + out + '  But want ' + want);
}

testGetThisWeekDateRange = function() {
	// 2017-03-17
	var d = new Date(2017, 2, 17);
	var out = getThisWeekDateRange(d);
	var want = ["2017-03-13", "2017-03-16"];
	assertRangesMatch(d, out, want);

	// 2017-03-12 
	var d = new Date(2017, 2, 12);  
	var out = getThisWeekDateRange(d);
	var want = ["2017-03-06", "2017-03-11"];
	assertRangesMatch(d, out, want);

	// 2017-01-01 
	var d = new Date(2017, 0, 1);  
	var want = ["2016-12-26", "2016-12-31"];
	var out = getThisWeekDateRange(d);
}


testYearToDateDateRange = function() {
	// 2017-03-17
	var d = new Date(2017, 2, 17);
	var out = getThisYearDateRange(d);
	var want = ["2017-01-01", "2017-03-16"];
	assertRangesMatch(d, out, want);
}

updateThisWeekDoughnut = function() {
	// Is today Monday? 
	var today = new Date();
	// If so, doughnut should be like a blurry placeholder gif.
	// Because this week not ready on Mondays.
	if (today.getDay() == 1) {
		console.log('Today is Monday so this week is empty so far.');
		return;
	} else {
		var out = getThisWeekDateRange(today);
		last_monday_string = out[0];
		yesterday_string = out[1];
	}
	parameters = {
		// ?end-date=2017-01-03&start-date=2017-01-01&summary-type=%3Acore-category
		'end-date': yesterday_string, // 'end-date': '2017-01-03',
		'start-date': last_monday_string, // 'start-date': '2017-01-01',
		'summary-type': // ':core-category'
			':core-category:project-identifier'
	}

	response = querySummaryWithParams(parameters, "#graph_this_week");

}

updateLastWeekDoughnut = function() {
	var today = new Date();

	var out = getLastWeekDateRange(today);
	start_string = out[0];
	end_string = out[1];

	parameters = {
		'end-date': end_string,
		'start-date': start_string,
		'summary-type':
			':core-category:project-identifier'
	}

	response = querySummaryWithParams(parameters, "#graph_last_week");
}

updateThisMonthDoughnut = function() {
	var today = new Date();

	var out = getThisMonthDateRange(today);
	start_string = out[0];
	end_string = out[1];

	parameters = {
		'end-date': end_string,
		'start-date': start_string,
		'summary-type':
			':core-category:project-identifier'
	}

	response = querySummaryWithParams(parameters, "#graph_this_month");
}

updateLastMonthDoughnut = function() {
	var today = new Date();

	var out = getLastMonthDateRange(today);
	start_string = out[0];
	end_string = out[1];

	parameters = {
		'end-date': end_string,
		'start-date': start_string,
		'summary-type':
			':core-category:project-identifier'
	}

	response = querySummaryWithParams(parameters, "#graph_last_month");
	// console.log('response: "' + response + '"');

}

replaceSpacesPluses = function(x) {
	return x.replace(/ /g, '+');}


makeSortedParamString = function(parameters) {
	// Sort alphabetical order
	var the_keys = Object.keys(parameters).sort();

	var pairs = [];
    the_keys.forEach(function(key) {
		pairs.push(key + '=' + encodeURIComponent(parameters[key]));
	});

	var output = '?' + pairs.join('&');
	return output;
}


testMakeSortedParamString = function() {
	// Test one
	parameters = {
		'end-date': '2017-01-03',
		'start-date': '2017-01-01',
		'summary-type': ':core-category:project-identifier'
	}

	var param_string = makeSortedParamString(parameters);
	var expected = "?end-date=2017-01-03&start-date=2017-01-01&summary-type=%3Acore-category%3Aproject-identifier";

	assert((param_string == expected),
			"Dont have a match. Got " + param_string);

	// Test two
	parameters = {
		'end-date': '2017-01-03',
		'start-date': '2017-01-01',
		'summary-type': ':core-category:project-identifier',
		'period': 'daily'
	}

	var param_string = makeSortedParamString(parameters);
	var expected = "?end-date=2017-01-03&period=daily&start-date=2017-01-01&summary-type=%3Acore-category%3Aproject-identifier";

	assert((param_string == expected),
			"Dont have a match. Got " + param_string);
}

arraysEqual = function(a1, a2) {
	// check each element at a time.
	if (a1.length != a2.length) return false;

	for (i = 0; i < a1.length; i++) {
		if (a1[i] != a2[i]) return false;
	}
	return true;
}

fillMissingDates = function(dates) {
	// Given a list of date strings, return a list without dates missing in between.

	// Find min and max
	// sort first.
	var sorted_dates = dates.slice();
	sorted_dates.sort(); // sorts ascending

	var start = sorted_dates[0];
	var end = sorted_dates[sorted_dates.length -1];

	var new_list = [];
	var start_date = dateFromDateString(start);
	var end_date = dateFromDateString(end);

	var d = copyDate(start_date);
	while(d <= end_date) {
		new_list.push(formatDateString(d));
		d = datePlusDelta(d, 1);
	}
	return new_list;
}

testFillMissingDates = function() {
	// Test One
	var input_dates_list = ["2017-12-01","2017-12-04","2017-12-05","2017-12-06","2017-12-07","2017-12-08","2017-12-11","2017-12-12","2017-12-13","2017-12-14","2017-12-15"];
	var output_dates_list = fillMissingDates(input_dates_list);
	var expected_output = ["2017-12-01", "2017-12-02", "2017-12-03", "2017-12-04","2017-12-05","2017-12-06","2017-12-07","2017-12-08", "2017-12-09", "2017-12-10", "2017-12-11","2017-12-12","2017-12-13","2017-12-14","2017-12-15"];

	assert(arraysEqual(output_dates_list, expected_output), "Test1: Expected did not match. Got: " + output_dates_list);

	// Test Two
	var input_dates_list = ["2017-11-28","2017-12-04","2017-12-05"];
	var output_dates_list = fillMissingDates(input_dates_list);
	var expected_output = ["2017-11-28","2017-11-29","2017-11-30","2017-12-01", "2017-12-02", "2017-12-03", "2017-12-04","2017-12-05"];

	assert(arraysEqual(output_dates_list, expected_output), "Test2: Expected did not match. Got: " + output_dates_list);

}

queryMyUrl = function(parameters, authparams, output_id, output_id_2) {


	// url = 'https://rmuxqpksz2.execute-api.us-east-1.amazonaws.com/default/myBikelearnSageLambda?start_station=Forsyth+St+%26+Broome+St&start_time=10%2F8%2F2015+18%3A04%3A57&rider_gender=2&rider_type=Subscriber&birth_year=1973'
	//base_url = 'https://rmuxqpksz2.execute-api.us-east-1.amazonaws.com/default/destinations'
	base_url = 'https://rmuxqpksz2.execute-api.us-east-1.amazonaws.com/default/myBikelearnSageLambda'
	
	var goodstuff = prepareAuthenticatedAPIRequest(parameters,
												authparams,
												base_url);
	console.log('good stuff');
	console.log(goodstuff);

	$.get({
		url: goodstuff['full_uri'],
		headers: goodstuff['headers'],

		success: function(response) {
			console.log(response);
			console.log('end response for ' + output_id);

			document.getElementById(output_id).innerHTML = "0: starting location " + JSON.stringify(response['start_location']) + "<br/><br/>And 1-9: Destination neighborhoods top 9 probabilities: " + JSON.stringify(response['probabilities']);

			//$('#' + output_id_2).text(response['map_html']);
			document.getElementById(output_id_2).innerHTML=response['map_html'];

		},
		error: function(response) {
			console.log('Crap. error: ' + response + '...');
			console.log(response);
			console.log('response len ' + response.length);

			$('#' + output_id).text(JSON.stringify(response));

		}
	});

}


prepareAuthenticatedAPIRequest = function(parameters,
										authparams,
										base_url) {

	var config = Object.assign(
								{},
								authparams, 
								{region: 'us-east-1',
								service: 'execute-api'});

	var signer = new awsSignWeb.AwsSigner(config);
	var full_uri = base_url + makeSortedParamString(parameters); 
	//console.log("full_uri: " + full_uri);
	console.log('using this config for signing ')
	console.log(config)

	var request = {
		method: 'GET',
		url: base_url, // URL w/o querystring here!
		headers: {},
		params: parameters,
		data: null
	};
	var signed = signer.sign(request);

	var headers = {
			'Authorization': signed.Authorization,
			'Accept': signed.Accept,
			'x-amz-date': signed['x-amz-date']};

	if ('sessionToken' in authparams){
		headers['X-Amz-Security-Token'] = authparams['sessionToken'];
	}

	return {
		base_url: base_url,
		full_uri: full_uri,
		headers: headers}
}


querySummaryWithParams = function(parameters, chart_id) {
	// Create a new signer
	var config = {
		region: 'us-east-1',
		service: 'execute-api',
		// AWS IAM credentials, here some temporary credentials
		accessKeyId: '',
		secretAccessKey: ''
	};
	console.log("config: ");
	console.log(config);
	var signer = new awsSignWeb.AwsSigner(config);

	// Update params with work 
	parameters['core-category'] = 'work';

	// Make request url
	var base_url = 'https://m8fe2knl2f.execute-api.us-east-1.amazonaws.com/staging/summary';

	// var full_uri = base_url + '?end-date=' + parameters['end-date'] + '&start-date=' + parameters['start-date'] + '&summary-type=%3Acore-category%3Aproject-identifier';

	var full_uri = base_url + makeSortedParamString(parameters); 
	console.log("full_uri: " + full_uri);

	// var full_uri = base_url + '?end-date=2017-01-03&start-date=2017-01-01&summary-type=%3Acore-category';

	// Sign a request
	var request = {
		method: 'GET',
		// URL w/o querystring here.
		url: base_url,
		headers: {},
		params: parameters,
		data: null
	};
	var signed = signer.sign(request);

	// Make Request.
	$.get({
		url: full_uri,
		headers: {
			'Authorization': signed.Authorization,
			'Accept': signed.Accept,
			'x-amz-date': signed['x-amz-date'],
		},

		success: function( response ) {
			console.log('success for chart_id:' + chart_id);
			console.log('parameters:');
			console.log(parameters);
			console.log(response);
			console.log('end response for ' + chart_id);

			if (chart_id == "daily_stacked_area_chart") {
				// Currently need this hack since the d3js code
				// is not specific to an #id yet.
				wrapperUpdateTheDailyStackedChart(response);
			} else {
				updateDoughnut(chart_id, response);
			}
		},
		error: function( response ) {
			console.log( 'error: ' + response ); // server response
		}
	});
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function asyncsleep() {
  console.log('Taking a break...');
  await sleep(2000);
  console.log('Two seconds later, showing sleep in a loop...');

  await sleep(2000);
}



authParametersFromCognito = function(callback, callback_params) {
	// Initialize the Amazon Cognito credentials provider
	AWS.config.region = 'us-east-1'; 
	AWS.config.credentials = new AWS.CognitoIdentityCredentials({
				IdentityPoolId: 'us-east-1:ef218443-9093-4480-8a26-47ed7bdeff24'
			});

	//console.log('start wait');
	//await sleep(2000);
	//console.log('end wait');

	AWS.config.credentials.get(function(err) {
		if (err) {
			console.log("Error: "+err);
			return;
		}
		console.log("Cognito Identity Id: " + AWS.config.credentials.identityId);

		authparameters = {
			'accessKeyId': AWS.config.credentials.accessKeyId,
			'secretAccessKey': AWS.config.credentials.secretAccessKey,
			// save for later...
			'sessionToken': AWS.config.credentials.sessionToken
		}
		console.log('authparameters', authparameters);
		//
		console.log('parameters from form: ' + JSON.stringify(callback_params));
		callback(callback_params, authparameters, 'out-div', 'out-div-2');
		});


}

