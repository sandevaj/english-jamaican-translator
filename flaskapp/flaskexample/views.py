from flask import render_template, request
from flaskexample import app
from combined_fx import *

@app.route('/')
@app.route('/input')
def translator_input():
	return render_template("input.html")

@app.route('/jamoutput')
def jamaican_output():
	try:
	    text = request.args.get("engtext")
	    result = evaluate('english2jamaican', text)
	    result = result[:-7]
	    fixed_result = result[0].upper() + result[1:] 
	    return render_template("jamoutput.html",jam_seq= fixed_result,
		eng_input= text)
	except KeyError:
		return render_template("engkeyerror.html", eng_input=text)

@app.route('/engoutput')
def english_output():
	try:
		text = request.args.get("jamtext")
		result = evaluate('jamaican2english',text)
		result = result[:-7]
		fixed_result = result[0].upper() + result[1:] 
		return render_template("engoutput.html",eng_seq= fixed_result,
		jam_input= text)
	except KeyError:
		return render_template("jamkeyerror.html", jam_input=text)
