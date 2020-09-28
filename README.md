# Movie-Recommendation-System-Using-Flask-and-Machine-Learning

This System Generate Movies Recommendation based upon a movie.It consists of Machine Learning Model which generate recommendations according to input.
It also allows user to create its favorites list of movies for that user needs to signup and signin.

## To run this app follow the following instructions:
First make sure you have python version 3.7 and pip version 19.0.3 or higher installed on your system. 
To install various dependencies for this project run the following command

"pip install -r requirements.txt"

In this project file name "app.py" works as an backend which helps in the interaction of user and ML model.
"Recommender_complete.py" is a Machine Learning model which generate recommendations.

After installation of dependencies run folllowing command to run the model.

"python Recommender_complete.py"

After that run folllowing commmand to run the main app.

"python app.py"

The above command will start the local server on the system at address "http://127.0.0.1:5000/" which is default address of flask to serve templates.

To change the address and port of flask app so that everyone can access it, you need to edit "app.py" file.
At the very bottom of the "app.py" there is command "app.run(debug=True)" you can add various parameters to this function:
    
#### example "app.run(host='0.0.0.0',port=3000,debug=True)" in this case you can provide your suitable IP address at "host" and port number in "port" parameter.

After this the app will run on given IP address and generate the Recommendations.
