from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_mysqldb import MySQL
from Recommender_complete import recommend, function_to_return_link, Return_details, highest_imdb_movies    
import MySQLdb.cursors
import re

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

app = Flask(__name__)
app.secret_key = 'niket'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'movie_recommendation'

mysql = MySQL(app)

# highest_imdb_movies_list = highest_imdb_movies()
# highest_imdb_movies_data = []

# for i in highest_imdb_movies_list:
#     highest_imdb_movies_data.append(Return_details(i))
#     #print(highest_imdb_movies_data)

@app.route('/')
def home():
    return render_template('home.html',user_logged_in=False)

@app.route('/base')
def base():
    return render_template('base.html') 


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    movie_input = request.form.get("search-box")
    predict = recommend(movie_input)

    # data = []
    # for i in predict:
    #     data.append(function_to_return_link(i))
    #     print(data)

    data = []
    data.append(Return_details(movie_input))
    for i in predict:
        data.append(Return_details(i))
        print(data)

    # return render_template('index.html',movie_name = movie_input, prediction_text=predict,movie_details=data)
    return render_template('recommendation.html', movie_name=movie_input, movie_details=data)


# @app.route('/sign_up', methods=['GET', 'POST'])
# def sign_up():
#     if request.method == 'POST':
#         username = request.form.get("username")
#         email = request.form.get("email")
#         mob_no = request.form.get("mob_no")
#         age = request.form.get("age")
#         password = request.form.get("password")
#         cur = mysql.connection.cursor()
#         cur.execute("INSERT INTO user( `Username`, `Email`, `Mob_no`, `Age`, `Password`) VALUES (%s,%s,%s,%s,%s)",(username, email, mob_no, age, password))
#         mysql.connection.commit()
#         cur.close()
#         print(username, email, mob_no, age, password)
#         print('success')
#         return render_template('sign_in.html')

#     return render_template("sign_up.html")


@app.route('/login/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE Username = %s AND Password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['Username']
            # Redirect to home page
            return render_template('home.html',user_logged_in=True)
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('login.html', msg=msg)

@app.route('/login/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'mobile' in request.form and 'age' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        mob_no = request.form['mobile']
        age = request.form['age']


        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        #cursor = mysql.connection.cursor()
        # cursor.execute('SELECT * FROM user WHERE Email = %s', (email))
        # account = cursor.fetchone()
        # # If account exists show error and validation checks
        # if account:
        #     msg = 'Account already exists!'
        # elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
        #     msg = 'Invalid email address!'
        # elif not re.match(r'[A-Za-z0-9]+', username):
        #     msg = 'Username must contain only characters and numbers!'
        # elif not username or not password or not email:
        #     msg = 'Please fill out the form!'
        # else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            #cursor.execute("INSERT INTO user( `Username`, `Email`, `Mob_no`, `Age`, `Password`) VALUES (%s,%s,%s,%s,%s)",(username, email, mob_no, age, password))
        cursor.execute("INSERT INTO user( `Username`, `Email`, `Mob_no`, `Age`, `Password`) VALUES (%s,%s,%s,%s,%s)",(username, email, mob_no, age, password))
        mysql.connection.commit()
        msg = 'You have successfully registered!'
        

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('sign_up.html', msg=msg)


@app.route("/favorite", methods=['GET', 'POST'])
def index():
    if session:
        print('session exist')
        return "session_exist"
    else:
        print('session not exist')
        return "session_not_exist"
        # return redirect(url_for('login'))
    # if request.method == "POST":
    #     name = request.form["name"]
    #     user_id = session['id']
    #     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    #     cursor.execute("INSERT INTO favorite(`user_id`, `favorite_movie`) VALUES (%s,%s)",(user_id,name))
    #     mysql.connection.commit()
    #     print('favorite movie added to db')
    #     return name + " Hello"
    return render_template("base.html")


@app.route("/account")
def account():
    user_id = session['id']
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # cursor.execute('SELECT * FROM favorite where user_id = %s',(user_id,))
    cursor.execute('SELECT * FROM favorite where user_id = %s',[session['id']])
    output = cursor.fetchall()
    
    cursor.execute('select * from user where id = %s',[session['id']])
    user_details = cursor.fetchone()
    return render_template("account.html",user_logged_in=True,fav_movie_list=output,user_details=user_details)


@app.route("/remove_fav", methods=['GET', 'POST'])
def remove_fav():
    if request.method == "POST":
        name = request.form["name"]
        user_id = session['id']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('DELETE FROM favorite WHERE favorite_movie = %s and user_id = %s', (name,user_id,))
        mysql.connection.commit()
        print('movie removed form favorites')
        return name + " Hello"
    return render_template("account.html",user_logged_in=True,)



if __name__ == "__main__":
    app.run(debug=True)
