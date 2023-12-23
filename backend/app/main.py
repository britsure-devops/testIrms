from flask import Flask, render_template, request, redirect, url_for, session
import re
import pymssql
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from model.utils2 import InsuranceDataPrediction2
from model.utils5 import InsuranceDataPrediction3


import config  # Make sure you have a config.py with Port_Number defined

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'xyzsdfg'


# Azure SQL Server credentials
db_server = 'britsure.database.windows.net'
db_name = 'britsure_sql'
db_user = 'britsure_admin'
db_password = 'Money@Brit2023'  # Replace with your actual password or use environment variables

# Function to create a database connection and cursor
def get_db_connection():
    conn = pymssql.connect(server=db_server, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    return conn, cursor

html_files = ["chart.html","index8.html", "index5.html"]
current_index = 0

@app.route('/')
def home():
    print('We are in the home route')
    return render_template('chart.html')

# Route to generate and display the pie chart
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    selected_date = request.form['selected_date']

    # Fetch data from Azure SQL Server based on the selected date
    query = "SELECT genuine_count, fraud_count FROM count_of_processed_cases WHERE date = %s"
    conn, cursor = get_db_connection()

    try:
        cursor.execute(query, (selected_date,))
        data = cursor.fetchall()

        # Check if data is present for the selected date
        if not data:
            raise ValueError("No data found for the selected date")

        # Create a pie chart using matplotlib
        labels = ['Genuine', 'Fraud']
        sizes = [data[0][0], data[0][1]]

        # Modify the px.pie call in your Python code
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#0BF49C', '#1e81b0'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Save the plot to a BytesIO object
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)

        # Encode the image data to base64
        img_base64 = base64.b64encode(img_data.read()).decode('utf-8')

        return render_template('chart.html', chart_base64=img_base64)

    except ValueError as ve:
        # Handle the case where no data is found for the selected date
        return render_template('chart.html', error_message=str(ve))

    except Exception as e:
        # Handle other exceptions (e.g., database query errors)
        error_message = f"An error occurred: {str(e)}"
        return render_template('chart.html', error_message=error_message)

    finally:
        # Close the database connection
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    conn, cursor = get_db_connection()
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        cursor.execute('SELECT * FROM app_irms_user_credentials WHERE email_id = %s AND password = %s', (email, password,))
        user = cursor.fetchone()
        if user:
            # Assuming emp_id is the first element in the tuple
            session['loggedin'] = True
            session['emp_id'] = user[0]  # Use index to access elements in the tuple
            session['First_Name'] = user[1]  # Replace with the correct index for First_Name
            session['Last_Name'] = user[2]  # Replace with the correct index for Last_Name
            session['email_id'] = user[3]  # Replace with the correct index for email_id
            message = 'Logged in successfully!'
            return render_template('index8.html', message=message)
        else:
            message = 'Please enter correct email / password!'
    return render_template('login.html', message=message)

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    conn, cursor = get_db_connection()  # Add this line to obtain the cursor
    if request.method == 'POST' and 'empid' in request.form and 'name' in request.form and 'email' in request.form and 'password' in request.form:
        empid = request.form['empid']
        firstname = request.form['name']  # 'First_Name' in HTML, but 'name' in form
        lastname = request.form['lastname']   # 'Last_Name' in HTML, but 'name' in form
        password = request.form['password']
        email = request.form['email']
        
        # Use the obtained cursor for database operations
        cursor.execute('SELECT * FROM app_irms_user_credentials WHERE email_id = %s', (email,))
        account = cursor.fetchone()
        if account:
            message = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            message = 'Invalid email address!'
        else:
            # Adjust the SQL query to match your database schema
            query = 'INSERT INTO app_irms_user_credentials (emp_id, First_Name, Last_Name, email_id, password) VALUES (%s, %s, %s, %s, %s)'
            cursor.execute(query, (empid, firstname, lastname, email, password))
            conn.commit()
            message = 'You have successfully registered!'
    elif request.method == 'POST':
        message = 'Please fill out the form!'
    
    # Close the cursor and connection after using them
    cursor.close()
    conn.close()

    return render_template('register.html', message=message)

@app.route("/predict_fraud1", methods=["GET", "POST"])
def predicted_fraud1():
    if request.method == "GET":
        print("Using GET method")

        # Check if prediction1 has already been made
        if not session.get('prediction1_made', False):
            File_Name = request.args.get("File_Name")
            download_file_name = request.args.get('download_file_name')

            if File_Name is not None:
                insurance_data_subscibe = InsuranceDataPrediction2(File_Name)
                output = insurance_data_subscibe.get_predicted_Fraud_or_geniune()
                print(File_Name)

                # Set the session variable to indicate that prediction1 has been made
                session['prediction1_made'] = True

                if output == 0:
                    return render_template("index8.html", prediction='Genuine')
                else:
                    return render_template("index8.html", prediction='Fraud')
        else:
            # Redirect to next page if prediction1 has already been made
            return redirect(url_for('next_page'))

    return render_template("index8.html")



@app.route('/next')
def next_page():
    global current_index
    if session.get('prediction1_made', False):
        # Allow access to the next page only if prediction1 has been made
        if current_index < len(html_files) - 1:
            current_index += 1
            return render_template(html_files[current_index])
        else:
            return render_template('index5.html')
    else:
        # Set the message in the session and display it in the modal
        message = 'Please choose a file before proceeding to the next page...'
        session['message'] = message
        return render_template('index8.html', message=message)

@app.route("/predict_fraud2", methods=["POST"])
def predicted_fraud2():
    claim_number = request.form.get("claim_number")
    if claim_number:
        insurance_data_subscribe = InsuranceDataPrediction3(int(claim_number))
        output = insurance_data_subscribe.generate_claim_pdf1()
        output1 = insurance_data_subscribe.generate_claim_pdf()
        return render_template("index5.html", output=output or output1)

    return render_template("index5.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)