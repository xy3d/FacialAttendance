import cv2, os, json, mysql.connector, time, logging
import numpy as np
from datetime import datetime, date, timedelta
from flask import Flask, render_template, Response, jsonify, request
from mysql.connector import Error
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import face_recognition
import speech_recognition as sr

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# cap = cv2.VideoCapture(0)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    connection = mysql.connector.connect(
            host='JAYED',
            database='attendance',
            user='jayed',
            password='1234'
    )  # Replace with your actual MySQL connection details

    cursor = connection.cursor()

    # Fetch data from the 'attendance' table
    cursor.execute('SELECT * FROM attendance')
    attendance_data = cursor.fetchall()

    # Fetch data from the 'time' table
    cursor.execute('SELECT * FROM time')
    time_data = cursor.fetchall()

    # Fetch data from the 'anomaly' table
    cursor.execute('SELECT * FROM anomaly')
    anomaly_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template('dashboard.html', attendance_data=attendance_data, time_data=time_data, anomaly_data=anomaly_data)

# Function to fetch data from a table in the database
def get_table_data(connection, table_name):
    try:
        cursor = connection.cursor()
        select_query = f"SELECT * FROM {table_name}"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        data = []
        for row in rows:
            data.append(dict(zip(column_names, row)))
        return data
    except Error as e:
        print(f"The error '{e}' occurred while fetching data from the table {table_name}")
        return []

# Function to establish a connection with the MySQL database
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='JAYED',
            database='attendance',
            user='jayed',
            password='1234'
        )
        # print("Connected to MySQL database")
    except Error as e:
        print(f"The error '{e}' occurred while connecting to the MySQL database")
    return connection

# Function to create the attendance and time tables if they don't exist
def create_tables(connection):
    try:
        cursor = connection.cursor()
        create_attendance_table_query = """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            time DATETIME NOT NULL
        )
        """
        cursor.execute(create_attendance_table_query)
        create_time_table_query = """
        CREATE TABLE IF NOT EXISTS time (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            intime DATETIME NOT NULL,
            outtime DATETIME,
            totaltime VARCHAR(10)
        )
        """
        cursor.execute(create_time_table_query)
        create_anomaly_table_query = """
        CREATE TABLE IF NOT EXISTS anomaly (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            intime DATETIME NOT NULL,
            outtime DATETIME,
            totaltime VARCHAR(10),
            date DATE NOT NULL,
            UNIQUE KEY unique_anomaly (name, date)
        )
        """
        cursor.execute(create_anomaly_table_query)
        
        # Retrieve all records from the 'time' table
        select_all_time_records_query = "SELECT * FROM time"
        cursor.execute(select_all_time_records_query)
        time_records = cursor.fetchall()

        # Check each record for anomaly and insert into the 'anomaly' table if required
        for record in time_records:
            time_id, name, intime, outtime, total_time = record
            if total_time and is_anomaly(total_time) and (datetime.now() - intime) > timedelta(hours=20):
                # Check if the anomaly entry already exists based on name and date
                select_existing_anomaly_query = """
                SELECT id FROM anomaly WHERE name = %s AND date = %s
                """
                cursor.execute(select_existing_anomaly_query, (name, intime.date()))
                existing_anomaly = cursor.fetchone()
                
                if not existing_anomaly:
                    # Insert the anomaly record
                    insert_anomaly_query = """
                    INSERT INTO anomaly (name, intime, outtime, totaltime, date)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_anomaly_query, (name, intime, outtime, total_time, intime.date()))
                    connection.commit()
                    print("Anomaly recorded successfully")

        connection.commit()
        print("Attendance, Time, and Anomaly tables created successfully")
    except Error as e:
        print(f"The error '{e}' occurred while creating the tables")

# Function to retrieve attendance data from the database
def get_attendance_data(connection):
    try:
        cursor = connection.cursor()
        select_query = """
        SELECT *
        FROM attendance
        ORDER BY time DESC
        """
        cursor.execute(select_query)
        attendance_data = cursor.fetchall()
        return attendance_data
    except Error as e:
        print(f"The error '{e}' occurred while retrieving attendance data")

# Function to calculate the total time in the office
def calculate_total_time(intime, outtime):
    if outtime is None:
        return None

    time_format = '%H:%M'
    intime_str = intime.strftime('%Y-%m-%d %H:%M:%S')
    outtime_str = outtime.strftime('%Y-%m-%d %H:%M:%S')

    intime_obj = datetime.strptime(intime_str, '%Y-%m-%d %H:%M:%S')
    outtime_obj = datetime.strptime(outtime_str, '%Y-%m-%d %H:%M:%S')

    # Calculate the time difference
    time_diff = outtime_obj - intime_obj

    # Format the total time as HH:MM
    total_time = (datetime.min + time_diff).time().strftime(time_format)

    return total_time

def is_anomaly(total_time):
    if total_time is not None:
        hours, minutes = total_time.split(':')
        total_hours = int(hours) + int(minutes) / 60
        if total_hours < 8:
            return True
    return False

def takeAttendance(name, connection):
    today = date.today()
    now = datetime.now()

    try:
        cursor = connection.cursor()

        # Check if the employee has an existing entry for the current date
        select_query = """
        SELECT id, intime, outtime, totaltime
        FROM time
        WHERE name = %s
        AND DATE(intime) = %s
        """
        cursor.execute(select_query, (name, today))
        result = cursor.fetchone()

        if result:
            # If the employee already has an entry, update the outtime
            time_id, intime, _, total_time = result
            update_query = """
            UPDATE time
            SET outtime = %s
            WHERE id = %s
            """
            cursor.execute(update_query, (now, time_id))
            connection.commit()
        else:
            # If the employee doesn't have an entry for the current date, insert a new row with intime
            insert_query = """
            INSERT INTO time (name, intime)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (name, now))
            connection.commit()

            # Retrieve the newly inserted time_id
            time_id = cursor.lastrowid

        # Get the updated intime and outtime for calculating total time
        cursor.execute(select_query, (name, today))
        updated_result = cursor.fetchone()
        _, updated_intime, updated_outtime, total_time = updated_result

        # Calculate the total time
        total_time = calculate_total_time(updated_intime, updated_outtime)

        # Update the totaltime column
        update_total_time_query = """
        UPDATE time
        SET totaltime = %s
        WHERE id = %s
        """
        cursor.execute(update_total_time_query, (total_time, time_id))
        connection.commit()

        # Insert the attendance record
        insert_attendance_query = """
        INSERT INTO attendance (name, time)
        SELECT %s, %s
        FROM DUAL
        WHERE NOT EXISTS (
            SELECT id
            FROM attendance
            WHERE name = %s
            AND time > DATE_SUB(NOW(), INTERVAL 600 SECOND)
        )
        """
        cursor.execute(insert_attendance_query, (name, now, name))
        connection.commit()

    except Error as e:
        print(f"The error '{e}' occurred while inserting attendance record")
        
# Function to capture images
def capture_images(name, output_folder):
    # Create a directory for the person if it doesn't exist
    person_folder = os.path.join(output_folder, name)
    os.makedirs(person_folder, exist_ok=True)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Capture 20 images
    count = 0
    while count < 10:
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Capture Images', frame)

        # Introduce a delay of 0.5 seconds
        time.sleep(0.5)

        # Save the frame as an image
        image_path = os.path.join(person_folder, f'{name}_{count}.jpg')
        cv2.imwrite(image_path, frame)

        count += 1

        # Break the loop when 20 images are captured
        if count == 20:
            break

        # Break the loop and move to the next person when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy the window
    cap.release()
    cv2.destroyAllWindows()
    
# Route for registering a new person
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        if name:
            capture_images(name, 'data')
            return "Images captured and registered successfully!"
        else:
            return "Please provide a valid name."
    return render_template('register.html')

        

def gen():
    data = []
    dir_path = 'data'
    
    known_face_encodings = []
    known_face_names = []

    for person_folder in os.listdir(dir_path):
        person_folder_path = os.path.join(dir_path, person_folder)
        if os.path.isdir(person_folder_path):
            for image_filename in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_filename)
                try:
                    known_image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(known_image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_folder)
                except IndexError:
                    print(f"No face found in {image_path}. Skipping...")


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        success, img = cap.read()
        # print("Capture Success:", success)
        # print("Image Shape:", img.shape if img is not None else None)
        
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Establish a new database connection and create a cursor for each iteration
        connection = create_connection()
        cursor = connection.cursor()
        
        if img is not None and success:
            # Continue with face detection and other processing
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)
        
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            y1, x2, y2, x1 = face_location
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, name, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Retrieve the password for the recognized name from the database
            if name != "Unknown":
                cursor.execute("SELECT pwd FROM emp WHERE name = %s", (name,))
                password_from_db = cursor.fetchone()[0]
                
                # Verify the password using voice recognition
                password_verified = verify_voice_password(password_from_db)
                
                if password_verified:
                    print("Password verified successfully.")
                    print(f"Detected: {name}")
                    takeAttendance(name, connection)
                    # Proceed with attendance and other actions
                else:
                    print("Password verification failed.")
                    

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.putText(img, f"{person_folder} ({matching_percentage:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            match_found = True

            # Send detection status through WebSocket
            detection_status = {
                "detected": name != "Unknown",
                "name": name,
                "pass": password_verified
            }
            socketio.emit("detection_status", json.dumps(detection_status))
            
            # # Delay for 3 seconds
            # time.sleep(3)    
            
            
        ret, buffer = cv2.imencode('.jpg', img)

        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            print("Failed to encode frame using cv2.imencode")
    else:
        print("Failed to capture frame from the camera")

        cap.release()
        


        
def verify_voice_password(password_from_db):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say your password:")
        audio = recognizer.listen(source)

    try:
        recognized_text = recognizer.recognize_google(audio)
        print("You said:", recognized_text)
        
        # Compare the recognized text with the password from the database
        if recognized_text == password_from_db:
            return True
        else:
            return False
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return False


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Event handler for WebSocket connection
@socketio.on('connect')
def handle_connect():
    print('Client connected')

# Event handler for WebSocket disconnection
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    connection = create_connection()
    create_tables(connection)
    socketio.run(app, debug=True, port=5000)
