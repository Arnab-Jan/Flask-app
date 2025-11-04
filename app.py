import numpy as np
from flask import Flask, request, render_template, jsonify

# --- 1. FLASK LESSON: IMPORTS ---
# Flask: The main class we use to create our web application.
# request: An object that holds all the data coming *from* the user (like form data).
# render_template: A function that combines an HTML file (from the 'templates' folder)
#                  with Python data to produce a final HTML page.
# jsonify: A function to turn Python dictionaries into a JSON response (useful for APIs).

# Import our Neural_Network class from the other file
from nn_model import Neural_Network

# --- 2. FLASK LESSON: APP CREATION ---
# Create an "instance" of the Flask application.
# __name__ is a special Python variable that gives Flask a name.
app = Flask(__name__)

# --- 3. MODEL PREPARATION ---
# We must train the model *once* when the server starts, not on every request!
# We'll store the trained model in a global variable.

print("--- TRAINING MODEL (this happens once per server start) ---")

# Define the XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the network instance
# We'll make the hidden layer a bit bigger (e.g., 4 nodes) to ensure it can solve XOR
nn = Neural_Network(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
nn.fit(X, y, epochs=10000)

print("--- MODEL TRAINED AND READY! ---")


# --- 4. FLASK LESSON: ROUTES ---
# A "route" is a URL on your website. We use a Python "decorator" (@)
# to tell Flask: "When a user visits THIS URL, run THIS Python function."

# This is the main page (e.g., http://127.0.0.1:5000/)
@app.route('/')
def home():
    # This function just shows the user the main HTML page.
    # It passes an empty 'prediction_text' so the page loads cleanly.
    return render_template('index.html', prediction_text="")

# This route handles the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # --- 5. FLASK LESSON: REQUESTS & FORMS ---
    # We check if the request method is 'POST', which means a form was submitted.
    if request.method == 'POST':
        try:
            # 'request.form' is a dictionary holding all the data from our HTML form.
            # We get the values by their 'name' attribute (e.g., <select name="input1">)
            # We must convert them from strings to numbers (float or int).
            input1 = float(request.form['input1'])
            input2 = float(request.form['input2'])

            # Prepare the data for our model: a single 2D numpy array
            input_data = np.array([[input1, input2]])

            # Use our trained model (nn) to get a prediction
            prediction_raw = nn.predict(input_data)
            
            # Extract the single number from the result
            prediction_value = prediction_raw[0][0]
            
            # Round the result to get a clean 0 or 1
            prediction_rounded = np.round(prediction_value)

            # Create a friendly text string to send back to the user
            result_text = f"Input: [{int(input1)}, {int(input2)}]  ->  Prediction: {prediction_value:.4f}  ->  Rounds to: {int(prediction_rounded)}"

            # --- 6. FLASK LESSON: RENDERING WITH CONTEXT ---
            # We re-render the *same* HTML page, but this time we pass our
            # 'result_text' to the 'prediction_text' variable in the HTML.
            return render_template('index.html', prediction_text=result_text)

        except Exception as e:
            # Handle any errors gracefully
            error_text = f"Error processing input: {e}"
            return render_template('index.html', prediction_text=error_text)

# This route is optional: it provides a way to get predictions as JSON.
# This is what you would use if another program (not a human user)
# wanted to use your model.
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json # Get data sent as JSON
        input1 = float(data['input1'])
        input2 = float(data['input2'])
        
        input_data = np.array([[input1, input2]])
        prediction_raw = nn.predict(input_data)
        prediction_value = prediction_raw[0][0]

        # Return a JSON response
        return jsonify({
            'input': [input1, input2],
            'prediction_raw': prediction_value,
            'prediction_rounded': int(np.round(prediction_value))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- 7. FLASK LESSON: RUNNING THE APP ---
# This standard Python line checks if the script is being run directly
# (not imported as a module).
if __name__ == '__main__':
    # app.run() starts the web server.
    # debug=True is great for development:
    # 1. It automatically restarts the server when you save the file.
    # 2. It shows detailed error messages in the browser.
    # **NEVER** use debug=True in a real, live (production) application!
    app.run(debug=True)
