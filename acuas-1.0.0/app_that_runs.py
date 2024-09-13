from flask import Flask, send_from_directory, render_template

# Create the Flask application
app = Flask(__name__, template_folder='acuas-1.0.0')

# Define the route for the home page
@app.route('/')
def home():
    # Render the 'index.html' template located in the 'acuas-1.0.0' folder
    return render_template('index.html')

# Define a route to serve static files from the 'acuas-1.0.0' folder
@app.route('/<path:filename>')
def serve_static(filename):
    # Serve the file requested by the URL from the 'acuas-1.0.0' folder
    return send_from_directory('acuas-1.0.0', filename)

# Run the application if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
