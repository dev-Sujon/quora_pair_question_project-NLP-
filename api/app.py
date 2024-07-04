from flask import Flask,rander_template

app = Flask(__name__)

@app.route('/')
def index():
    return rander_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
