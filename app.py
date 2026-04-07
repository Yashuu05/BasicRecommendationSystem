from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/api/signup", methods=["POST"])
def api_signup():
    """Future: Handle user signup logic"""
    return {"message": "Signup endpoint - to be implemented"}, 200

@app.route("/api/signin", methods=["POST"])
def api_signin():
    """Future: Handle user signin logic"""
    return {"message": "Signin endpoint - to be implemented"}, 200

if __name__ == "__main__":
    app.run(debug=True)

