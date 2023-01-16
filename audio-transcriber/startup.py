from main import app

if __name__ == "__main__":
    app.run(debug=False, port=5000) # debug has to be false or else the program will crash due to lack of GPU memory