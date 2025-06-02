# serve.py
from flask import Flask, send_from_directory
import os

app = Flask(__name__)
RESULTS_DIR = os.path.join("data", "results")


@app.route("/results/<path:filename>")
def download_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/results/")
def list_files():
    files = os.listdir(RESULTS_DIR)
    csv_files = [f for f in files if f.endswith(".csv")]
    return "\n".join(csv_files)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
