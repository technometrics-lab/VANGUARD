import psycopg2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)

# Initialize the parser
parser = argparse.ArgumentParser(description="Flask API for Grafana Dashboard")

# Add a required argument for API_TOKEN
parser.add_argument("api_token", help="API token for Grafana")

# Parse the command-line arguments
args = parser.parse_args()

# Use the provided API token
API_TOKEN = args.api_token

GRAFANA_URL = "http://localhost:3000"
DASHBOARD_UID = "d0f26f44-b259-4648-b999-565ff2023e4d"
get_url = f"{GRAFANA_URL}/api/dashboards/uid/{DASHBOARD_UID}"
save_url = f"{GRAFANA_URL}/api/dashboards/db"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}


def connect_db():
    return psycopg2.connect(
        dbname="tmd_db",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )


def perform_db_operation(sql, params=None):
    conn = connect_db()
    cur = conn.cursor()

    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        conn.commit()
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        cur.close()
        conn.close()

    return None


def close_tab_response(message):
    response = requests.get(get_url, headers=HEADERS)
    if response.status_code != 200:
        print("Failed to fetch dashboard data.")
        # Handle error
        exit()
    dashboard_data = response.json()["dashboard"]
    response = requests.post(save_url, json={"dashboard": dashboard_data, "overwrite": True}, headers=HEADERS)
    if response.status_code == 200:
        print("Dashboard saved (refreshed) successfully!")
    else:
        print("Failed to save (refresh) dashboard.")

    return f'''
    <html>
        <body onload="window.close();">
            {message}. This tab will close automatically.
        </body>
    </html>
    '''


@app.route('/mark-trivial', methods=['GET'])
def mark_as_trivial():
    keyword = request.args.get('keyword')

    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    sql = "INSERT INTO trivial_keywords (keyword) VALUES (%s) ON CONFLICT (keyword) DO NOTHING"
    error_response = perform_db_operation(sql, (keyword,))
    if error_response:
        return jsonify(error_response)

    return close_tab_response("Keyword ignored")


@app.route('/unmark-trivial', methods=['GET'])
def unmark_as_trivial():
    keyword = request.args.get('keyword')

    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    sql = "DELETE FROM trivial_keywords WHERE keyword = %s"
    error_response = perform_db_operation(sql, (keyword,))
    if error_response:
        return jsonify(error_response)

    return close_tab_response("Keyword unignored")


if __name__ == '__main__':
    app.run(port=8080)
