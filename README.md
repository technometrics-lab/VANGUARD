# VANGUARD

## Set up DB

### Create database

```
sudo apt install postgresql                   # install postgres
sudo -u postgres psql                         # connect to postgres
\password postgres                            # set password to 'admin' (without quotes)
CREATE database tmd_db;                       # create the database
\c tmd_db                                     # connect to the database
```

### Create tables

```
CREATE TABLE repo_urls (
    repo_name TEXT PRIMARY KEY NOT NULL,
    repo_url TEXT NOT NULL
);
```

```
CREATE TABLE repo_stars_cumul (
    time TIMESTAMPTZ NOT NULL,
    repo_name TEXT NOT NULL,
    cumul_stars INT NOT NULL
);
```

```
CREATE TABLE repo_keywords (
    time TIMESTAMPTZ NOT NULL,
    repo_name TEXT NOT NULL,
    keyword TEXT NOT NULL
);
```

```
CREATE TABLE keyword_cooccurrence (
    time TIMESTAMPTZ NOT NULL,
    keyword1 TEXT NOT NULL,
    keyword2 TEXT NOT NULL,
    count INTEGER NOT NULL,
    repo_name TEXT NOT NULL,
    PRIMARY KEY(keyword1, keyword2, time, repo_name)
);
```

```
CREATE TABLE trivial_keywords (
    keyword TEXT PRIMARY KEY
);
```

Quit postgres with `\q`.

## Install the dependencies

Create a virtual environment and activate it, then run `pip install -r requirements.txt`.

Then run `python3 -m spacy download en_core_web_lg`.

## Scrape today's data and store in database

Generate 2 GitHub API tokens from `https://github.com/settings/tokens` and place them in `scripts/modules/imports_analysis.py`.

Create a file with the keywords you want to monitor, one per line, and save it in the `data/keywords` folder.

Move to the scripts directory then run `python3 store_today_data_to_db.py --db_name DB_NAME --keywords_path KEYWORDS_PATH`

e.g. `python3 store_today_data_to_db.py --db_name tmd_db --keywords_path ../data/keywords/fpga.txt`

## Set up the dashboard

### Import dashboard

Follow the instructions [here](https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/) to install Grafana.

- ```sudo /bin/systemctl start grafana-server```
- Navigate to http://localhost:3000/.
- Login with username `admin` and password `admin`.
- Click on "+" at the top right, then "Import dashboard"
- Upload the `GitHub-Insights.json` file
- Add a PostgreSQL data source:
    - Host: `localhost:5432`
    - Database: `tmd_db`
    - User: `postgres`
    - Password: `admin`

### Run local server in the background (for the 'ignore keyword' functionality)

- In Grafana, create a new service account and associated service account token in Home->Administration->Service
  accounts
- Run `python3 ignore_keyword_server.py TOKEN`
