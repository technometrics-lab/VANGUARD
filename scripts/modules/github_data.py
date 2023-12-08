import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from statsmodels.nonparametric.smoothers_lowess import lowess

from .imports_analysis import analyze_imports_history, get_readme_keywords_history, GITHUB_TOKENS
from .utils import get_owner_and_repo


"""
GENERAL
"""


def get_pages_parallel(url, num_workers=50):
    """
    Get all pages of a paginated API endpoint in parallel
    :param url: api endpoint
    :param num_workers: number of threads to use
    :return: pages of the api endpoint
    """

    def get_page(url, page, token):
        headers = {
            "Accept": "application/vnd.github.v3.star+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        response = requests.get(url, headers=headers, params={'page': page, 'per_page': 100})
        return response

    def get_pages(url, pages, token):
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            res = list(pool.map(get_page, [url] * len(pages), pages, [token] * len(pages)))
        # retry failed requests recursively until all succeed
        failed_res = [i for i, r in enumerate(res) if r.status_code == 403]
        if failed_res:
            res = [r for r in res if r.status_code != 403]
            res += get_pages(url, failed_res, token)
        return res

    headers = {
        "Accept": "application/vnd.github.v3.star+json",
        "Authorization": f"Bearer {GITHUB_TOKENS[-1]}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    # find number of pages to request
    r = requests.get(url, headers=headers, params={"per_page": 100})
    num_pages = int(r.headers["Link"].split(",")[1].split(";")[0].split("=")[2][:-1]) if "Link" in r.headers else 1
    # do 50 requests at a time
    page_group_size = 50
    pages = [list(range(i, i + page_group_size)) for i in range(1, num_pages, page_group_size)]
    if num_pages % page_group_size != 0:
        pages = (pages if num_pages % page_group_size == 1 else pages[:-1]) + [
            list(range(num_pages - num_pages % page_group_size + 1, num_pages + 1))]
    # make requests
    res = []
    for i, page in enumerate(pages):
        res += [p.json() for p in get_pages(url, page, GITHUB_TOKENS[i % len(GITHUB_TOKENS)])]
        print(len(res) * 100)
    return [item for sublist in res for item in sublist]


def compute_ts_bins(cumulative, bins, dates):
    """
    Compute the timeseries bins for a given list of dates
    :param cumulative: whether to compute the cumulative or non-cumulative timeseries
    :param bins: bins to fill
    :param dates: list of dates
    :return:
    """
    for date in dates:
        for bin_date in bins:
            if date < bin_date:
                bins[bin_date] += 1
                break
    if cumulative:
        for i in range(1, len(bins)):
            bins[list(bins.keys())[i]] += bins[list(bins.keys())[i - 1]]


def check_rate_limit():
    """
    Check the rate limit for the github api
    """
    url = 'https://api.github.com/rate_limit'
    headers = {
        'Accept': "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKENS[0]}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    r = requests.get(url, headers=headers).json()
    print(json.dumps(r, indent=4, sort_keys=True))


def get_repos_data(repo_urls, dates, divide_by_loc, cumulative_imports, mode_stars, cumulative_forks, relative_stars,
                   keywords, imports=True, force=False, median_smoothing_window=20, impute_median=False):
    """
    Get all the data for a list of repository urls
    :param repo_url: url of the repos
    :param dates: dates to get the data for
    :param divide_by_loc: whether to divide the imports by the number of lines of code
    :param cumulative_imports: whether to compute cumulative imports
    :param cumulative_stars_forks: whether to compute cumulative stars and forks
    :param relative_stars: whether to divide the stars by the smoothed weekly median
    :param keywords: whether to compute the keywords history
    :param force: whether to force recomputing the data
    :return: dict with all the data
    """

    data = {}
    for i, repo_url in enumerate(repo_urls):
        print(i, repo_url)
        repo_name = repo_url.split("/")[-1]

        commits_by_dates = get_commits_at_dates(repo_url, dates, force=force)
        if imports:
            try:
                import_history, _ = analyze_imports_history(repo_url, commits_by_dates, dates,
                                                            divide_by_loc=divide_by_loc,
                                                            debug=False, cumulative=cumulative_imports,
                                                            force=force)
            except Exception as e:
                print("There was an error analyzing the imports history of ", repo_url)
                # raise e
                continue
        stars_ts = get_stars_ts(repo_url, dates, mode=mode_stars)
        forks_ts = get_forks_ts(repo_url, dates, cumulative=cumulative_forks)
        if keywords:
            keyword_history = get_readme_keywords_history(repo_url, commits_by_dates, dates, force=force)
        data[repo_name] = {}
        data[repo_name]["commits_by_dates"] = commits_by_dates
        if imports:
            data[repo_name]["module_usage"] = import_history
        data[repo_name]["stars"] = stars_ts
        data[repo_name]["forks"] = forks_ts
        if keywords:
            data[repo_name]["keyword_history"] = keyword_history

    # weekly medians of star gain
    medians = {date: np.median([data[repo]["stars"][date] for repo in data if data[repo]["stars"][date] > 0]) for
               date
               in dates}
    # smooth medians with LOESS
    time_series_data = pd.DataFrame(list(medians.items()), columns=["Date", "Value"])
    time_series_data["Date"] = pd.to_datetime(time_series_data["Date"])
    time_series_data.set_index("Date", inplace=True)
    time_series_data_filled = time_series_data.fillna(time_series_data.mean())
    span_20_weeks = median_smoothing_window / len(time_series_data_filled)
    smoothed_medians = lowess(time_series_data_filled["Value"], time_series_data_filled.index, frac=span_20_weeks)

    if impute_median:
        first_relevant_median_idx = sum(
            [len([data[repo]["stars"][date] for repo in data if data[repo]["stars"][date] > 0]) <= 4 for
             date
             in dates])
        smoothed_medians[:first_relevant_median_idx] = smoothed_medians[first_relevant_median_idx]

    smoothed_medians = dict(zip(dates, smoothed_medians[:, 1]))

    if relative_stars:  # todo remonter ça
        for repo in data:
            data[repo]["stars"] = {date: stars / smoothed_medians[date] for date, stars in data[repo]["stars"].items()}

    return data, smoothed_medians


"""
COMMITS
"""


def get_all_commits_for_repo(github_url, force=False):
    """
    Get all commits of a repository.
    :param github_url: url of the repository
    :return: list of (hash, message, date) tuples
    """
    owner, repo = get_owner_and_repo(github_url)
    commits_path = f'../data/commits/commits_{repo}.json'
    if not os.path.exists(commits_path) or force:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        raw_commits = get_pages_parallel(url)
        commits = [(d["sha"], d["commit"]["message"], d["parents"][0]["sha"] if d["parents"] else "",
                    d["commit"]["committer"]["date"].split("T")[0]) for d in raw_commits]

        # filter out commits on forks
        parent_sha = commits[0][2]
        filtered_commits = [commits[0]]
        for i in range(1, len(commits)):
            if commits[i][0] == parent_sha:
                filtered_commits.append(commits[i][:2] + commits[i][3:])
                parent_sha = commits[i][2]

        # sort by increasing date
        filtered_commits = sorted(filtered_commits, key=lambda x: x[-1])

        with open(commits_path, 'w') as f:
            json.dump(filtered_commits, f)
    else:
        with open(commits_path, 'r') as f:
            filtered_commits = json.load(f)
    return filtered_commits


def get_commits_at_dates(github_url, dates, force=False):
    """
    Gets a list of commits for a github repo, where each commit is the closest previous commit to the given date.
    :param github_url: url of the repository
    :param dates: list of dates to get commits for
    :param force: whether to force fetching the data
    :return: list of (hash, message, date) tuples
    """
    commits = get_all_commits_for_repo(github_url, force=force)  # list of (commit hash, message, date) tuples
    # print(commits)
    commits_by_dates = []
    for date in dates:
        previous_commit = (None, None)
        for commit in commits:
            if commit[-1] > date:
                break
            previous_commit = commit
        commits_by_dates.append((previous_commit[0], previous_commit[1], date))
    # print(commits_by_dates)
    return commits_by_dates


"""
FORKS
"""


def get_all_forks_for_repo(github_url):
    """
    Get all forks of a repository.
    :param github_url: url of the repository
    :return: list of forks
    """
    owner, repo = get_owner_and_repo(github_url)
    forks_path = f'../data/forks/forks_{repo}.json'
    if not os.path.exists(forks_path):
        url = f"https://api.github.com/repos/{owner}/{repo}/forks"
        raw_forks = get_pages_parallel(url)
        forks = [(d['created_at'].split("T")[0], d["pushed_at"].split("T")[0]) for d in raw_forks]
        forks = sorted(forks, key=lambda x: x[0])
        with open(forks_path, 'w') as f:
            json.dump(forks, f)
    else:
        with open(forks_path, 'r') as f:
            forks = json.load(f)
    return forks


def get_forks_ts(github_url, dates, cumulative):
    """
    Get the timeseries of forks for a given repository.
    :param github_url: url of the repository
    :param dates: list of dates for which to get the timeseries
    :param cumulative: whether to compute the cumulative timeseries
    :return: list of (date, number of forks) tuples
    """
    forks = get_all_forks_for_repo(github_url)
    fork_dates = [f[0] for f in forks]

    fork_bins = {}
    for date in dates:
        fork_bins[date] = 0

    compute_ts_bins(cumulative, fork_bins, fork_dates)

    if not cumulative:
        fork_vals = list(fork_bins.values())
        # fork gain is ratio between current and previous number of forks, accounting for 0 values
        fork_bins = {
            d[0]: d[1] / max(d[1] if (d[1] != 0 and fork_vals[i - 1] == 0) else 1, fork_vals[i - 1] if i > 0 else 1) for
            i, d in enumerate(fork_bins.items())}
        fork_bins[dates[0]] = 0.

    return fork_bins


"""
STARS
"""


def get_all_stars_for_repo(github_url):
    """
    Get all stars of a repository.
    :param github_url: url of the repository
    :return: list of stars
    """
    owner, repo = get_owner_and_repo(github_url)
    stars_path = f'../data/stars/stars_{repo}.json'
    if not os.path.exists(stars_path):
        variables = {"after": None, "owner": owner, "name": repo}
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKENS[0]}",
        }
        stars = []
        while True:
            print(len(stars))
            query = """
                query($after:String, $owner:String!, $name:String!) {
                  repository(owner: $owner, name: $name) {
                    stargazers(first: 100, after: $after) {
                      edges {
                        starredAt
                        cursor
                      }
                    }
                  }
                }
                """
            response = requests.post('https://api.github.com/graphql', json={'query': query, 'variables': variables},
                                     headers=headers)
            if response.status_code == 200:
                if response.json()["data"]["repository"]["stargazers"]["edges"]:
                    stars.extend(response.json()["data"]["repository"]["stargazers"]["edges"])
                    variables["after"] = response.json()["data"]["repository"]["stargazers"]["edges"][-1]["cursor"]
                else:
                    break
            else:
                print("Error:", response.json())
                break
        stars = [s["starredAt"].split("T")[0] for s in stars]
        with open(stars_path, 'w') as f:
            json.dump(stars, f)
    else:
        with open(stars_path, 'r') as f:
            stars = json.load(f)
    return stars


def get_stars_ts(github_url, dates, mode):
    """
    Get the timeseries of stars for a given repository.
    :param github_url: url of the repository
    :param dates: list of dates for which to get the timeseries
    :param mode: cumulative or additive or multiplicative
    :return: list of (date, number of stars) tuples
    """
    stars = get_all_stars_for_repo(github_url)

    star_bins = {}
    for date in dates:
        star_bins[date] = 0

    if mode == "cumulative":
        compute_ts_bins(True, star_bins, stars)
    elif mode == "additive":
        compute_ts_bins(False, star_bins, stars)
        star_bins[dates[0]] = 0
    elif mode == "multiplicative":
        compute_ts_bins(True, star_bins, stars)
        star_vals = list(star_bins.values())
        # star gain is ratio between current and previous number of stars, accounting for 0 values
        star_bins = {
            d[0]: d[1] / max(d[1] if (d[1] != 0 and star_vals[i - 1] == 0) else 1, star_vals[i - 1] if i > 0 else 1) for
            i, d in enumerate(star_bins.items())}
        star_bins[dates[0]] = 0.

    return star_bins


"""
TOPICS
"""


def get_topic_count(topic):
    """
    Get the number of repos on github for a given topic
    :param topic: topic to search for
    :return: number of repos for the given topic
    """
    url = 'https://api.github.com/search/repositories?q=topic:' + topic
    headers = {
        'Accept': "application/vnd.github.mercy-preview+json",
        "Authorization": f"Bearer {GITHUB_TOKENS[0]}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    r = requests.get(url, headers=headers).json()

    if "total_count" in r:
        return r['total_count']
    else:
        print(r)
        return 0


def get_repos_for_topic(topic, num_repos, created_before_dec_2022=False, force=False, min_stars=None,
                        pushed_since=None, search_tags=True, python_only=True):
    """
    Get the num_repos github repos with the most stars whose main language is Python or Jupyter Notebooks for a given
    topic
    :param topic: topic to search for
    :param num_repos: number of repos to get
    :param created_before_dec_2022: whether to only get repos created before December 2022
    :param force: whether to force recomputing the data
    :param min_stars: minimum number of stars for the repos
    :param pushed_since: only get repos pushed since this date
    :param search_tags: whether to search for the topic only in the tags, or in the tags, readme and "about" section
    :param python_only: whether to only get repos whose main language is Python or Jupyter Notebooks
    :return: dict containing the url, description, topics, stars and forks for each repo
    """

    topic_repos_path = f'../data/repos_for_topic/repos_for_topic_{topic}_{num_repos}_{"before_dec_2022" if created_before_dec_2022 else ""}.json'

    if not os.path.exists(topic_repos_path) or force:
        url = (f'https://api.github.com/search/repositories?q={"topic:" if search_tags else ""}{topic}{"+language:python+language:jupyter-notebook" if python_only else ""}'
               f'+pushed:>{pushed_since if pushed_since else "2022-06-01"}{"+stars:>" + str(min_stars) if min_stars else ""}{"+created:<2022-06-01" if created_before_dec_2022 else ""}&sort=stars&order=desc&per_page={min(num_repos,1000)}')

        headers = {
            'Accept': "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_TOKENS[0]}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        params = {"per_page": 100, "page": 1}
        # get the repos, getting several pages if needed
        repos = []
        while len(repos) < min(num_repos,1000):
            r = requests.get(url, headers=headers, params=params).json()
            if "items" in r:
                r = r["items"]
            else:
                print(r)
                return []
            repos += [{"url": repo["html_url"], "description": repo["description"], "topics": repo["topics"],
                       "stars": repo["stargazers_count"], "forks": repo["forks"],
                       "created_at": repo["created_at"].split("T")[0]} for repo in r]
            params["page"] += 1
            if len(r) < 100:
                break
        repos = repos[:num_repos]
        with open(topic_repos_path, 'w') as f:
            json.dump(repos, f)
    else:
        with open(topic_repos_path, 'r') as f:
            repos = json.load(f)
    if len(repos) < num_repos:
        print(f'Only {len(repos)} instead of {num_repos} repos found for {topic}')
    return repos


def get_repo_urls(keywords_base, num_repos, created_before_dec_2022, min_stars=None, pushed_since=None, force=False, search_tags=True, python_only=True):
    """
    Get the repo URLs for the given keywords.
    :param keywords_base: list of keywords
    :param num_repos: number of repos to get
    :param created_before_dec_2022: whether to only get repos created before December 2022
    :param min_stars: minimum number of stars for the repos
    :param pushed_since: only get repos pushed since this date
    :param force: whether to force recomputing the data
    :param search_tags: whether to search for the topic only in the tags, or in the tags, readme and "about" section
    :param python_only: whether to only get repos whose main language is Python or Jupyter Notebooks
    :return: repo URLs, repo metadata
    """
    keywords = keywords_base.copy()
    keywords += [keyword.lower() for keyword in keywords_base]

    github_search_keywords = [keyword.lower().replace(" ", "-") for keyword in keywords_base]

    all_repos = set()
    for keyword in github_search_keywords:
        repos = get_repos_for_topic(keyword, num_repos, created_before_dec_2022=created_before_dec_2022,
                                    min_stars=min_stars, pushed_since=pushed_since, force=force, search_tags=search_tags, python_only=python_only)
        repos = [(repo["url"], repo["stars"], repo["created_at"]) for repo in repos]
        all_repos.update(repos)

    repos = sorted(all_repos, key=lambda x: x[1], reverse=True)[:num_repos]
    repos = [r for r in repos if r[0] not in (
        "https://github.com/FedML-AI/FedML", "https://github.com/XX-net/XX-Net",
        "https://github.com/drduh/macOS-Security-and-Privacy-Guide", "https://github.com/goauthentik/authentik",
        "https://github.com/Consensys/mythril", "https://github.com/nucypher/nucypher")]
    repo_urls = [r[0] for r in repos]

    return repo_urls, repos
