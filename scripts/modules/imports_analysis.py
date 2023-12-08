import ast
import json
import logging
import os
import re
import subprocess

import requests

# put 2 tokens in here
GITHUB_TOKENS = ["change_me_1", "change_me_2"]

from .keyword_extraction import SpacyExtractor
from .utils import get_owner_and_repo


def clean_invalid_lines_from_list_of_lines(list_of_lines):
    """
    Filters out invalid lines from notebook
    :param list_of_lines: lines to clean
    :return: cleaned lines
    """
    invalid_starts = ['!', '%']
    valid_python_lines = []
    for line in list_of_lines:
        if not any([line.startswith(x) for x in invalid_starts]):
            valid_python_lines.append(line)
    return valid_python_lines


def get_import_string_from_source(source):
    """
    Extracts lines of code corresponding to imports from source code
    :param source: source code
    :return: import lines
    """
    imports = []
    splitted = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if any([isinstance(node, ast.Import), isinstance(node, ast.ImportFrom)]):
            imports.append(splitted[node.lineno - 1])
    return imports


def color_text(text, pattern, color=3):
    """
    Color the first occurrence of a pattern in a text, used for debugging count_module_usage
    :param text: text in which to color the pattern
    :param pattern: pattern to color
    :param color: color code (between 0 and 9)
    :return: colored text
    """
    lines = text.split("\n")
    for i in range(len(lines)):
        line = lines[i]
        # ignore import lines and comment lines
        if not line.startswith("from") and not line.startswith("import") and not line.lstrip().startswith(":"):
            re_pattern = r"(?<![a-zA-Z_])" + re.escape(pattern) + r"(?![a-zA-Z_])"
            lines[i], num_subs = re.subn(re_pattern, f"\033[3{color}m{pattern}\033[m", line, count=1)
            # only color the first occurrence
            if num_subs == 1:
                break
    replaced_text = "\n".join(lines)
    return replaced_text


def get_imports(contents):
    """
    Extracts imports from source code
    :param contents: source code
    :return: dicts of imported names and imported names to modules
    """
    imported_name_usage = {}  # imported name (e.g. nltk, plt, StandardScaler) -> number of usages
    imported_name2module = {}  # imported name -> corresponding module

    try:
        tree = ast.parse(contents)
    except SyntaxError:
        return imported_name_usage, imported_name2module

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for subnode in node.names:
                if subnode.asname:
                    imported_name_usage[subnode.asname] = 0
                    imported_name2module[subnode.asname] = subnode.name.partition(".")[0]
                else:
                    imported_name_usage[subnode.name] = 0
                    imported_name2module[subnode.name] = subnode.name.partition(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for subnode in node.names:
                if subnode.name != "*" and node.module:  # ignore wildcard imports and relative imports
                    imported_name_usage[subnode.name] = 0
                    imported_name2module[subnode.name] = node.module.partition(".")[0]

    return imported_name_usage, imported_name2module


def get_usages(contents, imported_name_usage, debug):
    """
    Counts the number of usages of each imported name in source code
    :param contents: source code
    :param imported_name_usage: current dict of imported name -> number of usages
    :param debug: whether to print colored source code
    :return: updated dict of imported name -> number of usages
    """
    colored_text = contents

    try:
        tree = ast.parse(contents)
    except SyntaxError:
        return imported_name_usage

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in imported_name_usage:
                imported_name_usage[node.id] += 1
                if debug:
                    colored_text = color_text(colored_text, node.id)

    if debug:
        print(colored_text)

    return imported_name_usage


def analyze_imports_locally(github_url, commit="", debug=False):
    """
    Analyzes usage of imports in a github repo, without making API calls but by cloning the repo locally.
    :param github_url: url of github repo
    :param commit: commit hash
    :param debug: whether to print colored source code to debug
    :return: dicts of module -> number of usages and module -> imported names -> usages
    """
    imported_name_usage = {}  # imported name (e.g. nltk, plt, StandardScaler) -> number of usages
    imported_name2module = {}  # imported name -> corresponding module
    ignore_dirs = [".hg", ".svn", ".git", ".tox", "__pycache__", "env", "venv", ".ipynb_checkpoints", ".idea", "docs",
                   "test", "tests", "data", "bin", "doc", "example", "examples", "sample", "samples", "img", "imgs",
                   "image", "images", "asset", "assets", "dist", "build"]
    # where to clone the repos
    clone_dir = "../data/cloned_repos"
    candidates = []

    owner, repo = get_owner_and_repo(github_url)

    # ssh url to clone the repo
    a = f"git@github.com:{owner}/{repo}.git"
    b = f"{clone_dir}/{repo}"
    # clone the repo if it doesn't exist yet
    if not os.path.exists(b):
        subprocess.run(["git", "clone", a, b], check=True, text=True)
    if commit:
        subprocess.run(["git", "-C", b, "checkout", commit], check=True, text=True)

    walk = os.walk(b)
    for root, dirs, files in walk:
        candidates, ipynb_files, py_files = filter_files(candidates, dirs, files, ignore_dirs, root)
        for file_name in py_files:
            file_name = os.path.join(root, file_name)

            try:
                with open(file_name, "r") as f:
                    contents = f.read()
                    new_imported_name_usage, new_imported_name2module = get_imports(contents)
                    imported_name_usage.update(new_imported_name_usage)
                    imported_name2module.update(new_imported_name2module)
            except Exception as exc:
                logging.error("Failed on file: %s" % file_name)
                raise exc

        for file_name in ipynb_files:
            file_name = os.path.join(root, file_name)
            with open(file_name, "r") as f:

                try:
                    contents = f.read()
                    nb = json.loads(contents)
                    nb_imports = []
                    for n_cell, cell in enumerate(nb['cells']):
                        if cell['cell_type'] == 'code':
                            valid_lines = clean_invalid_lines_from_list_of_lines(cell['source'])
                            source = ''.join(valid_lines)
                            nb_imports += get_import_string_from_source(source)
                    nb_imports = [i.lstrip() for i in nb_imports]
                    nb_imports = "\n".join(nb_imports)
                    new_imported_name_usage, new_imported_name2module = get_imports(nb_imports)
                    imported_name_usage.update(new_imported_name_usage)
                    imported_name2module.update(new_imported_name2module)
                except Exception as e:
                    print(
                        "Exception occurred while working on file {}".format(file_name))
                    # raise e

    walk = os.walk(b)
    for root, dirs, files in walk:
        candidates, ipynb_files, py_files = filter_files(candidates, dirs, files, ignore_dirs, root)

        for file_name in py_files:
            file_name = os.path.join(root, file_name)

            try:
                with open(file_name, "r") as f:
                    contents = f.read()
                    imported_name_usage = get_usages(contents, imported_name_usage, debug=debug)
            except Exception as exc:
                logging.error("Failed on file: %s" % file_name)
                raise exc

        for file_name in ipynb_files:
            file_name = os.path.join(root, file_name)
            with open(file_name, "r") as f:

                try:
                    contents = f.read()
                    nb = json.loads(contents)
                    sources = []
                    for n_cell, cell in enumerate(nb['cells']):
                        if cell['cell_type'] == 'code':
                            valid_lines = clean_invalid_lines_from_list_of_lines(cell['source'])
                            source = ''.join(valid_lines)
                            sources.append(source)
                    source_code = "\n".join(sources)
                    imported_name_usage = get_usages(source_code, imported_name_usage, debug=debug)
                except Exception as e:
                    print(
                        "Exception occurred while working on file {}".format(file_name))
                    # raise e

    module_usage = {}  # module -> number of usages
    for name, usages in imported_name_usage.items():
        module = imported_name2module[name]
        if module not in module_usage:
            module_usage[module] = 0
        module_usage[module] += usages

    with open("../data/package_names/stdlib", "r") as f:
        stl = {x.strip() for x in f}

    with open("../data/package_names/import2pip", "r") as f:
        mapp = dict(x.strip().split(":") for x in f)

    # remove packages from stdlib and map to their actual names
    module_usage = {mapp.get(key, key): value for key, value in module_usage.items() if
                    key not in candidates and key not in stl and value > 0}

    module_usage = dict(sorted(module_usage.items(), key=lambda x: -x[1]))

    module_usage_breakdown = {}  # module -> imported name -> number of usages
    for name, usages in imported_name_usage.items():
        module = imported_name2module[name]
        if module not in module_usage_breakdown:
            module_usage_breakdown[module] = {}
        module_usage_breakdown[module][name] = usages

    module_usage_breakdown = {mapp.get(key, key): value for key, value in module_usage_breakdown.items() if
                              key not in candidates and key not in stl and sum(value.values()) > 0}

    for module, name_usage in module_usage_breakdown.items():
        module_usage_breakdown[module] = dict(
            sorted({key: value for key, value in name_usage.items() if value > 0}.items(), key=lambda x: -x[1]))

    module_usage_breakdown = dict(sorted(module_usage_breakdown.items(), key=lambda x: -sum(x[1].values())))

    return module_usage, module_usage_breakdown


def filter_files(candidates, dirs, files, ignore_dirs, root):
    """
    Filter files to only include .py and .ipynb files, and update candidates.
    """
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    candidates.append(os.path.basename(root))
    py_files = [fn for fn in files if os.path.splitext(fn)[1] == ".py"]
    candidates += [os.path.splitext(fn)[0] for fn in py_files]
    ipynb_files = [fn for fn in files if os.path.splitext(fn)[1] == ".ipynb"]
    candidates += [os.path.splitext(fn)[0] for fn in ipynb_files]
    return candidates, ipynb_files, py_files


def count_loc(github_url, commits):
    """
    Count lines of code in a repository at a certain commit.
    :param github_url: url of the repository
    :param commits: list of commit hashes for which to count lines of code
    :return: dictionary mapping commit hashes to lines of code
    """
    # where to clone the repos
    clone_dir = "../data/cloned_repos"
    owner, repo = get_owner_and_repo(github_url)
    # ssh url to clone the repo
    a = f"git@github.com:{owner}/{repo}.git"
    b = f"{clone_dir}/{repo}"
    # clone the repo if it doesn't exist yet
    if os.path.exists(b) and not commits:
        # !rm -rf "$b"
        subprocess.run(["git", "clone", a, b], check=True, text=True)
    elif not os.path.exists(b):
        subprocess.run(["git", "clone", a, b], check=True, text=True)
    loc_at_commit = {}
    if not commits:
        res = subprocess.run(["tokei", "--output", "json", b], check=True, text=True, capture_output=True)
        json_res = json.loads(res.stdout)
        python_loc = json_res["Python"]["code"] if "Python" in json_res else 0
        ipynb_loc = sum([nb["stats"]["code"] for nb in
                         json_res["Jupyter Notebooks"]["children"]["Python"]]) if "Jupyter Notebooks" in json_res else 0
        total_loc = python_loc + ipynb_loc
        loc_at_commit["latest"] = total_loc
    else:
        for i, commit in enumerate(commits):
            if commit:
                # try:
                subprocess.run(["git", "-C", b, "checkout", commit], check=True, text=True)
                # except Exception: # commit is corrupt or doesn't exist, extremely rare
                #     loc_at_commit[commit] = 0
                #     continue
                res = subprocess.run(["tokei", "--output", "json", b], check=True, text=True, capture_output=True)
                json_res = json.loads(res.stdout)
                python_loc = json_res["Python"]["code"] if "Python" in json_res else 0
                ipynb_loc = sum([nb["stats"]["code"] for nb in json_res["Jupyter Notebooks"]["children"][
                    "Python"]]) if (
                        "Jupyter Notebooks" in json_res and "children" in json_res["Jupyter Notebooks"] and "Python" in
                        json_res["Jupyter Notebooks"]["children"]) else 0
                total_loc = python_loc + ipynb_loc
                loc_at_commit[commit] = total_loc

    return loc_at_commit


def analyze_imports_history(github_url, commits, dates, divide_by_loc=True, debug=False, cumulative=True, force=False):
    """
    Analyze imports of a repository over time.
    :param github_url: url of the repository
    :param commits: list of (commit hash, message, date) tuples
    :param divide_by_loc: whether to divide the number of usages by the number of lines of code at that commit
    :param debug: whether to print debug information
    :param cumulative: whether to count the number of usages cumulatively or to count as deltas between commits
    :param force: whether to force recompute the import history
    :return: module_usage_history, module_usage_breakdown_history, sorted by total number of usages
    """
    _, repo = get_owner_and_repo(github_url)
    import_history_path = f'../data/import_history/import_histories_{repo}_{dates[0]}_{dates[-1]}{"_not_divide_by_loc" if not divide_by_loc else ""}.json'
    if not os.path.exists(import_history_path) or force:

        module_usage_history = {}  # module -> date -> number of usages
        module_usage_breakdown_history = {}  # module -> date -> imported name -> number of usages

        if divide_by_loc:
            locs = count_loc(github_url, [commit[0] for commit in commits])

        # analyze imports for each commit
        for commit in commits:
            if commit[0]:
                module_usage, module_usage_breakdown = analyze_imports_locally(github_url, commit=commit[0],
                                                                               debug=debug)
                for module, count in module_usage.items():
                    if module not in module_usage_history:
                        module_usage_history[module] = {}
                    module_usage_history[module][commit[-1]] = (
                        count / locs[commit[0]] if locs[commit[0]] > 0 else 0) if divide_by_loc else count
                for module, name in module_usage_breakdown.items():
                    if module not in module_usage_breakdown_history:
                        module_usage_breakdown_history[module] = {}
                    if commit[-1] not in module_usage_breakdown_history[module]:
                        module_usage_breakdown_history[module][commit[-1]] = {}
                    for name_, count in name.items():
                        module_usage_breakdown_history[module][commit[-1]][name_] = (count / locs[
                            commit[0]] if locs[commit[0]] > 0 else 0) if divide_by_loc else count

        # sort modules by total number of usages
        module_usage_history = dict(sorted(module_usage_history.items(), key=lambda x: -sum(x[1].values())))
        module_usage_breakdown_history = dict(
            sorted(module_usage_breakdown_history.items(), key=lambda x: -sum(sum(y.values()) for y in x[1].values())))

        with open(import_history_path, 'w') as f:
            json.dump((module_usage_history, module_usage_breakdown_history), f)
    else:
        with open(import_history_path, 'r') as f:
            module_usage_history, module_usage_breakdown_history = json.load(f)

    # fill in missing dates with 0
    for module, usage in module_usage_history.items():
        for date in dates:
            if date not in usage:
                usage[date] = 0
    # TODO same for module_usage_breakdown_history

    for module, usage in module_usage_history.items():
        module_usage_history[module] = dict(sorted(usage.items(), key=lambda x: x[0]))

    if not cumulative:
        for module, usage in module_usage_history.items():
            usage_vals = list(module_usage_history[module].values())
            module_usage_history[module] = {
                date: usage[date] - usage_vals[i - 1] if i > 0 else (0 if date == dates[0] else usage[date]) for i, date
                in
                enumerate(usage)}
        # TODO same for module_usage_breakdown_history

    return module_usage_history, module_usage_breakdown_history


"""
README KEYWORDS
"""

extractor = SpacyExtractor('en_core_web_lg', 100_000, False)
# different possible names for README files to try
readme_file_names = ["README.md", "readme.md", "README", "readme", "README.txt", "readme.txt", "README.rst",
                     "readme.rst", "README.MD", "readme.MD", "README.TXT", "readme.TXT", "README.RST", "readme.RST",
                     "README_en.md", "readme_en.md", "README_EN.md", "readme_EN.md",
                     "README_EN.txt", "readme_EN.txt", "README_EN.rst", "readme_EN.rst", "README_en.txt",
                     "readme_en.txt", "README_en.rst", "readme_en.rst",
                     "README_EN", "readme_EN", "README_en", "Readme.md", "readMe.md", "ReadMe.md", "README.Md",
                     "readme.Md", "ReadME.md"
                     ]


def get_readme_keywords_history(repo_url, commits, dates, force=False):
    """
    Extracts keywords from READMEs of a repository over time.
    :param repo_url: url of the repository
    :param commits: list of (commit hash, message, date) tuples
    :param dates: dates for which to extract keywords
    :param force: whether to force recompute the keyword history
    :return: dict of date -> keywords
    """
    owner, repo = get_owner_and_repo(repo_url)
    kw_history_path = f'../data/readme_kw_history/readme_kw_history_{repo}_{dates[0]}_{dates[-1]}.json'

    no_readmes_found_flag = True

    if not os.path.exists(kw_history_path) or force:
        readme_kw_history = {}
        cleaned_readme_tokens_history = {}

        clone_dir = f"../data/cloned_repos/{repo}"
        if not os.path.exists(clone_dir):
            a = f"git@github.com:{owner}/{repo}.git"
            subprocess.run(["git", "clone", a, clone_dir], check=True, text=True)

        for commit in commits:
            if commit[0]:
                subprocess.run(["git", "-C", clone_dir, "checkout", commit[0]], check=True, text=True)
                i = 0
                readme_path = ""
                for i in range(len(readme_file_names)):
                    readme_path = f"{clone_dir}/{readme_file_names[i]}"
                    if os.path.exists(readme_path):
                        break
                else:
                    print(f"Could not find README for {repo_url} at commit {commit[0]}")
                    readme_kw_history[commit[-1]] = []
                    cleaned_readme_tokens_history[commit[-1]] = ""
                    continue
                with open(readme_path, 'r') as f:
                    readme_text = f.read()
                readme_kws, cleaned_tokens = extractor.extract_keywords(readme_text)
                no_readmes_found_flag = False
                readme_kw_history[commit[-1]] = readme_kws
                cleaned_readme_tokens_history[commit[-1]] = cleaned_tokens
        if no_readmes_found_flag:
            print(f"Could not find any READMEs for {repo_url}")
        with open(kw_history_path, 'w') as f:
            json.dump(readme_kw_history, f)
        with open(
                f'../data/cleaned_readme_tokens_history/cleaned_readme_tokens_history_{repo}_{dates[0]}_{dates[-1]}.json',
                'w') as f:
            json.dump(cleaned_readme_tokens_history, f)
    else:
        with open(kw_history_path, 'r') as f:
            readme_kw_history = json.load(f)

    return readme_kw_history


def get_readme_keywords_today(repo_url):
    """
    Extracts keywords from README of a repository today.
    :param repo_url: url of the repository
    :return: keywords, cleaned_tokens
    """
    owner, repo = get_owner_and_repo(repo_url)

    headers = {
        "Accept": "application/vnd.github.v3.star+json",
        "Authorization": f"Bearer {GITHUB_TOKENS[-1]}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    for filename in readme_file_names:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filename}"

        response = requests.get(api_url, headers=headers)

        # If the response code is not 404 (not found), then process the response
        if response.status_code != 404:
            try:
                json_data = response.json()
                content = requests.get(json_data["download_url"]).text
                readme_kws, cleaned_tokens = extractor.extract_keywords(content)
                return readme_kws, cleaned_tokens
            except Exception as e:
                print(f"Could not find README for {repo_url}")
                return [], ""

    print(f"Could not find any READMEs for {repo_url}")
    return [], ""
