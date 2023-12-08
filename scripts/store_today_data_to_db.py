import argparse
import pickle
import re
from collections import defaultdict
from datetime import date

import numpy as np
import psycopg2
from scipy.stats import chi2_contingency

from modules.cooccurrence import compute_cooccurrences_123
from modules.github_data import get_repo_urls
from modules.imports_analysis import get_readme_keywords_today


def compute_volcano_coords(readme_freqs):
    """
    Computes volcano plot coordinates for each word in the given dictionary of word frequencies.
    :param readme_freqs: dictionary of word frequencies
    :return: coordinates for each word in the given dictionary
    """
    word_coords = {}
    total_readme = sum(readme_freqs.values())
    with open('../data/freq_book.pkl', 'rb') as file:
        book_freqs = dict(pickle.load(file))
    total_book = sum(book_freqs.values())

    for word in set(readme_freqs) | set(book_freqs):
        readme_count = readme_freqs.get(word, 0)
        book_count = book_freqs.get(word, 0)

        # Calculate Log2 Fold-Change
        fold_change = (readme_count + 1) / (book_count + 1)  # +1 to avoid division by zero
        log2_fold_change = np.log2(fold_change)

        # Calculate p-value using Chi-squared test
        _, p_value, _, _ = chi2_contingency([[readme_count, total_readme - readme_count],
                                             [book_count, total_book - book_count]])
        word_coords[word] = (log2_fold_change, p_value)
    return word_coords


def main():
    parser = argparse.ArgumentParser(description="Process GitHub repository data")
    parser.add_argument("--db_name", required=True, help="Database name")
    parser.add_argument("--keywords_path", required=True, help="Path to keywords file")

    args = parser.parse_args()

    db_name = args.db_name
    keywords_path = args.keywords_path

    with open(keywords_path, 'r') as file:
        keywords = [line.strip() for line in file]

    repo_urls, repos = get_repo_urls(keywords, 100_000, False, min_stars=50, pushed_since="2023-01-01", force=True,
                                     search_tags=False, python_only=False)
    repo_names = [repo[0].split('/')[-1] for repo in repos]

    print(f"Found {len(repo_names)} repos today!")

    # Insert repo urls into database
    conn = psycopg2.connect(
        dbname=db_name,
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    for repo_name, repo_url in zip(repo_names, repo_urls):
        cur.execute("""
		INSERT INTO repo_urls (repo_name, repo_url) VALUES (%s, %s)
		ON CONFLICT (repo_name) 
		DO UPDATE SET repo_url = EXCLUDED.repo_url;
	    """, (repo_name, repo_url))
    conn.commit()
    cur.close()
    conn.close()

    print("Computing volcano plot coordinates...")

    readme_kws = {}
    readme_tokens = {}
    for repo_url, repo_name in zip(repo_urls, repo_names):
        kws, cleaned_tokens = get_readme_keywords_today(repo_url)
        if kws:
            readme_kws[repo_name] = [kw[0] for kw in kws]
            readme_tokens[repo_name] = [(group, [e[1] for e in kws if e[0] == group]) for group in
                                        set([e[0] for e in kws])], cleaned_tokens  # group by clean keyword

    readme_freqs = {}
    for repo_name in repo_names:
        if repo_name in readme_kws:
            for kw in readme_kws[repo_name]:
                readme_freqs[kw] = readme_freqs.get(kw, 0) + 1

    word_coords = compute_volcano_coords(readme_freqs)

    filtered_words = {word: coords for word, coords in word_coords.items()}

    for repo_name in repo_names:
        if repo_name in readme_kws:
            readme_kws[repo_name] = [kw for kw in readme_kws[repo_name] if kw in filtered_words]

    print("Computing co-occurrences...")
    for repo_name in readme_tokens:
        readme_tokens[repo_name] = [t for t in readme_tokens[repo_name][0] if t[0] in filtered_words], \
            readme_tokens[repo_name][1]

    cooccurrences_by_repo = {}
    for repo_name in readme_tokens:
        cooccurrence = compute_cooccurrences_123(readme_tokens[repo_name][1], readme_tokens[repo_name][0],
                                                 defaultdict(int))
        cooccurrences_by_repo[repo_name] = cooccurrence

    date_obj = date.today().strftime("%Y-%m-%d")

    conn = psycopg2.connect(
        dbname=db_name,
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    print("Inserting keywords into database...")
    for repo_name in repo_names:
        if repo_name in readme_kws:
            print(repo_name)
            keywords = readme_kws[repo_name]
            keywords = list(set(
                re.sub(r'\s*-\s*', '-', kw.replace("_", " ")) for kw in keywords if
                "=" not in kw and "<" not in kw and ">" not in kw and not
                kw[0].isdigit()))

            for keyword in keywords:
                cur.execute(
                    "INSERT INTO repo_keywords (time, repo_name, keyword) VALUES (%s, %s, %s)",
                    (date_obj, repo_name, keyword)
                )

            conn.commit()

    print("Inserting stars into database...")
    for repo_name in repo_names:
        print(repo_name)
        num_stars = repos[repo_names.index(repo_name)][1]

        cur.execute("""
                    INSERT INTO repo_stars_cumul (time, repo_name, cumul_stars)
                    VALUES (%s, %s, %s);
                """, (date_obj, repo_name, num_stars))
        conn.commit()

    print("Inserting co-occurrences into database...")
    for repo_name, cooccurrences in cooccurrences_by_repo.items():
        for (kw1, kw2), count in cooccurrences.items():
            cur.execute("""
                        INSERT INTO keyword_cooccurrence (time, keyword1, keyword2, count, repo_name)
                        VALUES (%s, %s, %s, %s, %s);
                    """, (date_obj, kw1, kw2, count, repo_name))
            conn.commit()

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
