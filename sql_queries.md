## Top 10 Repositories by Stars
```
SELECT 
    rs.repo_name, 
    rs.stars AS "Stars", 
    ru.repo_url
FROM 
    repo_stars rs
INNER JOIN 
    repo_urls ru 
ON 
    rs.repo_name = ru.repo_name
WHERE 
    rs.time = TO_DATE('$selected_date', 'DD.MM.YYYY')
ORDER BY 
    "Stars" DESC
LIMIT 10;
```

## Top 10 Fastest Growing Repositories
```
WITH PreviousStar AS (
    SELECT
        rs.repo_name,
        rs.time,
        rs.stars AS current_star,
        LAG(rs.stars, 1, 0) OVER (
            PARTITION BY rs.repo_name
            ORDER BY rs.time
        ) AS previous_star
    FROM repo_stars rs
),
GrowthRates AS (
    SELECT
        ps.repo_name,
        ps.time,
        ps.current_star,
        ps.previous_star,
        CASE
            WHEN ps.previous_star = 0 THEN 1
            ELSE ps.current_star::float / ps.previous_star
        END AS growth_rate
    FROM PreviousStar ps
    WHERE ps.time = TO_DATE('$selected_date', 'DD.MM.YYYY')
)
SELECT
    gr.repo_name,
    gr.time,
    gr.growth_rate AS "Growth Rate",
    ru.repo_url
FROM GrowthRates gr
INNER JOIN repo_urls ru ON gr.repo_name = ru.repo_name
ORDER BY "Growth Rate" DESC
LIMIT 10;
```

## Top 10 Keywords in READMEs
```
WITH TotalRepos AS (
    SELECT
        COUNT(DISTINCT repo_name) AS total_repos
    FROM
        repo_keywords
    WHERE
        time = TO_DATE('$selected_date', 'DD.MM.YYYY')
),
KeywordRepos AS (
    SELECT
        rk.keyword,
        ARRAY_AGG(rk.repo_name ORDER BY r.stars DESC) AS ordered_repos
    FROM
        repo_keywords rk
    JOIN
        repo_stars r ON rk.repo_name = r.repo_name AND rk.time = r.time
    WHERE
        rk.time = TO_DATE('$selected_date', 'DD.MM.YYYY')
        AND rk.keyword NOT IN (SELECT keyword FROM trivial_keywords)
    GROUP BY
        rk.keyword
)
SELECT
    kr.keyword AS "Keyword",
    (COUNT(DISTINCT rk.repo_name)::FLOAT / tr.total_repos) * 100
    AS "% of READMEs in which keyword appears",
    kr.ordered_repos AS "Repos in which keyword appears"
FROM
    repo_keywords rk
JOIN
    KeywordRepos kr ON rk.keyword = kr.keyword
CROSS JOIN
    TotalRepos tr
WHERE
    rk.time = TO_DATE('$selected_date', 'DD.MM.YYYY')
    AND rk.keyword NOT IN (SELECT keyword FROM trivial_keywords)
GROUP BY
    kr.keyword, tr.total_repos, kr.ordered_repos
ORDER BY
    "% of READMEs in which keyword appears" DESC
LIMIT 10;
```

## Top 10 Fastest Growing Keywords in READMEs
```
WITH
previous_date AS (
    SELECT MAX(time) AS prev_date
    FROM repo_keywords
    WHERE time < TO_DATE('$selected_date', 'DD.MM.YYYY')
),
keyword_counts_after AS (
    SELECT
        keyword,
        COUNT(*) AS count_after
    FROM repo_keywords
    WHERE time = TO_DATE('$selected_date', 'DD.MM.YYYY')
    GROUP BY keyword
),
keyword_counts_before AS (
    SELECT
        keyword,
        COUNT(*) AS count_before
    FROM repo_keywords
    WHERE time = (SELECT prev_date FROM previous_date)
    GROUP BY keyword
),
TotalReposAfter AS (
    SELECT
        COUNT(DISTINCT repo_name) AS total_repos_after
    FROM repo_keywords
    WHERE time = TO_DATE('$selected_date', 'DD.MM.YYYY')
),
TotalReposBefore AS (
    SELECT
        COUNT(DISTINCT repo_name) AS total_repos_before
    FROM repo_keywords
    WHERE time = (SELECT prev_date FROM previous_date)
)
SELECT
    COALESCE(b.keyword, a.keyword) AS keyword,
    ((COALESCE(a.count_after, 0)::FLOAT / tra.total_repos_after) -
    (COALESCE(b.count_before, 0)::FLOAT / trb.total_repos_before)) * 100
        AS "Change in % of READMEs in which keyword appears"
FROM
    keyword_counts_before b
FULL JOIN
    keyword_counts_after a ON a.keyword = b.keyword
CROSS JOIN
    TotalReposBefore trb
CROSS JOIN
    TotalReposAfter tra
WHERE
    COALESCE(b.keyword, a.keyword) NOT IN (SELECT keyword FROM trivial_keywords)
ORDER BY
    "Change in % of READMEs in which keyword appears" DESC
LIMIT 10;
```

## Shared Keywords in Selected Repos
```
WITH keyword_with_counts AS (
    SELECT
        keyword,
        COUNT(*) AS count,
        time
    FROM
        repo_keywords
    WHERE
        repo_name IN ($selected_repos)
        AND time = TO_DATE('$selected_date', 'DD.MM.YYYY')
    GROUP BY
        keyword,
        time
    HAVING
        COUNT(*) > 1
),

keyword_cooccurrence_sum AS (
    SELECT
        time,
        keyword1,
        keyword2,
        SUM(score) AS score
    FROM
        keyword_cooccurrence
    GROUP BY
        time,
        keyword1,
        keyword2
)

SELECT
    kwc.keyword AS "Keyword",
    kwc.count AS "Occurrences in selected repos",
    (ARRAY_AGG(co.keyword2 ORDER BY co.score DESC))[1:3]
        AS "Top 3 co-occurring keywords"
FROM
    keyword_with_counts kwc
    LEFT JOIN keyword_cooccurrence_sum co ON kwc.keyword = co.keyword1
        AND kwc.time = co.time
WHERE
    kwc.keyword NOT IN (SELECT keyword FROM trivial_keywords)
GROUP BY
    kwc.keyword,
    kwc.count
ORDER BY
    "Occurrences in selected repos" DESC,
    kwc.keyword
LIMIT 10;
```

## Repos where Keywords Co-occur
```
WITH SelectedKeywords AS (
    SELECT
        keyword,
        COUNT(*) AS keyword_count,
        time,
        ARRAY_AGG(DISTINCT repo_name) AS repos_with_keyword
    FROM
        repo_keywords
    WHERE
        repo_name IN ($selected_repos)
            AND time = TO_DATE('$selected_date', 'DD.MM.YYYY')
    GROUP BY
        keyword, time
    HAVING
        COUNT(*) > 1
),

CooccurrenceData AS (
    SELECT
        time,
        keyword1,
        keyword2,
        SUM(score) AS cooccurrence_score,
        ARRAY_AGG(repo_name ORDER BY score DESC) AS cooccurring_repos
    FROM
        keyword_cooccurrence
    GROUP BY
        time, keyword1, keyword2
),

TopCooccurringKeywords AS (
    SELECT
        sk.keyword AS primary_keyword,
        (ARRAY_AGG(cd.keyword2 ORDER BY cd.cooccurrence_score DESC))[1:3] 
            AS top_cooccurring_keywords
    FROM
        SelectedKeywords sk
        LEFT JOIN CooccurrenceData cd ON sk.keyword = cd.keyword1
            AND sk.time = cd.time
    WHERE
        sk.keyword NOT IN (SELECT keyword FROM trivial_keywords)
    GROUP BY
        sk.keyword
    LIMIT 10
),

ExpandedKeywords AS (
    SELECT
        primary_keyword,
        UNNEST(top_cooccurring_keywords) AS secondary_keyword
    FROM
        TopCooccurringKeywords
)

SELECT
    ek.primary_keyword AS "Keyword 1",
    ek.secondary_keyword AS "Keyword 2",
    cd.cooccurring_repos AS "Repos where keywords co-occur"
FROM
    ExpandedKeywords ek
    JOIN CooccurrenceData cd ON ek.primary_keyword = cd.keyword1
        AND ek.secondary_keyword = cd.keyword2
WHERE
    cd.time = TO_DATE('$selected_date', 'DD.MM.YYYY');
```