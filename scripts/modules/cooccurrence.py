from collections import defaultdict

def printf():
    print("hello")
def compute_cooccurrences(text, phrases, max_distance, cooccurrence):
    """
    Compute co-occurrences of phrases (keywords) in a given text.
    :param text: text in which to compute co-occurrences
    :param phrases: phrases to compute co-occurrences of
    :param max_distance: maximum distance between co-occurring phrases (1=adjacent, 2=one word in between, etc.)
    :param cooccurrence: dictionary of co-occurrences to update
    :return: updated co-occurrence dictionary
    """
    words = text.split()

    for i, word in enumerate(words):
        for phrase_clean, dirty_phrases in phrases:
            matched = False
            for phrase_dirty in dirty_phrases:
                if ' '.join(words[i:i + len(phrase_dirty.split())]) == phrase_dirty:
                    for j in range(i + len(phrase_dirty.split()),
                                   min(i + max_distance + len(phrase_dirty.split()), len(words))):
                        for other_phrase_clean, other_dirty_phrases in phrases:
                            if other_phrase_clean != phrase_clean:  # same word can't co-occur
                                for other_phrase_dirty in other_dirty_phrases:
                                    if ' '.join(words[j:j + len(other_phrase_dirty.split())]) == other_phrase_dirty:
                                        if not matched:
                                            other_phrase_clean_save = other_phrase_clean
                                        matched = True
            if matched:
                cooccurrence[
                    (min(phrase_clean, other_phrase_clean_save), max(phrase_clean, other_phrase_clean_save))] += 1

    return cooccurrence


def compute_cooccurrences_123(text, phrases, cooccurrence):
    """
    Compute co-occurrences of phrases (keywords) in a given text, where adjacent phrases get 3 points, phrases with one
     word in between get 2 points, phrases with two words in between get 1 point, and phrases with more words in between
     get 0 points.
    :param text: text in which to compute co-occurrences
    :param phrases: phrases to compute co-occurrences of
    :param cooccurrence: dictionary of co-occurrences to update
    :return: co-occurrence dictionary
    """

    cooccurrence_temp = defaultdict(int)
    for i in range(1, 4):
        cooccurrence_temp = compute_cooccurrences(text, phrases, i, cooccurrence_temp)

    for key, value in cooccurrence_temp.items():
        cooccurrence[key] += min(value, 5)  # co-occ score contributed by a single repo is capped at 5

    return cooccurrence
