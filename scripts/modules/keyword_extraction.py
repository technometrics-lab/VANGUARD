# https://github.com/imaximumGit/cyd_pdm/blob/main/topic_extraction/src/extractors.py

import os
import pickle
import re

import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer


class Extractor():
    """Abstract class for extractors"""

    def __init__(self, name, model, nb_tokens, splitlines):
        """Init the extractor

        Parameters:
            name (str): name of the extractor
            model (object): model used to extract keywords
            pickle_path (str): path where to save the result
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
                               currently only used by HuggingFaceExtractor
        """
        self.name = name
        self.model = model
        self.nb_tokens = nb_tokens
        self.splitlines = splitlines

    def get_file_name(self, category=None, pdf_name=""):
        """Return file name to save the result

        Parameters:
            category (str) default=None: category of the pdf
            pdf_name (str) default='': name of the pdf

        Return:
            file_name (str): file name where to save the result
        """
        if category is None:
            category = "none/"
        else:
            category = f"{category}/"
        if pdf_name != "":
            pdf_name = pdf_name + "/"
        split_str = "_split" if self.splitlines else ""
        return f"pickle_model_keyword/{category}{pdf_name}{self.name}_{self.nb_tokens}{split_str}.pkl"

    def extract_keywords(self, doc: str) -> list[str]:
        """Extract keywords from a document

        Parameters:
            doc (str): a text document

        Return:
            keywords (list[str]): list of keywords"""
        raise NotImplementedError

    def save_result(
            self, keywords, root_path, category=None, pdf_name="", remove_score=False
    ):
        """Save the extracted keywords

        Parameters:
            keywords (list[(str, float)]): list of keywords with their score
            root_path (str): path where to save the result
            category (str) default=None: category of the pdf
            pdf_name (str) default='': name of the pdf
            remove_score (bool) default=True: if True, remove the score from the keywords
        """
        if remove_score:
            keywords = [word[0] for word in keywords]

        file_name = f"{root_path}/{self.get_file_name(category, pdf_name)}"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(keywords, f)

    def check_parenthesis(self, txt: str) -> bool:
        """Check if parenthesis are balanced

        Parameters:
            txt (str): text to check

        Returns:
            bool: True if parenthesis are balanced, False otherwise
        """

        open_list = ["[", "{", "("]
        close_list = ["]", "}", ")"]
        stack = []
        for i in txt:
            if i in open_list:
                stack.append(i)
            elif i in close_list:
                pos = close_list.index(i)
                if stack and (open_list[pos] == stack[-1]):
                    stack.pop()
                else:
                    return False
        if len(stack) == 0:
            return True
        return False

    def clear_filter_keywords(self, keywords: list[str]):
        """Clear and filter keywords

        Parameters:
            keywords (list[str]): list of keywords

        Returns:
            list[str]: list of filtered keywords
        """
        # remove non ascii tokens and with parenthesis not balanced
        keywords = [
            (keyword[0].strip(), keyword[1].strip())
            for keyword in keywords
            if keyword[0].isascii() and self.check_parenthesis(keyword[0])
        ]

        for i in range(len(keywords)):
            keyword = keywords[i][0]
            keyword = keyword.replace("\n", " ")  # remove new lines
            keyword = re.sub(" \d* ", " ", keyword)  # remove numbers
            keyword = re.sub(
                "(\s*-\s+|\s+-\s*)", "", keyword
            )  # remove hyphens surronunded by spaces

            # if all letters of a word are upper case lower it
            split_keyword = keyword.split(" ")
            for j in range(len(split_keyword)):
                if split_keyword[j].isupper():
                    split_keyword[j] = split_keyword[j].lower()
                if len(split_keyword[j]) <= 2:
                    split_keyword[j] = ""
            keyword = " ".join([word for word in split_keyword if word != ""])

            # split word by capital letters if there are no spaces
            # keyword = " ".join(
            #     [
            #         catch_group[0]
            #         for catch_group in re.findall(
            #         "(([\da-z]+|[A-Z\.]+)[^A-Z\s()\[\]{}]*)", keyword
            #     )
            #     ]
            # )
            keyword = keyword.lower()  # put to lower case
            keyword = re.sub("\s{2,}", " ", keyword)  # remove double spaces
            keyword = re.sub("[,\.']", "", keyword)  # remove punctuation

            if len(keyword) > 0 and keyword[-1] == "-":
                keyword = keyword[:-1]
            keywords[i] = (keyword.strip(), keywords[i][1])  # remove leading and trailing spaces

        # remove keywords with no letters
        keywords = [
            (keyword[0], keyword[1])
            for keyword in keywords
            if any(char.isalpha() for char in keyword[0])
        ]

        return [keyword for keyword in keywords if len(keyword[0]) > 2]

    def __str__(self):
        return self.name




class SpacyExtractor(Extractor):
    """Spacy extractor extends Extractor class"""

    MODEL_NAMES = [
        "en_core_web_lg",  # cpu
        "en_core_web_trf",  # gpu
    ]

    RULES = {
        "Noun and specific dep": [
            {
                "DEP": {"IN": ["compound", "amod", "ccomp", "conj", "nmod"]},
                "OP": "{1,}",
            },
            {"POS": {"IN": ["NOUN", "X", "PROPN"]}},
        ],
        "Noun and adjective": [{"POS": {"IN": ["NOUN", "X", "PROPN"]}}, {"POS": "ADJ"}],
        "Noun": [
            {
                "POS": {"IN": ["NOUN", "X", "PROPN"]},
                "LIKE_EMAIL": False,
                "LIKE_URL": False,
            }
        ],
        # "Noun and verb": [{"POS": {"IN": ["NOUN", "X", "PROPN", "ADJ"]}}, {"POS": "VERB"}],
        # "Noun and amod": [{"DEP": "vocative", "OP": "{1,}"}, {"POS": {"IN": ["NOUN", "X", "PROPN"]}}],
    }

    CHUNK_SIZE = 990_000

    def __init__(self, model_name, nb_tokens, splitlines):
        """Init the extractor

        Parameters:
            model_name (str): name of the model to use. See MODEL_NAMES for available models
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords # pas besoin
        """

        # set_gpu_allocator("pytorch")
        # require_gpu(0)

        # spacy.prefer_gpu()
        def custom_tokenizer(nlp):
            """
            Custom tokenizer to avoid splitting on hyphens
            """
            infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~/]''')
            prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
            suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

            return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                             suffix_search=suffix_re.search,
                             infix_finditer=infix_re.finditer)

        nlp = spacy.load(model_name)
        nlp.tokenizer = custom_tokenizer(nlp)

        # remove elements that use memory after the pipeline as run
        nlp.add_pipe("doc_cleaner")
        # nlp.add_pipe("merge_entities")

        self.rule_matcher = Matcher(nlp.vocab)
        for rule_name, rule_tags in self.RULES.items():  # register rules in matcher
            self.rule_matcher.add(rule_name, [rule_tags])

        super().__init__(model_name, nlp, nb_tokens, splitlines)

    def _remove_sub_interval(self, matches: list[tuple[int, int, int]]):
        """Remove sub intervals from a list of intervals

        Parameters:
            matches (list[tuple[int, int, int]]): list of intervals (match_id, start, end)

        Returns:
            list[tuple[int, int, int]]: list of intervals without sub intervals
        """
        keep_matches = []

        if len(matches) > 0:
            _, longest_start_id, longest_end_id = matches[0]
            id_to_add = 0
            for i, match in enumerate(matches[1:], 1):
                _, start_id, end_id = match
                if (longest_start_id <= end_id and longest_end_id >= end_id) or (
                        start_id <= longest_end_id and end_id >= longest_start_id
                ):
                    if end_id - start_id > longest_end_id - longest_start_id:
                        longest_start_id, longest_end_id = start_id, end_id
                        id_to_add = i
                else:
                    if end_id > longest_end_id:
                        keep_matches.append(matches[id_to_add])
                        longest_start_id, longest_end_id = start_id, end_id
                        id_to_add = i
        return keep_matches

    def extract_keywords(self, doc):
        results = []

        # split doc into chunks
        chunks_file = []
        curr_return_line = 0
        next_return_line = doc.find("\n", curr_return_line + self.CHUNK_SIZE) + 1
        while next_return_line > curr_return_line:
            chunks_file.append(doc[curr_return_line:next_return_line])
            curr_return_line = next_return_line
            next_return_line = doc.find("\n", curr_return_line + self.CHUNK_SIZE) + 1
        chunks_file.append(doc[curr_return_line:])
        cleaned_tokens = []
        for chunk in chunks_file:
            chunk = re.sub(r'^\[!.*\n?', '', chunk, flags=re.MULTILINE)  # remove images
            chunk = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                           chunk, flags=re.MULTILINE)  # remove links
            # Remove content between single backticks
            chunk = re.sub(r'`([^`]+)`', '', chunk)
            # Remove content between triple backticks
            chunk = re.sub(r'```([^`]+)```', '', chunk)
            # Remove content inside <>
            chunk = re.sub(r'<[^>]+>', '', chunk)
            # replace punctuation by spaces (except for hyphens)
            chunk = re.sub(r'[^\w\s-]', ' ', chunk)

            tokens = self.model(chunk)
            cleaned_tokens.append(re.sub(r'\W+', ' ', tokens.text))  # only keep words
            matches = self.rule_matcher(tokens)
            keep_matches = self._remove_sub_interval(matches)
            results.extend((tokens[start:end].lemma_, tokens[start:end].text) for _, start, end in
                           keep_matches)

        cleaned_tokens = " ".join(cleaned_tokens)
        current_index = 0
        for t in [e[1] for e in results]:
            index = cleaned_tokens.find(t, current_index)
            if index != -1:  # If t[1] is found
                cleaned_tokens = cleaned_tokens[:index] + " " + t + " " + cleaned_tokens[index + len(t):]
                current_index = index + len(t) + 2  # Move the pointer to after the current found t[1]
        cleaned_tokens = re.sub(' +', ' ', cleaned_tokens)  # Remove duplicate spaces

        # search for the most common tokens
        if len(results) > 0:
            keywords = self.clear_filter_keywords(results)
            keywords = [(e[0], re.sub(r'\W+', ' ', e[1]).strip()) for e in keywords]
            # return set(keywords)
            unique_keywords = []
            for keyword in keywords:
                if keyword not in unique_keywords:
                    unique_keywords.append(keyword)
            return unique_keywords, cleaned_tokens
        return [], ""
