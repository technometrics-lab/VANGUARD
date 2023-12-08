def get_owner_and_repo(github_url):
    """
    Get the owner and repository name from a github url.
    :param github_url: url of the github repository
    :return: owner, repo
    """
    split_url = github_url.split("/")
    owner = split_url[3]
    repo = split_url[4]
    return owner, repo


def flatten(l):
    """
    Flatten a list of lists
    :param l: list of lists
    :return: flattened list
    """
    return [item for sublist in l for item in sublist]
