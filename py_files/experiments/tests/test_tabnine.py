# %%
import json
import urllib.request
from cgi import test
from attr import define
import pandas as pd

# count numbers
import sys

# count numbers


def count(list): return len(list)

# load data from file


# %%

def sumtwonumbers(VALUE1, VALUE2):
    return VALUE1 + VALUE2


def ITERATEOVERDATAFRMAE(DATAFRAME):
    for index, row in DATAFRAME.iterrows():
        print(index, row['col1'], row['col2'])


# %%


def get_repositories(org):
    """List all names of GitHub repositories for an org."""
    url = 'https://api.github.com/orgs/{}/repos'.format(org)
    response = urllib.request.urlopen(url)
    return json.loads(response.read().decode())


def fetch_tweets_from_user(user_name):
    """Fetch tweets from a user."""
    url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name={}'.format(
        user_name)
    response = urllib.request.urlopen(url)
    return json.loads(response.read().decode())
