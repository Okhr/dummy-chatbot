import json
import os
import matplotlib.pyplot as plt


def get_number_of_comments(subreddit):
    total = 0
    for file_name in os.listdir(f'data/raw/{subreddit}'):
        with open(f'data/raw/{subreddit}/{file_name}') as f:
            thread_data = json.load(f)
        total += len(thread_data)
    return total


def get_comment_length_distribution(subreddit):
    lengths = []
    for file_name in os.listdir(f'data/raw/{subreddit}'):
        with open(f'data/raw/{subreddit}/{file_name}') as f:
            thread_data = json.load(f)
        lengths.extend([len(comment['content']) for comment in thread_data])
    return lengths


def graph_comment_length_distribution(subreddit):
    lengths = get_comment_length_distribution(subreddit)
    plt.hist(lengths, bins=100)
    plt.show()
