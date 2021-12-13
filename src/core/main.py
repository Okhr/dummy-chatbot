from src.core.stats import *

from src.core.collecting import download_top_threads
from dotenv import load_dotenv
import yaml

if __name__ == '__main__':
    load_dotenv()

    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    for time_filter in params['Collector']['time_filters']:
        download_top_threads(params['Collector']['subreddit'], time_filter)

    print(get_number_of_comments(params['Collector']['subreddit']))
    # graph_comment_length_distribution(params['Collector']['subreddit'])
