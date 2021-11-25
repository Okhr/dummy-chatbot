from src.collector import RedditCollector
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    rc = RedditCollector('https://www.reddit.com/r/explainlikeimfive/comments/r142mq/eli5_why_is_the_galaxy_flat/')
    rc.get_comments()
    rc.save('data/raw')
