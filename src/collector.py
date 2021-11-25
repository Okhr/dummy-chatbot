import json
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict

from praw.models import MoreComments
from tqdm import tqdm
import praw


class Collector(ABC):
    comments: List[Dict]
    url: str
    id: str

    def __init__(self, url):
        if self.is_valid_url(url):
            self.url = url
            self.id = self.get_id()
            self.comments = []
        else:
            raise AttributeError('Invalid URL format')

    @staticmethod
    @abstractmethod
    def is_valid_url(url):
        pass

    @abstractmethod
    def get_id(self):
        pass

    @abstractmethod
    def get_comments(self):
        pass

    @abstractmethod
    def save(self, dir_path: str):
        pass


class RedditCollector(Collector):
    @staticmethod
    def is_valid_url(url):
        if re.search("^(https?://)?(www.)?reddit\.com/r/([a-zA-Z0-9_-]+)/comments/([a-zA-Z0-9_-]+)", url):
            return True
        else:
            return False

    def get_id(self):
        return self.url.split('comments/')[1].split('/')[0]

    def get_comments(self):
        reddit = praw.Reddit(user_agent="Comment grabber",
                             client_id=os.getenv('REDDIT_CLIENT_ID'),
                             client_secret=os.getenv('REDDIT_CLIENT_SECRET'))

        submission = reddit.submission(id=self.id)

        comment_queue = submission.comments[:]

        total_processed = 0
        progress_bar = tqdm(total=len(comment_queue))

        while comment_queue:
            obj = comment_queue.pop(0)
            total_processed += 1
            progress_bar.update(1)

            if isinstance(obj, MoreComments):
                comment_queue.extend(obj.comments())
            else:
                comment_queue.extend(obj.replies)
                if obj.author is not None:
                    comment_id = obj.id
                    if obj.parent_id.split('_')[0] == 't1':
                        comment_parent_id = obj.parent_id.split('_')[1]
                    else:
                        comment_parent_id = ''
                    comment_content = obj.body
                    comment_author = obj.author.name
                    comment_like_count = obj.score
                    comment_timestamp = obj.created_utc

                    self.comments.append({
                        "id": comment_id,
                        "content": comment_content,
                        "author": comment_author,
                        "likes": comment_like_count,
                        "time": comment_timestamp,
                        "parent_id": comment_parent_id
                    })
            progress_bar.total = len(comment_queue) + total_processed

    def save(self, dir_path: str):
        if not os.path.exists(os.path.join(dir_path, 'reddit')):
            os.makedirs(os.path.join(dir_path, 'reddit'))
        with open(os.path.join(dir_path, 'reddit', self.id + '.json'), 'w') as fd:
            json.dump(self.comments, fd, indent=4)
