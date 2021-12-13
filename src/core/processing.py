import json
import os
import re
from pprint import pp

import redditcleaner
import yaml
from tqdm import tqdm


def preprocess_comments(subreddit: str, post_id: str):
    with open(f'data/raw/{subreddit}/{post_id}.json', 'r', encoding='utf-8') as f:
        post_data = json.load(f)

    preprocessed_comments = {}

    for comment in post_data:
        comment_id = comment['id']
        parent_id = comment['parent_id']
        content = comment['content']

        # here are all the processing we apply to comments' content
        # clean reddit formatting characters
        content = redditcleaner.clean(content, link=False)

        # removing links but not link texts
        pattern = re.compile('\[(.*?)\]\(.*?\)')
        content = pattern.sub(r'\1', content)

        # removing ELI5 tags (in the rules of eli5 sub)
        content = re.sub('^[eE][lL][iI]5:* *', '', content)

        # removing \n
        content = re.sub('\r?\n|\r', ' ', content)

        preprocessed_comments[comment_id] = {'content': content, 'parent_id': parent_id}

    return preprocessed_comments


def make_pairs(subreddit: str, top_comments: bool = False):
    pairs = []
    wrong_ids = 0

    for file_name in tqdm(os.listdir(f'data/raw/{subreddit}')):
        post_id = file_name.split('.')[0]
        preprocessed_comments = preprocess_comments(subreddit, post_id)

        # using only the top level comments as second element of the pair, and main post as first element
        if top_comments:
            for comment_id, comment_data in preprocessed_comments.items():
                if comment_data['parent_id'] == post_id:
                    if comment_data['parent_id'] in preprocessed_comments.keys():
                        pairs.append([preprocessed_comments[post_id]['content'], comment_data['content']])
                    else:
                        wrong_ids += 1

        # using every comment
        else:
            for comment_id, comment_data in preprocessed_comments.items():
                if comment_data['parent_id'] != '':
                    if comment_data['parent_id'] in preprocessed_comments.keys():
                        pairs.append(
                            [preprocessed_comments[comment_data['parent_id']]['content'], comment_data['content']]
                        )
                    else:
                        wrong_ids += 1

    print(f'Created {len(pairs)} pairs')
    print(f'{wrong_ids} wrong ids')

    if not os.path.exists(f'data/pairs/{subreddit}'):
        os.makedirs(f'data/pairs/{subreddit}')

    if top_comments:
        with open(f'data/pairs/{subreddit}/top_comments.json', 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=4)
    else:
        with open(f'data/pairs/{subreddit}/all_comments.json', 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=4)


if __name__ == '__main__':
    with open('config.yaml') as fd:
        params = yaml.load(fd.read(), Loader=yaml.CLoader)

    make_pairs(params['Pairs']['subreddit'], top_comments=params['Pairs']['top_comments'])
