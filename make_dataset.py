import os
import time
import argparse
from pathlib import Path

import tweepy
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--year",
        type=int,
        required=True,
    )
    parser.add_argument("--months", type=int_list, required=True)

    args = parser.parse_args()

    if print_args:
        _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def get_bearer_token():
    load_dotenv()
    bearer_token = os.environ["BEARER_TOKEN"]
    return bearer_token


def format_month_or_day(m):
    if 0 < m and m < 10:
        m_str = f"0{m}"
    else:
        m_str = f"{m}"
    return m_str


def find_start_end_y_m(year, month, day=None, max_day=None):
    start_m = format_month_or_day(month)
    start_y = year
    if day:
        start_d = format_month_or_day(day)
        if day == max_day:
            end_d = format_month_or_day(1)
            if month == 12:
                end_m = format_month_or_day(1)
                end_y = year + 1
            else:
                end_m = format_month_or_day(month + 1)
                end_y = year
        else:
            end_d = format_month_or_day(day + 1)
            end_m = start_m
            end_y = year
        return start_y, start_m, start_d, end_y, end_m, end_d
    else:
        if month == 12:
            end_m = format_month_or_day(1)
            end_y = year + 1
        else:
            end_m = format_month_or_day(month + 1)
            end_y = year
        return start_y, start_m, end_y, end_m


def get_month_days(month):
    m_d = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    return m_d[month]

def save_response_to_csv(args, month, responses, day=None):
    tweets = []
    user_dict = {}
    start_time = time.time()
    for response in tqdm(responses):
        for user in response.includes["users"]:
            user_dict[user.id] = {"username": user.username, "location": user.location}
        for tweet in response.data:
            author_info = user_dict[tweet.author_id]
            tweets.append(
                {
                    "author_id": tweet.author_id,
                    "username": author_info["username"],
                    "author_location": author_info["location"],
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "retweets": tweet.public_metrics["retweet_count"],
                    "replies": tweet.public_metrics["reply_count"],
                    "likes": tweet.public_metrics["like_count"],
                    "quote_count": tweet.public_metrics["quote_count"],
                }
            )

    df = pd.DataFrame(tweets)
    if day:
        print(
            f"Successfully retrieved tweets in year {args.year} month {month} day {day}! -- {df.shape[0]} entries"
        )
        path_str = f"{args.data_dir}/{args.year}/{month}/{day}.csv"
    else:
        print(
            f"Successfully retrieved tweets in year {args.year} month {month}! -- {df.shape[0]} entries"
        )
        path_str = f"{args.data_dir}/{args.year}/{month}.csv"
    print(f"Saving to {path_str}")
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    end_time = time.time()
    saving_time = end_time - start_time
    print(f"Saving time: {saving_time:0.2f} s")


def save_tweet(args, month, client, day=None):
    start_time = time.time()
    
    if day:
        start_y, start_m, start_d, end_y, end_m, end_d = find_start_end_y_m(args.year, month, day, max_day=get_month_days(month))
        start_time_str = f"{start_y}-{start_m}-{start_d}T00:00:00Z"
        end_time_str = f"{end_y}-{end_m}-{end_d}T00:00:00Z"
    else:
        start_y, start_m, end_y, end_m = find_start_end_y_m(args.year, month)
        start_time_str = f"{start_y}-{start_m}-01T00:00:00Z"
        end_time_str = f"{end_y}-{end_m}-01T00:00:00Z"

    print("\n" + "#" * 60 + "\n")
    print(
        f"Start retrieving tweet from {start_time_str} to {end_time_str}!"
    )

    responses = []
    for response in tweepy.Paginator(
        client.search_all_tweets,
        query="-is:quote -is:reply -is:retweet lang:en place_country:CA",
        user_fields=["username", "location"],
        tweet_fields=["created_at", "public_metrics", "text"],
        expansions="author_id",
        start_time=start_time_str,
        end_time=end_time_str,
        max_results=500,
    ):
        time.sleep(1)
        responses.append(response)
    end_time = time.time()
    retrieve_time = end_time - start_time
    print(f"Retrieving time: {retrieve_time:0.2f} s")

    save_response_to_csv(args, month, responses, day)


if __name__ == "__main__":
    args = get_args()

    bearer_token = get_bearer_token()

    client = tweepy.Client(bearer_token, wait_on_rate_limit=True)

    for m in args.months:
        for d in range(1, get_month_days(m)+1):
            save_tweet(args, m, client, d)
