import csv

def extract_user_ids(file_path):
    user_data = []

    """with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for item in row:
                parts = item.split('->')
                if len(parts) == 2:
                    tweet_info = parts[1].strip("[]").split(',')
                    user_ID = tweet_info[0].strip().strip("'")
                    tweet_ID = tweet_info[1].strip().strip("'")
                    user_data.append({'tweet_ID': tweet_ID, 'user_ID': user_ID})"""
    
   
    with open(file_path, 'r', encoding='utf-8') as file_tweet_tree:
        for line in file_tweet_tree:
            parent, child = line.strip().split('->')
            parent = parent.strip().strip('[]').split(',')
            child = child.strip().strip('[]').split(',')

            parent_user_id = parent[0].strip().strip("'")   #dds: non usato dopo
            parent_tweet_id = parent[1].strip().strip("'")
            
            child_user_id = child[0].strip().strip("'")
            child_tweet_id = child[1].strip().strip("'")  #dds: non usato dopo
            user_data.append({'tweet_ID': parent_tweet_id, 'user_ID': parent_user_id, 'retweet_user_ID': child_user_id})

    return user_data

if __name__ == "__main__":
    file_path = 'rumor_detection_2017/twitter16/tree/767803368081358848.txt'
    user_data = extract_user_ids(file_path)
    print(user_data)