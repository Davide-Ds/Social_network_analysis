import csv

def extract_user_ids(file_path):
    user_data = {}
   
    with open(file_path, 'r', encoding='utf-8') as file_tweet_tree:
        next(file_tweet_tree)  # Skip the first line ['ROOT', 'ROOT', '0.0']
        for line in file_tweet_tree:
            # Split the line into parent and child parts
            parent, child = line.strip().split('->')
            # Process parent part
            parent = parent.strip().strip('[]').split(',')
            # Process child part
            child = child.strip().strip('[]').split(',')

            # Extract user IDs and tweet IDs for parent and child
            parent_user_id = parent[0].strip().strip("'")
            parent_tweet_id = parent[1].strip().strip("'")
            
            child_user_id = child[0].strip().strip("'")
            child_tweet_id = child[1].strip().strip("'")
            
            # Update user_data dictionary with parent tweet information
            if parent_tweet_id not in user_data:
                user_data[parent_tweet_id] = {'user_ID': parent_user_id, 'retweeting_user_ID': [child_user_id]}
            else:
                user_data[parent_tweet_id]['retweeting_user_ID'].append(child_user_id)
            
            # Update user_data dictionary with child tweet information
            if child_tweet_id not in user_data:
                user_data[child_tweet_id] = {'user_ID': child_user_id, 'father_tweet_ID': parent_tweet_id, 'retweeting_user_ID': []}
            else:
                user_data[child_tweet_id]['father_tweet_ID'] = parent_tweet_id

    return user_data

if __name__ == "__main__":
    file_path = 'rumor_detection_2017/twitter16/tree/498430783699554305.txt'
    user_data = extract_user_ids(file_path)
    print(user_data)