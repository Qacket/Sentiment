import pandas as pd

def deal_data():
    data = pd.read_csv('./datasets/mturk_answers.csv')
    data[['WorkerId', 'Input.id']] = data[['Input.id', 'WorkerId']]
    data.columns = ['task_id', 'annotator_id', 'task', 'dealt_task', 'ground_truth', 'answer']
    data = data.drop(columns=['dealt_task'])
    task2idx = {}
    task_idx = 0
    annotator2idx = {}
    annotator_idx = 0
    label2idx = {}
    label_idx = 0
    for i in range(len(data)):
        if data.loc[i, 'task_id'] not in task2idx:
            task2idx[data.loc[i, 'task_id']] = task_idx
            task_idx += 1
        if data.loc[i, 'annotator_id'] not in annotator2idx:
            annotator2idx[data.loc[i, 'annotator_id']] = annotator_idx
            annotator_idx += 1
        if data.loc[i, 'ground_truth'] not in label2idx:
            label2idx[data.loc[i, 'ground_truth']] = label_idx
            label_idx += 1
        if data.loc[i, 'answer'] not in label2idx:
            label2idx[data.loc[i, 'answer']] = label_idx
            label_idx += 1
        data.loc[i, 'task_id'] = task2idx[data.loc[i, 'task_id']]
        data.loc[i, 'annotator_id'] = annotator2idx[data.loc[i, 'annotator_id']]
        data.loc[i, 'ground_truth'] = label2idx[data.loc[i, 'ground_truth']]
        data.loc[i, 'answer'] = label2idx[data.loc[i, 'answer']]
    data.to_csv('./datasets/total_sentiment.csv', index=False, header=True)

    crowd = data.drop(columns=['task', 'ground_truth'], inplace=False)
    crowd.to_csv('./datasets/sentiment_crowd.txt', sep='\t', index=False, header=False)

    ground_truth = data.drop_duplicates(subset=['task_id'], keep='first', inplace=False). \
        drop(columns=['annotator_id', 'task', 'answer'], inplace=False)

    ground_truth.to_csv('./datasets/sentiment_truth.txt', sep='\t', index=False, header=False)


deal_data()