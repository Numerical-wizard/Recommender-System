import numpy as np
import nmslib
from tqdm import notebook
from collections import ChainMap
import pandas as pd
import os
import torch
import sqlite3


def get_neighbors(embeddings, articles, k, space, meth = 'hnsw'):

    if space[:-1] != "lp":
      index = nmslib.init(method=meth, space=space)
    else:
      index = nmslib.init(method=meth, space='lp', space_params={'p': int(space[-1])})
    articles = np.array(articles).astype('int32')
    index.addDataPointBatch(embeddings, ids = articles)
    if meth == 'hnsw':
        index.createIndex({'efConstruction': 2000}, print_progress=True)
    neighbors = [closests[0] for closests in index.knnQueryBatch(embeddings, k=k+1, num_threads=4)]
    if meth == 'hnsw':
        for i, closests in enumerate(neighbors):
            if len(closests)<k+1:
                neighbors[i] = np.concatenate((closests, np.array([-1 for i in range(k+1-len(closests))])))
    return np.array(neighbors)


def get_recommendations(articles, db_file, recommendation_table = 'related_item'):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(f'''SELECT * FROM
                    (
                        SELECT article
                        FROM
                        (
                            SELECT recommended_items.article, COUNT(recommended_item_article) recommended_count FROM
                            (
                                SELECT item.article, recommended_item_article, {recommendation_table}.weight
                                FROM {recommendation_table}
                                JOIN item
                                ON item.item_id = {recommendation_table}.item_id
                                WHERE article in {tuple(articles)} AND recommended_item_article in {tuple(articles)}
                            ) recommended_items
                            JOIN
                            (
                                SELECT item.article, MAX({recommendation_table}.weight)-4 min_weight
                                FROM {recommendation_table}
                                JOIN item
                                ON item.item_id = {recommendation_table}.item_id
                                WHERE article in {tuple(articles)}
                                GROUP BY item.article
                            ) min_weights
                            ON recommended_items.article = min_weights.article
                            WHERE weight >= min_weight
                            GROUP BY recommended_items.article
                        )
                        WHERE recommended_count = 5
                    )
                    ''')
    testing_articles = [article[0] for article in cur.fetchall()]
    cur.execute(f'''SELECT recommended_items.article, recommended_item_article FROM
                    (
                        SELECT item.article, recommended_item_article, {recommendation_table}.weight
                        FROM {recommendation_table}
                        JOIN item
                        ON item.item_id = {recommendation_table}.item_id
                        WHERE article in {tuple(testing_articles)}
                    ) recommended_items
                    JOIN
                    (
                        SELECT item.article, MAX({recommendation_table}.weight)-4 min_weight
                        FROM {recommendation_table}
                        JOIN item
                        ON item.item_id = {recommendation_table}.item_id
                        WHERE article in {tuple(testing_articles)}
                        GROUP BY item.article
                    ) min_weights
                    ON recommended_items.article = min_weights.article
                    WHERE weight >= min_weight
                    ''')
    recommendations = []
    for i, row in enumerate(cur.fetchall()):
        if i%5 == 0:
            recommendations.append([row[0]])
        recommendations[-1].append(row[1])
    return recommendations


def mrr(neighbors, db_file, recommendation_table = 'related_item'):
    neighbors = np.array(neighbors)

    recommendations = get_recommendations(neighbors[:, 0], db_file, recommendation_table)

    mrr = 0
    for recommended in recommendations:
        ranks, = np.where(np.in1d(neighbors[neighbors[:, 0] == recommended[0]][0, 1:], recommended) == True)
        mrr += np.sum(1/(ranks + 1))/5

    return mrr/len(recommendations)


def MRR(weight_img):
  weight_attr = 1 - weight_img
  embeddings = (np.vstack(embeddings_attributes)*weight_attr + np.vstack(embeddings_image) * weight_img * norm_coef) / 2
  neighbors = get_neighbors(embeddings, articles, 200, 'cosinesimil')
  return {weight_img: mrr(neighbors, f'{drive_path}/items.db')}

drive_path = "/content/drive/MyDrive/"
folder = f'/content/Node2Vec на атрибутах и изображениях'
articles = os.listdir(folder)
articles = [int(article[:-3]) for article in articles]
embeddings = []
for article in notebook.tqdm(articles):
    with torch.no_grad():
        embeddings.append(torch.load(f'{folder}/{article}.pt', map_location='cpu'))
folder = f'/content/images_embeddings_128'
articles = os.listdir(folder)
articles = [int(article[:-3]) for article in articles]

embeddings_image = []
for article in notebook.tqdm(articles):
    with torch.no_grad():
        embeddings_image.append(torch.load(f'{folder}/{article}.pt', map_location='cpu'))


embeddings_attributes = []
for article in notebook.tqdm(articles):
    with torch.no_grad():
        embeddings_attributes.append(torch.load(f'/content/images_attributes/{article}.pt', map_location='cpu'))

mean_len_attributes = np.sum(np.square(embeddings_attributes))
mean_len_img = np.sum(np.square(embeddings_image))
mean_len_attributes = (mean_len_attributes / len(articles)) ** 0.5
mean_len_img = (mean_len_img / len(articles)) ** 0.5
norm_coef = mean_len_img / mean_len_attributes
folder = f'/content/Сконкатенированные Node2Vec атрибуты и Node2Vec изображения'
articles = os.listdir(folder)
articles = [int(article[:-3]) for article in articles]
embeddings = []
for article in notebook.tqdm(articles):
    with torch.no_grad():
        embeddings.append(torch.load(f'{folder}/{article}.pt', map_location='cpu'))
folder = f'/content/images_embeddings_128'
articles = os.listdir(folder)
articles = [int(article[:-3]) for article in articles]
embeddings_image = []
for article in notebook.tqdm(articles):
    with torch.no_grad():
        embeddings_image.append(torch.load(f'{folder}/{article}.pt', map_location='cpu'))


spaces = ['l2', 'angulardist', 'negdotprod', 'lp3', 'lp1']
mrrs = []
for space in notebook.tqdm(spaces):
  neighbors = get_neighbors(embeddings, articles, 200, space)
  mrrs.append(round(mrr(neighbors, f'{drive_path}/items.db'), 4))


results = []
for weight_img in notebook.tqdm(np.arange(0.1, 0.9, 0.2)):
  weight_attr = 1-weight_img
  embeddings = np.concatenate((embeddings_attributes*weight_attr, embeddings_image * weight_img * norm_coef), axis = 1)
  neighbors = get_neighbors(embeddings, articles, 200, 'cosinesimil')
  mrr_weighted = mrr(neighbors, f'{drive_path}/items.db')
  results.append({weight_img:  round(mrr_weighted, 4)})


neighbors_cat = get_neighbors(embeddings, articles, 150, 'cosinesimil')
print(mrr(neighbors, f'{drive_path}/items.db'))

for i, article in notebook.tqdm(enumerate(articles)):
    torch.save(torch.tensor(neighbors[:, :6][i]), f'neighbours_cat/{article}.pt')
