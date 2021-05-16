import argparse
import numpy as np
import pandas as pd
import json
from catboost import CatBoostRegressor


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t", parse_dates=['release_date'])

with open('features.json', 'r') as features:
    (genres_index, languages_index, companies_index, countries_index,  model_features) = json.load(features)


def has_uncommon_production(index):
    def has_uncommon_production_(production):
        names = set()
        for item in eval(production):
            names.add(item['name'])
        return len(names.difference(index)) > 0

    return has_uncommon_production_


def transform_column(column, names, uncommon_production_column_name=None):
    def transform_list(items):
        values = []
        existing = set()
        for item in eval(items):
            existing.add(item['name'])
        for item in names:
            values.append(item in existing)
        return pd.Series(values, index=names, dtype=bool)

    transformed = column.apply(transform_list)
    if uncommon_production_column_name:
        transformed[uncommon_production_column_name] = column.apply(has_uncommon_production(names))
    return transformed


def season(date):
    if date.month in [12, 1, 2]:
        return 'Winter'
    elif date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def count_spoken_languages(spoken_languages):
    if pd.isnull(spoken_languages):
        return 0
    languages = set()
    for language in eval(spoken_languages):
        languages.add(language['name'])
    return len(languages)


data['has_collection'] = ~data['belongs_to_collection'].isnull()
data = pd.concat([data, transform_column(data['genres'], genres_index)], axis=1)
data['has_homepage'] = ~data['homepage'].isnull()
for language in languages_index:
    data[language] = data['original_language'] == language
data['Uncommon Language'] = data['original_language'].apply(lambda l: l not in languages_index)
data = pd.concat([data, transform_column(data['production_companies'], companies_index, 'Uncommon Production Company')],
                 axis=1)
data = pd.concat([data, transform_column(data['production_countries'], countries_index, 'Uncommon Production Country')],
                 axis=1)
data['release_year'] = data['release_date'].dt.year
data['release_month'] = data['release_date'].dt.month
for s in ['Winter', 'Spring', 'Summer', 'Autumn']:
    data[s] = False
data[['Winter', 'Spring', 'Summer', 'Autumn']] = pd.get_dummies(data['release_date'].apply(season)).astype(bool)
data['runtime'].fillna(data['runtime'].mean(), inplace=True)  # Mean Imputation
data['spoken_languages_count'] = data['spoken_languages'].apply(count_spoken_languages)
data['log1p_budget'] = np.log1p(data['budget'])

data = data.drop(['backdrop_path', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'original_language',
                  'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries',
                  'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'video', 'Keywords', 'cast',
                  'crew'], axis=1)

model = CatBoostRegressor().load_model('catboost.model')
predicted = model.predict(data[model_features])

# Example:
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = np.expm1(predicted)
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


### Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


### Example - Calculating RMSLE
res = rmsle(data['revenue'], prediction_df['revenue'])
print("RMSLE is: {:.6f}".format(res))


