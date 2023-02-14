# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # My Notebook

# +
import pandas as pd
import stringcase

from sklearn.datasets import load_iris, load_wine


# -

def rename_cols(df):
    return df.rename(columns={
        col: stringcase.camelcase(col)
        for col in df.columns
    })


# ## Iris Data

iris_data = load_iris(as_frame=True)
X, y = rename_cols(iris_data.data), iris_data.target

X

# ## Wine Data

wine_data = load_wine(as_frame=True)
X, y = rename_cols(wine_data.data), wine_data.target

X


