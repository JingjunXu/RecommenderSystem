import datetime
import random

import os
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from main import train, eval

# Show app title and description.
st.set_page_config(page_title="Recommender System", page_icon="🎫")
st.title("🎫 Recommender System")
st.write(
    """
    This app demonstrate a simple demo of how to use GNN-based autoencoder 
    to implement a Recommender System.
    """
)

# Create the dataset
# Function used to load the data
def read_data():
    data_dir = os.path.join("data", "ml-100k")
    # edge data
    edge_train = pd.read_csv(os.path.join(data_dir, 'u1.base'), sep='\t',
                                header=None, names=['User_ID', 'Movie_ID', 'Rating', 'timestamp'])
    edge_train.loc[:, 'usage'] = 'train'
    edge_test = pd.read_csv(os.path.join(data_dir, 'u1.test'), sep='\t',
                            header=None, names=['User_ID', 'Movie_ID', 'Rating', 'timestamp'])
    edge_test.loc[:, 'usage'] = 'test'
    edge_df = pd.concat((edge_train, edge_test),
                        axis=0).drop(columns='timestamp')
    edge_df.loc[:, 'Rating'] -= 1
    # item feature
    sep = r'|'
    movie_file = os.path.join(data_dir, 'u.item')
    movie_headers = ['Movie_ID', 'Movie_Title', 'Release_Date', 'Video_Release_Date',
                        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']
    movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                            names=movie_headers, encoding='latin1')
    # user feature
    users_file = os.path.join(data_dir, 'u.user')
    users_headers = ['User_ID', 'Age',
                        'Gender', 'Occupation', 'Zip_code']
    users_df = pd.read_csv(users_file, sep=sep, header=None,
                            names=users_headers, encoding='latin1')
    return edge_df, users_df, movie_df
# Load Data
interactions, users, movies = read_data()
# Add data into session_state
if "interactions" not in st.session_state:
    st.session_state.interactions = interactions
if "users" not in st.session_state:
    st.session_state.users = users
if "movies" not in st.session_state:
    st.session_state.movies = movies

# Show information of the data
st.header("Data overview")
st.info(
    "You can edit the data by double clicking on a cell. Note how the plots below "
    "update automatically! You can also sort the table by clicking on the column headers.",
    icon="✍️",
)
# Interaction Information
st.header("Interactions between Users and Movies")
st.write(f"Number of interactions: `{len(st.session_state.interactions)}`")
# Show information of the data
# cells. The edited data is returned as a new dataframe.
edited_interactions = st.data_editor(
    st.session_state.interactions,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rating": st.column_config.SelectboxColumn(
            "Rating",
            help="Rating Types",
            options=["1", "2", "3", "4", "5"],
            required=True,
        )
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=["User_ID", "Movie_ID", 'usage'],
)
# Users Information
st.header("Users Information")
st.write(f"Number of users: `{len(st.session_state.users)}`")
# Show information of the data
# cells. The edited data is returned as a new dataframe.
edited_users = st.data_editor(
    st.session_state.users,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Gender": st.column_config.SelectboxColumn(
            "Gender",
            help="Gender Types",
            options=["F", "M"],
            required=True,
        )
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=['User_ID', 'Age', 'Occupation', 'Zip_code'],
)
# Movies Information
st.header("Movie Information")
st.write(f"Number of movies: `{len(st.session_state.movies)}`")
# Show information of the data
# cells. The edited data is returned as a new dataframe.
edited_movies = st.data_editor(
    st.session_state.movies,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Action": st.column_config.SelectboxColumn(
            "Action",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Adventure": st.column_config.SelectboxColumn(
            "Adventure",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Animation": st.column_config.SelectboxColumn(
            "Animation",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Childrens": st.column_config.SelectboxColumn(
            "Childrens",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Comedy": st.column_config.SelectboxColumn(
            "Comedy",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Crime": st.column_config.SelectboxColumn(
            "Crime",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Documentary": st.column_config.SelectboxColumn(
            "Documentary",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Drama": st.column_config.SelectboxColumn(
            "Drama",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Fantasy": st.column_config.SelectboxColumn(
            "Fantasy",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Film-Noir": st.column_config.SelectboxColumn(
            "Film-Noir",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Horror": st.column_config.SelectboxColumn(
            "Horror",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Musical": st.column_config.SelectboxColumn(
            "Musical",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Mystery": st.column_config.SelectboxColumn(
            "Mystery",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Romance": st.column_config.SelectboxColumn(
            "Romance",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Sci-Fi": st.column_config.SelectboxColumn(
            "Sci-Fi",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Thriller": st.column_config.SelectboxColumn(
            "Thriller",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "War": st.column_config.SelectboxColumn(
            "War",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        ),
        "Western": st.column_config.SelectboxColumn(
            "Western",
            help="Movie Types",
            options=["0", "1"],
            required=True,
        )
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=['Movie_ID', 'Movie_Title', 'Release_Date', 'Video_Release_Date', 'IMDb_URL', 'unknown', 'usage'],
)

# Show a section to add a new ticket.
st.header("Start to Train the Model")

Train = st.button("Train Model")
if Train:
    st.session_state.Results = train()

Load_model = st.button("Load Model", type="primary")
if Load_model:
    st.session_state.Results = eval()

st.session_state.User_ID = st.number_input(
    "Insert the User_ID", value=None, placeholder="Type a User_ID..."
)
if st.session_state.User_ID:
    st.session_state.User_ID = int(st.session_state.User_ID)
st.write("The current User_ID is ", st.session_state.User_ID)

st.session_state.Movie_ID = st.number_input(
    "Insert the Movie_ID", value=None, placeholder="Type a Movie_ID..."
)
if st.session_state.Movie_ID:
    Movie_ID = int(st.session_state.Movie_ID)
st.write("The current Movie_ID is ", st.session_state.Movie_ID)

# Show the Results
st.header("Predict Result")
Predict_Rating = None
Predict = st.button("Predict", type="primary")
# print(Predict)
if Predict:
    # print("Here is")
    if st.session_state.Results is not None:
        # print("Here")
        if (st.session_state.User_ID is not None) and (st.session_state.Movie_ID is not None):
            # print("here")
            for index, row in interactions.iterrows():
                # Access individual values with column names
                user_node = row['User_ID']
                movie_node = row['Movie_ID']
                rating = row['Rating']
                usage = row['usage']
                if (user_node == st.session_state.User_ID) and (movie_node == st.session_state.Movie_ID) and (usage == 'train'):
                    Predict_Rating = rating
                    break
                if (user_node == st.session_state.User_ID) and (movie_node == st.session_state.Movie_ID) and (usage == 'test'):
                    Predict_Rating = st.session_state.Results[index]
                    break

st.write("The prediction of the rating between user ", st.session_state.User_ID, " and movie ", st.session_state.Movie_ID, "is: ", Predict_Rating)

if Predict_Rating is not None:
    if Predict_Rating >= 4:
        st.write("Movie ", st.session_state.Movie_ID, " should be recommended to user ", st.session_state.User_ID)
    else:
        st.write("Movie ", st.session_state.Movie_ID, " should not be recommended to user ", st.session_state.User_ID)