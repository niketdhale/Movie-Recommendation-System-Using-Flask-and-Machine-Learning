import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('complete.csv')

pd.set_option('display.max_columns', None)
# data.head()


# data.tail(3)

data.shape

# data.describe()

# data.info(verbose=False)
# data[data.columns[:]].isnull().sum()

data[data['imdb_score'] > 7.5].shape[0]

################################################################
# plt.rcParams['figure.figsize']=(18,9)
# data_groupby_ratings = data.groupby(['imdb_score'])['movie_title'].count()
# data_groupby_ratings.plot()
###############################################################
# data_groupby_duration = data.groupby(['duration'])['movie_title'].count()
# data_groupby_duration.plot()
#################################################################
data[data['duration'] <= 100].shape[0]

data[data['duration'] >= 180].shape[0]
################################################################
# data.boxplot(column='duration', by='imdb_score');
###############################################################
# data.boxplot(column='duration', by='content_rating')
###############################################################
data['language'].unique()
###############################################################
# sns.set(style="darkgrid")
# plt.figure(figsize = (12, 6))
# sns.countplot(x="language", data = data)
# ax = plt.xticks(rotation=90)
###############################################################
# sns.set(style="darkgrid")
# sns.countplot(x="color", data = data)
###############################################################
# data_groupby_gross = data.groupby(['title_year'])['gross'].count()
# data_groupby_gross.plot()
###############################################################
# data_groupby_gross = data.groupby(['title_year'])['budget'].count()
# data_groupby_gross.plot()
###############################################################
data[data['language'] == 'English'].shape[0]  # number of english movies

highest_imdb = data.sort_values('imdb_score', ascending=False)
high = highest_imdb.loc[:,
       ['movie_title', 'imdb_score', 'title_year', 'language', 'country', 'budget', 'director_name', 'duration',
        'gross']]
high.head(10)

def highest_imdb_movies():
    highest_imdb = data.sort_values('imdb_score', ascending=False)
    high = highest_imdb['movie_title'].head(10)
    # high = highest_imdb.head(10)
    #print(high)
    return high

#highest_imdb_movies()

french = high[high['language'] == 'French']
french.head(5)

prop_missing = round((data[data.columns[:]].isnull().sum() / data.shape[0]) * 100, 2)
prop_missing

# col_filling = []
# for s in data.columns:
#     ratio = (len(data[s])-data[s].isnull().sum()) / len(data[s])*100
#     number = data[s].notnull().sum()
#     col_filling.append([ratio, s, number])
# col_filling.sort(key = lambda x:x[0])
# #------------------------------------
# for ratio, s, number in col_filling:
#     print("{:<30} -> {:<6}%".format(s, round(ratio,2)))


clean_data = data[data.title_year.notnull() & data.duration.notnull()]
len(clean_data)

clean_data.loc[:, 'title_year'] = clean_data['title_year'].astype(int).astype(str)
clean_data.loc[:, 'year'] = pd.to_datetime(clean_data['title_year'], format='%Y')

clean_data.describe()

df_1 = clean_data[['title_year', 'movie_title']]
ser = df_1.groupby(df_1.title_year.astype(int) // 10 * 10).size()
df = pd.DataFrame({'decade': ser.index, 'movies': ser.values})
###############################################################
# fig,ax = plt.subplots()
# ax.bar(df.decade, df.movies, width=2.6, color='b')
# ax.set_xticks(df.decade+1.3)  # set the x ticks to be at the middle of each bar since the width of each bar is 2.6
# ax.set_xticklabels(df.decade)  #replace the name of the x ticks with your Groups name
# ax.grid(False) #remove gridlines
# plt.xlabel('Decade', fontsize=16)
# plt.ylabel('No of movies released', fontsize=16)
# plt.title('Movies released by decade', fontsize=24)
# plt.show()
###############################################################

data['decade'] = data['title_year'].apply(lambda x: ((x - 1900) // 10) * 10)


# __________________________________________________________________
# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}


# ______________________________________________________________
# Creation of a dataframe with statitical infos on each decade:
test = data['title_year'].groupby(data['decade']).apply(get_stats).unstack()['decade'] = data['title_year'].apply(
    lambda x: ((x - 1900) // 10) * 10)


# __________________________________________________________________
# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}


# ______________________________________________________________
# Creation of a dataframe with statitical infos on each decade:
test = data['title_year'].groupby(data['decade']).apply(get_stats).unstack()

###############################################################
# sns.set_context("poster", font_scale=0.85)
# #_______________________________
# # funtion used to set the labels
# def label(s):
#     val = (1900 + s, s)[s < 100]
#     chaine = '' if s < 50 else "{}'s".format(val)
#     return chaine
# #    if s < 50:        
# #        return ''
# #    elif s < 100:
# #        return "{}'s".format(int(s))
# #    else:
# #        return "{}'s".format(int(1900+s))
# #____________________________________
# plt.rc('font', weight='bold')
# f, ax = plt.subplots(figsize=(14, 6))
# labels = [label(s) for s in  test.index]
# sizes  = test['count'].values
# explode = [0.2 if sizes[i] < 100 else 0.01 for i in range(11)]
# ax.pie(sizes, explode = explode, labels=labels,
#        autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
#        shadow=False, startangle=0)
# ax.axis('equal')
# ax.set_title('% of films per decade',
#              bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
###############################################################

genre_labels = set()
for s in data['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))


def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in liste_keywords:
            if pd.notnull(s): keyword_count[s] += 1
    # ______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    keyword_occurences.sort(key=lambda x: x[1], reverse=True)
    return keyword_occurences, keyword_count


keyword_occurences, dum = count_word(data, 'genres', genre_labels)
keyword_occurences[:5]

###############################################################
# def random_color_func(word=None, font_size=None, position=None,
#                       orientation=None, font_path=None, random_state=None):
#     h = int(360.0 * tone / 255.0)
#     s = int(100.0 * 255.0 / 255.0)
#     l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
#     return "hsl({}, {}%, {}%)".format(h, s, l)


# words = dict()
# trunc_occurences = keyword_occurences[0:50]
# for s in trunc_occurences:
#     words[s[0]] = s[1]
# tone = 100 # define the color of the words
# f, ax = plt.subplots(figsize=(14, 6))
# wordcloud = WordCloud(width=550,height=300, background_color='black', 
#                       max_words=1628,relative_scaling=0.7,
#                       color_func = random_color_func,
#                       normalize_plurals=False)
# wordcloud.generate_from_frequencies(words)
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()
###############################################################

set_keywords = set()
for liste_keywords in data['plot_keywords'].str.split('|').values:
    if type(liste_keywords) == float: continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)

keyword_occurences, dum = count_word(data, 'plot_keywords', set_keywords)
keyword_occurences[:5]

###############################################################
# _____________________________________________
# # UPPER PANEL: WORDCLOUD
# fig = plt.figure(1, figsize=(18,13))
# ax1 = fig.add_subplot(2,1,1)
# #_______________________________________________________
# # I define the dictionary used to produce the wordcloud
# words = dict()
# trunc_occurences = keyword_occurences[0:50]
# for s in trunc_occurences:
#     words[s[0]] = s[1]
# tone = 55.0 # define the color of the words
# #________________________________________________________
# wordcloud = WordCloud(width=1000,height=300, background_color='black', 
#                       max_words=1628,relative_scaling=1,
#                       color_func = random_color_func,
#                       normalize_plurals=False)
# wordcloud.generate_from_frequencies(words)
# ax1.imshow(wordcloud, interpolation="bilinear")
# ax1.axis('off')
# #_____________________________________________
# # LOWER PANEL: HISTOGRAMS
# ax2 = fig.add_subplot(2,1,2)
# y_axis = [i[1] for i in trunc_occurences]
# x_axis = [k for k,i in enumerate(trunc_occurences)]
# x_label = [i[0] for i in trunc_occurences]
# plt.xticks(rotation=85, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xticks(x_axis, x_label)
# plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)
# ax2.bar(x_axis, y_axis, align = 'center', color='g')
# #_______________________
# plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)
# plt.show()
###############################################################


###############################################################
# #get data
# temp2 = clean_data[['title_year', 'imdb_score']]
# #plot
# temp2 = temp2.groupby(temp2.title_year.astype(int)).imdb_score.mean().plot(kind ='line', grid =False, title ='IMDB Average Score Trend', xlim=((1950, 2016)))
# temp2.xaxis.set_ticks(np.arange(1950, 2016, 7))
# plt.xlabel('Year', fontsize=18)
# plt.ylabel('Average IMDB Score', fontsize=18)
# plt.tight_layout()
###############################################################


###############################################################
# temp = clean_data[['title_year', 'imdb_score', 'movie_imdb_link']]
# temp = temp[temp.title_year.astype(int)>1949]
# res = temp.groupby(temp.title_year.astype(int)).agg({'imdb_score': 'mean', 'movie_imdb_link': 'count'}).reset_index()
# res.columns = ['title_year', 'avg_imdb_score', 'movies_created']
# rows = res.title_year

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(res.title_year, res.avg_imdb_score, color = 'blue')
# ax1.set_ylabel('Average IMDB Score', color = 'blue')
# ax1.set_xlabel('Year')
# ax1.grid(False)
# #ax1.legend(loc = 'upper right')


# ax2 = ax1.twinx()
# ax2.plot(res.title_year, res.movies_created, color='green')
# ax2.set_ylabel('Movies Released', color = 'green')
# ax2.grid(False)
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')
# #ax2.legend(loc = 'upper right')
# plt.title('Avg IMDB Score vs Movies Released ~ Trend')

# plt.show()
###############################################################


###############################################################
# Get data required for the plot
# temp = clean_data[['content_rating', 'imdb_score']]
# temp = temp.groupby(temp.content_rating.astype(str)).imdb_score.mean()
# df = pd.DataFrame({'Content_Rating':temp.index, 'IMDB_Score':temp.values})
# #sort data by score descending
# df = df.sort_values(['IMDB_Score'], ascending=[False])
# #plot
# df.plot('Content_Rating','IMDB_Score', kind='bar')
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=10)
# plt.xticks(rotation=0)
# plt.grid(False)
# plt.xlabel('Content Rating', fontsize=14)
# plt.ylabel('Average IMDB Score', fontsize=14)
# plt.title('Average IMDB Score per Content Rating')
# plt.tight_layout()
# plt.show()
###############################################################


# Overall, Approved and Passed content has highest score but there isnt much difference in scores for different content rating types, however TV-14 has lowest IMDB Score


###############################################################
# ###### 1.6 Director name
# Get data required for the plot
# temp = clean_data[['director_name', 'imdb_score']]
# temp = temp.groupby(temp.director_name.astype(str)).imdb_score.mean()
# df = pd.DataFrame({'Director_Name':temp.index, 'Avg_IMDB_Score':temp.values})
# #sort data by score descending
# df = df.sort_values(['Avg_IMDB_Score'], ascending=[False])
# df = df.head(10)
# #plot while sorting plot
# df.sort_values('Avg_IMDB_Score').plot('Director_Name','Avg_IMDB_Score', kind='barh')
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=10)
# plt.xticks(rotation=0)
# plt.grid(False)
# plt.xlabel('Director', fontsize=14)
# plt.ylabel('Average IMDB Score', fontsize=14)
# plt.title('Top 10 high scoring directors')
# plt.tight_layout()
# plt.legend().set_visible(False)
# plt.show()
###############################################################


###############################################################
# Movie Facebook Likes
# temp = clean_data[['movie_facebook_likes', 'imdb_score']]
# temp = temp[temp.imdb_score > 0]
# x = temp.plot(x='movie_facebook_likes', y = 'imdb_score',kind='scatter', xlim = (0, 100000), title='IMDB Score VS Movie facebook likes', legend=[True])
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=10)
# plt.grid(False)
# plt.xlabel('Movie Facebook Likes', fontsize=14)
# plt.ylabel('Average IMDB Score', fontsize=14)
# plt.title('Movie FB Likes VS IMDB Score')
# plt.tight_layout()
# plt.show()
###############################################################

# temp[['imdb_score','movie_facebook_likes']].corr()

###############################################################
# #Duration IMDB Score and Language
# temp = clean_data[['duration', 'imdb_score']]
# temp = temp.plot('duration', 'imdb_score', kind ='scatter', title ='Duration VS Mean IMDB Score')
# plt.xlabel('Duration', fontsize=12)
# plt.ylabel('Avg IMDB Score', fontsize=12)
# plt.title('Duration VS Avg IMDB Score')
# plt.grid(False)
# plt.tight_layout()
# plt.show()
###############################################################


# clean_data[['imdb_score','duration']].corr()

###############################################################
# Duration vs Avg IMDB Score
# temp = clean_data[['duration', 'imdb_score']]
# sns.regplot(x="duration", y="imdb_score", data=temp);
# plt.xlabel('Duration', fontsize=12)
# plt.ylabel('Avg IMDB Score', fontsize=12)
# plt.title('Duration VS Avg IMDB Score')
# plt.grid(False)
# plt.tight_layout()
# plt.show()
###############################################################


###############################################################
# Duration of Movie By Language
# temp = clean_data[['language', 'duration', 'title_year']]
# temp1 = temp[temp.title_year.astype(int) >= 2000]
# temp1 = temp1.loc[temp1['language'].isin(['English','Hindi'])]
# # temp1 = temp1[temp1.language == 'English' | temp1.language == 'Hindi']
# temp1.groupby(temp1.language).duration.mean().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.xlabel('Language')
# plt.ylabel('Avg Duration')
# plt.title('Duration of movie by Language')
# plt.grid(False)
# plt.show()
###############################################################


# hindi = temp1[temp1.language == 'Hindi']
# english = temp1[temp1.language == 'English'] 


# print("The dataset has {} Hindi and {} English movies".format(len(hindi), len(english)) )


# As we can see from the above plot, Hindi movies are longer than English movies, however the Hindi movie dataset only has 26 movies while the English movie dataset has 3308 movies (all realeased since year 2000). We need more data for Hindi movies in order to draw a comparison. Below we plot the histogram for these two datasets:
###############################################################
# plot histogram for hindi movie durations
# hindi.duration.plot(kind='hist',color='0.5', bins = 10, title = 'Histogram for duration of Hindi movies').set_xlabel('Duration')
# hindi_mean = round(hindi["duration"].mean(),2)
# hindi_sd = round((hindi["duration"]).std(),2)
# print("The mean duration of the Hindi movies is {} and standard deviation is {}".format(hindi_mean, hindi_sd))
###############################################################


###############################################################
# plot histogram for english movie durations
# english.duration.plot(kind='hist',color='0.5', bins = 10, title = 'Histogram for duration of English movies').set_xlabel('duration')
# english_mean = round(english["duration"].mean(),2)
# english_sd = round((english["duration"]).std(),2)
# print("The mean duration of the English movies is {} and standard duration is {}".format(english_mean, english_sd) )
###############################################################

#  - We see that the amount of movies created over time is increasing exponentially while the average IMDB score for the movies is decreasing over time
#  - We see a positive correlation between IMDB Score and Movie facebook likes and between Duration of movie and its IMDB Score
#  - We see a difference in mean duration of movies created since 2000 between groups of Hindi and English movies, this leads of us formulate a Hypothesis test to test if the durations are similar between the groups or not


#################################################
# # II. Movie Recommender Systems


data_use = data.loc[:, ['genres', 'plot_keywords', 'movie_title', 'actor_1_name',
                        'actor_2_name', 'actor_3_name', 'director_name', 'imdb_score']]

data_use['movie_title'] = [i.replace("\xa0", "") for i in list(data_use['movie_title'])]

# ###### 1.1 Cleaning


# print(data_use.shape)
clean_data = data_use.dropna(axis=0)
# print(clean_data.shape)
clean_data = clean_data.drop_duplicates(['movie_title'])
clean_data = clean_data.reset_index(drop=True)
# print(clean_data.shape)


people_list = []
for i in range(clean_data.shape[0]):
    name1 = clean_data.loc[i, 'actor_1_name'].replace(" ", "_")
    name2 = clean_data.loc[i, 'actor_2_name'].replace(" ", "_")
    name3 = clean_data.loc[i, 'actor_3_name'].replace(" ", "_")
    name4 = clean_data.loc[i, 'director_name'].replace(" ", "_")
    people_list.append("|".join([name1, name2, name3, name4]))
clean_data['people'] = people_list

# ###### 1.2 CountVectorizer


from sklearn.feature_extraction.text import CountVectorizer


def token(text):
    return (text.split("|"))


cv_kw = CountVectorizer(max_features=100, tokenizer=token)
keywords = cv_kw.fit_transform(clean_data["plot_keywords"])
keywords_list = ["kw_" + i for i in cv_kw.get_feature_names()]

cv_ge = CountVectorizer(tokenizer=token)
genres = cv_ge.fit_transform(clean_data["genres"])
genres_list = ["genres_" + i for i in cv_ge.get_feature_names()]

cv_pp = CountVectorizer(max_features=100, tokenizer=token)
people = cv_pp.fit_transform(clean_data["people"])
people_list = ["pp_" + i for i in cv_pp.get_feature_names()]

cluster_data = np.hstack([keywords.todense(), genres.todense(), people.todense() * 2])
criterion_list = keywords_list + genres_list + people_list

# ###### 1.3 KMeans


from sklearn.cluster import KMeans

mod = KMeans(n_clusters=100)
category = mod.fit_predict(cluster_data)
category_dataframe = pd.DataFrame({"category": category}, index=clean_data['movie_title'])

clean_data.loc[list(category_dataframe['category'] == 0), ['genres', 'movie_title', 'people']]


# ###### 1.4 Recommender System


def recommend(movie_name, recommend_number=20):
    if movie_name in list(clean_data['movie_title']):
        movie_cluster = category_dataframe.loc[movie_name, 'category']
        score = clean_data.loc[list(category_dataframe['category'] == movie_cluster), ['imdb_score', 'movie_title']]
        sort_score = score.sort_values(['imdb_score'], ascending=[0])
        sort_score = sort_score[sort_score['movie_title'] != movie_name]
        recommend_number = min(sort_score.shape[0], recommend_number)
        recommend_movie = list(sort_score.iloc[range(recommend_number), 1])
        # print(recommend_movie)
        return recommend_movie
    else:
        print("Can't find this movie!")


# def top_10_movies():
#   highest_imdb = data.sort_values('imdb_score', ascending = False)
#   high = highest_imdb.loc[:,['movie_title', 'imdb_score','title_year', 'language', 'country', 'budget', 'director_name', 'duration', 'gross' ]]
#   print(dict(high.head(1)))


# recommend('Spider-Man')
# recommend('Avatar')


def function_to_return_link(movie_name):
    movie = data.loc[data['movie_title'].str.contains(movie_name), 'poster']
    try:
        link = movie.iat[0]
    except IndexError:
        return 'None'
    else:
        # print(link)
        return {'title': movie_name, 'poster': link}


def Return_details(movie_name):
    t = data.loc[data['movie_title'].str.contains(movie_name), 'movie_title']
    imdb = data.loc[data['movie_title'].str.contains(movie_name), 'imdb_score']
    g = data.loc[data['movie_title'].str.contains(movie_name), 'genres']
    y = data.loc[data['movie_title'].str.contains(movie_name), 'title_year']
    p = data.loc[data['movie_title'].str.contains(movie_name), 'poster']
    time = data.loc[data['movie_title'].str.contains(movie_name), 'duration']
    b = data.loc[data['movie_title'].str.contains(movie_name), 'budget']
    act_1 = data.loc[data['movie_title'].str.contains(movie_name), 'actor_1_name']
    act_2 = data.loc[data['movie_title'].str.contains(movie_name), 'actor_2_name']
    act_3 = data.loc[data['movie_title'].str.contains(movie_name), 'actor_3_name']
    direct  = data.loc[data['movie_title'].str.contains(movie_name), 'director_name']
    key  = data.loc[data['movie_title'].str.contains(movie_name), 'plot_keywords']
    box = data.loc[data['movie_title'].str.contains(movie_name), 'gross']
    content = data.loc[data['movie_title'].str.contains(movie_name), 'content_rating']
    lang = data.loc[data['movie_title'].str.contains(movie_name), 'language']
    desh = data.loc[data['movie_title'].str.contains(movie_name), 'country']
    c = data.loc[data['movie_title'].str.contains(movie_name), 'color']
    imdb_L = data.loc[data['movie_title'].str.contains(movie_name), 'movie_imdb_link']
    try:
        title = t.iat[0]
        imdb_score = imdb.iat[0]
        genre = g.iat[0]
        year = y.iat[0]
        link = p.iat[0]
        running_time = time.iat[0]
        budget = b.iat[0]
        actor_1 = act_1.iat[0]
        actor_2 = act_2.iat[0]
        actor_3 = act_3.iat[0]
        director = direct.iat[0]
        keyword = key.iat[0]
        box_office = box.iat[0]
        content_rating = content.iat[0]
        language = lang.iat[0]
        country = desh.iat[0]
        color = c.iat[0]
        imdb_link = imdb_L.iat[0]
    except IndexError:
        return {'None': 'None'}
    else:
        # print(movie_name,link,imdb_score,genre,year,int(running_time),int(budget),actor_1,actor_2,actor_3,director,keyword,int(box_office),content_rating,language,country,color,imdb_link)
        return {'title': movie_name, 'poster': link, 'imdb_score': imdb_score, 'genre': genre, 'year': year,'running_time':running_time,'budget':budget,'actor_1':actor_1,'actor_2':actor_2,'actor_3':actor_3,'director':director,'keyword':keyword,'box_office':box_office,'content_rating':content_rating,'language':language,'country':country,'color':color,'imdb_link':imdb_link}
# function_to_return_link('Avatar')
# Return_details('Avatar')
