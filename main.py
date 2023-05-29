from flask import Flask,render_template,request,redirect,url_for,session
import pandas as pd
import numpy as np # for array manipulation
import json # for reading in Data
from itertools import islice # for slicing and dicing JSON records
import os # for getting the filepath information
import re # to identify characters that are to be removed
import nltk # for preprocessing of textual data
from nltk.corpus import stopwords # for removing stopwords
from nltk.tokenize import word_tokenize # for tokenizing text
from nltk.stem import WordNetLemmatizer # for lemmatizing text
from sklearn.feature_extraction.text import TfidfVectorizer # for featurizing text
from sklearn.metrics.pairwise import cosine_similarity # for getting similarity score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #for dimensionality reduction
from sklearn.cluster import KMeans #for clustering
from sklearn.manifold import TSNE
import plotly.express as px 
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.io as pio


df = pd.read_csv('initial_df.csv',dtype={'id': "str"})
idf = df.reset_index()


final_df = pd.read_csv('final_df.csv', dtype={'id': "str"})
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer2 = TfidfVectorizer(max_features=10000)
tfidf_matrix2 = tfidf_vectorizer2.fit_transform(final_df['final_text'])

# Generate the tf-idf vectors for the data
tfidf_matrix = tfidf_vectorizer.fit_transform(final_df['final_text'])



def get_recommendations(paper_id:str,tfidf_matrix,num_rec)-> list:
    idx = final_df.index[final_df['id'] == paper_id][0]
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix[idx])
    sim = sim.reshape(sim.shape[0])
    top_n_idx = np.argsort(-sim)[1:num_rec+1]
    top_n_id = [final_df['id'][x] for x in top_n_idx]
    return top_n_id

def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names_out()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
                
    keywords.sort(key = lambda x: x[1])  
    keywords.reverse()
    return_values = []
    for x in keywords:
        return_values.append(x[0])
    return " ".join(return_values)


def search_dataframe(df, column_name, keywords, num_results=None):
    """
    Perform search based on keywords in a specified column of a Pandas DataFrame
    
    Parameters:
    df (Pandas DataFrame): The DataFrame to search
    column_name (str): The name of the column to search in
    keywords (str or list of str): The keywords to search for
    num_results (int, optional): The maximum number of results to return
    
    Returns:
    Pandas DataFrame: A subset of the original DataFrame containing only the records that match the search criteria
    """
    if isinstance(keywords, str):
        keywords = [keywords]
        
    # Create a regular expression pattern that matches any of the keywords
    pattern = "|".join([re.escape(keyword) for keyword in keywords])
    
    # Apply the regular expression search to the specified column
    matches = df[column_name].str.findall(pattern, flags=re.IGNORECASE)
    
    # Create a boolean mask for each row that has at least one match
    mask = matches.apply(lambda x: len(x) > 0)
    
    # Apply the mask to the original DataFrame
    result_df = df[mask]
    
    # Optionally limit the number of results returned
    if num_results is not None:
        result_df = result_df.head(num_results)
    
    return result_df



app = Flask(__name__)

@app.route('/')
def welcomepage():
    return render_template('home.html')

@app.route('/search-by-name', methods = ['GET','POST'])
def searchname():
    if request.method=='POST':
        if request.form["action"]=="Enter":
            rnam = request.form['rname']
            npapers = request.form['npapers']
            print("Name of research paper: "+ rnam)
            return redirect (url_for('results',rname=rnam,n=npapers))
    return render_template('searchbyname.html')



@app.route('/recommend', methods = ['GET','POST'])
def recommend():
    if request.method=='POST':
        if request.form["action"]=="Enter":
            rid = request.form['rid']
            npapers = request.form['npapers']
            print("ID of research paper: "+ rid)
            return redirect (url_for('rec_results',rid=rid,n=npapers))
    return render_template('get_recommend.html')


@app.route('/rec-results/<rid>/<int:n>')
def rec_results(rid,n):
    rid=str(rid)
    dataorg = idf[idf['id']==rid]
    hula = pd.DataFrame()
    hula = hula.append(dataorg,ignore_index=True)
    orgtitle = hula['title'][0]
    mylist = get_recommendations(rid,tfidf_matrix,n)
    #print(mylist)
    recms = pd.DataFrame()
    for i in mylist:
        temp = idf[idf['id']==i]
        recms = recms.append(temp, ignore_index = True)
    answer = recms.to_dict(orient='index')
    #print("Answer: " + str(answer))
    return render_template('recoresults.html', props=answer,title=orgtitle)



@app.route('/search-by-id',methods = ['GET','POST'])
def search_byid():
    if request.method=='POST':
        if request.form["action"]=="Enter":
            rid = request.form['rid']
            print("ID of research paper: "+ rid)
            return redirect (url_for('paper_info',rid=rid))
    return render_template('search_papers.html')

@app.route('/paper-info/<rid>')
def paper_info(rid):
    ans = idf[idf['id']==rid]
    ans = ans.to_dict('records')
    #print(ans)
    return render_template('paperinfo.html',anscol=ans)

@app.route('/results/<rname>/<int:n>')
def results(rname,n):
    result = search_dataframe(df, 'title', rname,n)
    jsonresult = result.to_dict(orient='index')
    return render_template('search_result.html',props=jsonresult,name=rname)

# @app.route('/cluster-load/<id>')
# def loadcluster(id):
#     return render_template("loadingpage.html",rid=id)

@app.route('/cluster-graph/<id>')
def recocluster(id):
    rec = get_recommendations(id,tfidf_matrix,1000)
    idxs = list(final_df[final_df['id'].isin(rec)].index)
    rec_matrix = tfidf_matrix2[idxs]
    pca = PCA(n_components=0.95, random_state=42) #Keep 95% of the variance
    reduced_matrix = pca.fit_transform(rec_matrix.toarray())
    k = 10 # selectable
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(reduced_matrix)
    tsne = TSNE(perplexity=100, random_state=42)
    two_dim_matrix = tsne.fit_transform(reduced_matrix)
    topic_df = pd.DataFrame()
    topic_df['id'] =df[final_df['id'].isin(rec)]['id']
    topic_df['title'] = df[final_df['id'].isin(rec)]['title']
    topic_df['text'] = df[final_df['id'].isin(rec)]['title']+" "+df[final_df['id'].isin(rec)]['abstract']
    topic_df['cluster'] = y_pred
    vectorized_data = []
    
    vectorizers = []
        
    for x in range(0, k):
        # Creating a vectorizer
        vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))

    for current_cluster, cvec in enumerate(vectorizers):
        try:
            vectorized_data.append(cvec.fit_transform(topic_df.loc[topic_df['cluster'] == current_cluster, 'text']))
        except Exception as e:
            print("Not enough instances in cluster: " + str(current_cluster))
            vectorized_data.append(None)
    NUM_TOPICS_PER_CLUSTER = 5 #choose

    lda_models = []
    for x in range(0, k):
        # Latent Dirichlet Allocation Model
        lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=42)
        lda_models.append(lda)
    clusters_lda_data = []

    for current_cluster, lda in enumerate(lda_models):
        #print("Current Cluster: " + str(current_cluster))
        
        if vectorized_data[current_cluster] != None:
            clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
    all_keywords = []
    for current_vectorizer, lda in enumerate(lda_models):
        #print("Current Cluster: " + str(current_vectorizer))

        if vectorized_data[current_vectorizer] != None:
            all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))
    cluster_keyword = {x:all_keywords[x] for x in range(k)}
    word_pred = list(map(cluster_keyword.get, y_pred))
    topic_df['keywords'] = word_pred
    fig = px.scatter(topic_df, x=two_dim_matrix[:,0], y=two_dim_matrix[:,1], color='keywords',
                 hover_data=['id','title'],
                 height= 800, width=1200,
                title = "Clustered Papers")
    fig.update_layout(
    coloraxis_colorbar=dict(
        title_font=dict(size=20), # Set the font size of the color legend title to 20
        tickfont=dict(size=16) # Set the font size of the color legend tick labels to 16
    ),
    legend=dict(
        font=dict(size=13) # Set the font size of the marker legend to 16
    )
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1)
    )
    graph_html = pio.to_html(fig, full_html=False)
    return render_template('cluser_result.html',graph=graph_html)

# @app.route('/final-cluster/<graph>')
# def final_graph(graph):
#     return render_template('cluser_result.html',graph=graph)


app.run(debug=True)
