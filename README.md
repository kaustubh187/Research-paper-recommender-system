# Research Paper Recommender System
With the vast amount of scholarly articles and research papers being published every day, it can be challenging for researchers, students, and academics to keep up with the latest developments in their field and find the most relevant papers for their research. The process of discovering relevant literature is time-consuming and often requires manual searching, which can lead to missed opportunities and gaps in knowledge. Therefore, there is a need for a recommender system that can assist researchers in discovering relevant research papers efficiently and effectively. The recommender system must take into account the user's preferences and needs to provide personalized recommendations that are accurate. 
A system that maps research papers based on their keywords and topics using natural language processing. The similarity between different research papers will then be calculated using a suitable similarity metric. Topic-wise clustering would be done to map research papers according to topics and similarities between them and then recommendations will be made based on these clusters. The clusters formed will be used to get a visual representation of similar research papers.
The proposed system will help researchers to discover relevant research papers more efficiently, even in highly specialized domains. Moreover, it will help researchers stay up-to-date with the latest research trends and potentially discover new research collaborations. The system will also provide a personalized and accessible way for researchers to discover new research papers, improving the efficiency and effectiveness of their research.
# Dataset Used
## Arxiv Dataset https://www.kaggle.com/datasets/Cornell-University/arxiv
 
The arXiv dataset is a collection of scientific papers that have been submitted to arXiv.org, a repository of electronic preprints of scientific papers in various fields including mathematics, physics, computer science, and more. 
The arXiv dataset includes papers from a wide range of disciplines and is widely used by researchers to access the latest research in their field. The dataset is maintained by arXiv and is available for download through their website or through third-party services. 
The dataset can be used for various purposes such as natural language processing, machine learning, and data mining research, and it is a valuable resource for researchers who want to study the latest developments in their fields. 

# Approach
- Combined all the preprocessed text and calculated the Term Frequency-Inverse Document Frequency (TF-IDF) vector for each research paper.
- Used the TF-IDF matrix to calculate cosine similarity between research papers.
- Used cosine similarity score to recommend similar research papers for a given research paper.
- To search and recommend research papers to users, we created a web app using Flask.
- Used Principal Component Analysis (PCA) to project down the dimensions of the TF-IDF matrix to several dimensions that will keep a .95 variance
- Apply K-Means Clustering on the new data and use these clusters to make recommendations.
- Apply Dimensionality Reduction to each feature vector using t-Distributed Stochastic Neighbour Embedding (t-SNE) to cluster similar research articles in the two-dimensional plane.
- Apply Topic Modeling on the clustered data using Latent Dirichlet Allocation (LDA) to discover keywords from each cluster.

# Flask Web Application
A flask application and API were developed to implement the recommender system into a user interface. This app works on a dataset that contains approximately 250,000 research paper data. In this application, there are 3 options for searching research papers - 
Search for papers by name (keyword search)
In this we can search for research papers by entering the name of the research paper, it performs a keyword search across the database, based on titles. We can also choose the number of search results we need. It will return a group of research papers with their titles, authors, and ID. From this list, there are three options to redirect 

# Screenshots
![image](https://github.com/kaustubh187/Research-paper-recommender-system/assets/81306562/04e6c50e-aa31-4047-b68e-e6821d9ae02f)


### Choose type of search
![image](https://github.com/kaustubh187/Research-paper-recommender-system/assets/81306562/0c7a3777-030a-4043-b8fa-17f0b6c3de2a)



### Enter name of research paper
![image](https://github.com/kaustubh187/Research-paper-recommender-system/assets/81306562/bfe298e8-46a5-4964-a450-478e08ba7879)

### Get Results after keyword search

![image](https://github.com/kaustubh187/Research-paper-recommender-system/assets/81306562/4edf7c8a-9dfb-4d28-931c-5b5f381b73d9)

### Get research paper details 

![image](https://github.com/kaustubh187/Research-paper-recommender-system/assets/81306562/f9ba9492-dcd5-4234-b545-5fa3208c6c24)

### Get recommendations for the research paper

![image](https://github.com/kaustubh187/Research-paper-recommender-system/assets/81306562/f6c6257f-e669-450a-9022-a1f84cff15f4)


### Cluster of 1000 simillar research papers
