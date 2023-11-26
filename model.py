

class MicroTopicModeller():

    """
    Gets list of strings as an input. Returns a dictionary containing lists of words by topics and common key words
    """

    def __init__(self, n_clusters=None, sent_transformer='intfloat/multilingual-e5-base', use_pca=False, use_cuda=False, stop_words='english'):
        self.n_clusters = n_clusters # number of clusters. If None, HDBSCAN is used for getting sentence clusters. If stated, KMeans is used instead.
        self.data = None # contains sentence tokens, for more details see get_embeddings(self, data) function
        self.vectorizer = CountVectorizer(stop_words=stop_words, max_features=40000)
        self.use_pca = use_pca
        self.use_cuda = use_cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sent_transformer = SentenceTransformer(sent_transformer).to(self.device)

    def get_embeddings(self, data):
        """
        Split text by sentences and get embeddings. Gets filename as input
        """

        self.data = sent_tokenize(data) #split by punctuation marks at the end of the sentence

    #for large datasets, we recommend using GPU

        print('Getting sentence embeddings...')

        if self.use_cuda:
            sent_embs = self.sent_transformer.encode(self.data, batch_size=256, show_progress_bar=True)
        else:
            sent_embs = self.sent_transformer.encode(self.data) #sentence embeddings; an array of shape (self.n_documents, 768)

    #applying dimensionality reduction if needed

        if self.use_pca:
            embs_pca = PCA(n_components=192)
            sent_embs = embs_pca.fit_transform(sent_embs)
        return sent_embs

    def get_sentence_clusters(self, sent_embs):
        """
        sentences are grouped into clusters with either KMeans (recommended) or HDBSCAN

        returns 2 lists:
        emb_clusters contains lists of embeddings belonging to each cluster
        sent_collection contains corresponding sentences
        """

        if self.n_clusters is not None:
            cluster_maker = KMeans(n_clusters = self.n_clusters)
        else:
            cluster_maker = HDBSCAN(min_cluster_size=3)

        print('Getting sentence clusters...')

        cluster_maker.fit(sent_embs)

        n_clusters = len(np.unique(cluster_maker.labels_))
        emb_clusters = []
        sent_collection = []
        for j in range(n_clusters):
            emb_clusters.append(list())
            sent_collection.append(list())
            for i in range(len(self.data)):
                if cluster_maker.labels_[i] == j - 1: #because we have '-1' cluster
                    emb_clusters[j].append(sent_embs[i])
                    sent_collection[j].append(self.data[i])
        return emb_clusters, sent_collection

    def get_lda(self, sent_collection, cluster_number):

        """
        Getting key words for a sentence cluster (microtopic)
        """

        self.vectorizer.fit(self.data)
        part = self.vectorizer.transform(sent_collection[cluster_number])
        lda_model = LatentDirichletAllocation(n_components=5, learning_method='online', random_state=0, verbose=0)
        lda_topic_matrix = lda_model.fit_transform(part)
        feature_names = self.vectorizer.get_feature_names_out()
        base_words = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_ind = topic.argsort()[: -5 - 1 : -1]
            words = []
            for i in range(5):
                words.append(feature_names[top_features_ind[i]])
            base_words.append(words)
        return base_words

    def get_common_key_words(self, distr):

        """
        Collecting words belonging to more than a third of all microtopics to a separate list
        """

        print('Preparing your results...')

        raw_distr = []
        for wordgroup in distr:
            wg = wordgroup[1] + wordgroup[2] + wordgroup[3] + wordgroup[4] + wordgroup[0]
        raw_distr = raw_distr + wg # get all words from all classes
        word_counts = dict(Counter(raw_distr))
        base_words = [] # get pertinent words from all texts
        n_clusters = len(distr)
        for key, value in word_counts.items():
            if value >= n_clusters // 3:
                base_words.append(key)
        return base_words

    def get_topic_words(self, distr, base_words):

        """
        Creating a dictionary with common key words and key words by microtopics
        """

        topics = {}
        for i, wordgroup in enumerate(distr):
            wg = wordgroup[1] + wordgroup[2] + wordgroup[3] + wordgroup[4] + wordgroup[0]
            group = list(set(wg) - set(base_words))
            groupname = 'topic' + str(i)
            topics[groupname] = group
        return topics

    def pipeline(self, filename):

        """
        Putting it all together
        """

        sent_embs = self.get_embeddings(filename)
        _, sent_collection = self.get_sentence_clusters(sent_embs)
        distributions = []
        print('Computing LDA...')
        for i in range(len(sent_collection)):
            if len(sent_collection[i]) > 0:
                base_words = self.get_lda(sent_collection, i)
                distributions.append(base_words)
        common_key_words = self.get_common_key_words(distributions)
        topic_words = self.get_topic_words(distributions, common_key_words)
        topic_words['common_key_words'] = common_key_words
        print('Complete!')
        return topic_words

