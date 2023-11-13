# micro_topic_modelling
Topic modelling for a small collection of closely related texts

## Basic Description
MicroTopicModelling is a BERT-based and LDA-based method of finding common keywords for a set of closely related (i.e., dedicated to the same topic and written in the same style) texts and indentifying their microtopics. Class "MicroTopicModeller" gets a txt file as input and returns a dictionary containing common key words and key words for specific topics as output. Besides, you can extract sentence clusters and print out the whole collection of texts in different colors, each color corresponding to a specific cluster.

## Model Architecture.

For our model, we applied the following architecture:
![image](https://github.com/belom-nlp/micro_topic_modelling/assets/135965144/9e938c3f-15fe-4ee7-9e46-d84bc9ad270b)

The first stage of text processing is its splitting by sentences, which is implemented here with nltk realization. Then we get embeddings of all the sentences by means of BERT model (for our purposes, we used 'INSEEEEEEEEEEEEEEEEEEEEEERT' model; the reader is encouraged to use any BERT model available on Huggingface). Sentence embeddings (with or without dimensionality reduction) are then divided into clusters via KMeans or HDBSCAN (the first showed even better results on trial). For each cluster, Latent Dirichlet Allocation based topic modelling is applied in order to get key words for microtopics. Words that are common for more than a third of microtopics are considered common key words for the whole collection of texts; they are then collected in a separate list.

## Research Purposes

This model was created in order to investigate the way how media shape the image of a certain event in public opinion.
