import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from sklearn.cluster import Birch


def attributed_graph(mpg_filename):
    mpg_df = pd.read_csv(mpg_filename)
    
    DG = nx.DiGraph()    
    for _, row in mpg_df.iterrows():
        source = row['source']
        destination = row['destination']
        DG.add_edge(source, destination)

    for node in DG.nodes():
        if ':10250' in node: 
            DG.nodes[node]['type'] = 'host'
        else:
            DG.nodes[node]['type'] = 'service'
                
    return DG

def causality_graph(mpg_filename):
    mpg_df = pd.read_csv(mpg_filename)
    
    DG = nx.DiGraph()    
    for _, row in mpg_df.iterrows():
        source = row['source']
        destination = row['destination']
        if ':10250' not in destination:
            DG.add_edge(source, destination)     

    return DG 

def rt_invocations(latency_filename):

    latency_df = pd.read_csv(latency_filename)  

    return latency_df

def birch_ad_with_smoothing(latency_df, threshold, smoothing_window):
    # anomaly detection on response time of service invocation. 
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation
    anomalies = []
    for svc, latency in latency_df.items():
        # No anomaly detection in db
        if not any([opt in svc for opt in ['unknown', 'timestamp', 'time', 'index']]):
            latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1,1)

            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies

def aggregate_latency(latency_df):
    agg_data = {}
    for node in set(sum([col.split('_') for col in latency_df.columns], [])):
        relevant_cols = [col for col in latency_df.columns if node in col.split('_')]
        agg_data[node] = latency_df[relevant_cols].sum(axis=1)
    aggregated_df = pd.DataFrame(agg_data)
    return aggregated_df