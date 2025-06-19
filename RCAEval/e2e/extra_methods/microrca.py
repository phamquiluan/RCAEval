import math
import pandas as pd
import networkx as nx
from .interface import RootCauseAnalysis
from typing import List, Tuple
from .utils import attributed_graph, birch_ad_with_smoothing, rt_invocations
import logging

class MicroRCA(RootCauseAnalysis):
    def __init__(self, exp_dir_path):
        super().__init__(exp_dir_path)
        # Tuning parameters
        self.alpha = 0.55  

    def _node_weight(self, svc, anomaly_graph, baseline_df):
        #Get the average weight of the in_edges
        in_edges_weight_avg = 0.0
        num = 0
        for _, _, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            in_edges_weight_avg = in_edges_weight_avg + data['weight']
        if num > 0:
            in_edges_weight_avg  = in_edges_weight_avg / num

        df = pd.read_csv(self.base_dir.joinpath("data.csv"))
        df = df.loc[:, df.columns.str.contains(svc)]
        node_cols = ['node_cpu', 'node_network', 'node_memory']
        max_corr = 0.01
        metric = 'node_cpu'
        for col in node_cols:
            temp = abs(baseline_df[svc].corr(df[f"{svc}_{col}"]))
            if temp > max_corr:
                max_corr = temp
                metric = col
        data = in_edges_weight_avg * max_corr
        return data, metric

    def _svc_personalization(self, svc, anomaly_graph, baseline_df):
        df = pd.read_csv(self.base_dir.joinpath("data.csv"))
        df = df.loc[:, df.columns.str.contains(svc)]
        ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
        max_corr = 0.01
        metric = 'ctn_cpu'
        for col in ctn_cols:
            temp = abs(baseline_df[svc].corr(df[f"{svc}_{col}"]))     
            if temp > max_corr:
                max_corr = temp
                metric = col


        edges_weight_avg = 0.0
        num = 0
        for _, v, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

        for _, v, data in anomaly_graph.out_edges(svc, data=True):
            if anomaly_graph.nodes[v]['type'] == 'service':
                num = num + 1
                edges_weight_avg = edges_weight_avg + data['weight']

        edges_weight_avg  = edges_weight_avg / num

        personalization = edges_weight_avg * max_corr

        return personalization, metric


    def _anomaly_subgraph(self, DG, anomalies, latency_df, alpha):
        # Get the anomalous subgraph and rank the anomalous services
        # input: 
        #   DG: attributed graph
        #   anomlies: anoamlous service invocations
        #   latency_df: service invocations from data collection
        #   agg_latency_dff: aggregated service invocation
        #   faults_name: prefix of csv file
        #   alpha: weight of the anomalous edge
        # output:
        #   anomalous scores 

        # Get reported anomalous nodes
        edges = []
        nodes = []
        baseline_df = pd.DataFrame()
        edge_df = {}
        for anomaly in anomalies:
            edge = anomaly.split('_')
            edges.append(tuple(edge))
            svc = edge[1]
            nodes.append(svc)
            baseline_df[svc] = latency_df[anomaly]
            edge_df[svc] = anomaly

        nodes = set(nodes)

        personalization = {}
        for node in DG.nodes():
            if node in nodes:
                personalization[node] = 0

        # Get the subgraph of anomaly
        anomaly_graph = nx.DiGraph()
        for node in nodes:
            for u, v, data in DG.in_edges(node, data=True):
                edge = (u,v)
                if edge in edges:
                    data = alpha
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[v].corr(latency_df[normal_edge])

                # It is possible that the correlation is corrupted, so we set it to zero
                # This is to avoid the impact of the anomaly on the overall graph
                # Possible reason: Observed network communication between two nodes but no actual requests
                data = round(0 if math.isnan(data) or math.isinf(data) else data, 3)
                assert not math.isnan(data), f"data is NaN for in-edge {u} -> {v}"
                assert not math.isinf(data), f"data is inf for in-edge {u} -> {v}"
                if data == 0:
                    logging.debug(f"data is 0 for in-edge {u} -> {v}, either by accident or due to our corrective action")
                anomaly_graph.add_edge(u,v, weight=data)
                anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
                anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

        # Set personalization with container resource usage
            for u, v, data in DG.out_edges(node, data=True):
                edge = (u,v)
                if edge in edges:
                    data = alpha
                else:
                    if DG.nodes[v]['type'] == 'host':
                        data, _ = self._node_weight(u, anomaly_graph, baseline_df)
                    else:
                        normal_edge = u + '_' + v
                        data = baseline_df[u].corr(latency_df[normal_edge])

                # It is possible that the correlation is corrupted, so we set it to zero
                # This is to avoid the impact of the anomaly on the overall graph
                # Possible reason: Observed network communication between two nodes but no actual requests
                data = round(0 if math.isnan(data) or math.isinf(data) else data, 3)
                assert not math.isnan(data), f"data is NaN for out-edge {u} -> {v}"
                assert not math.isinf(data), f"data is inf for out-edge {u} -> {v}"
                if data == 0:
                    logging.debug(f"data is 0 for out-edge {u} -> {v}, either by accident or due to our corrective action")
                anomaly_graph.add_edge(u,v, weight=data)
                anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
                anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']


        for node in nodes:
            max_corr, _ = self._svc_personalization(node, anomaly_graph, baseline_df)
            assert not math.isnan(max_corr), f"max_corr is NaN for node {node}"
            assert not math.isinf(max_corr), f"max_corr is inf for node {node}"
            personalization[node] = max_corr / anomaly_graph.degree(node)
            assert not math.isnan(personalization[node]), f"personalization is NaN for node {node}"
            assert not math.isinf(personalization[node]), f"personalization is inf for node {node}"
            if personalization[node] <= 0:
                logging.debug(f"personalization is <=0 for node {node}, setting it to 0")
                personalization[node] = 0

        anomaly_graph = anomaly_graph.reverse(copy=True)

        edges = list(anomaly_graph.edges(data=True))

        for u, v, d in edges:
            if anomaly_graph.nodes[node]['type'] == 'host':
                anomaly_graph.remove_edge(u,v)
                anomaly_graph.add_edge(v,u,weight=d['weight'])

        return self.pagerank(anomaly_graph, personalization)


    def execute(self) -> List[Tuple[str, float]]:
        # construct attributed graph
        DG = attributed_graph(self.base_dir.joinpath("mpg.csv"))

        latency_df_detailed = rt_invocations(self.base_dir.joinpath("latency_merged_90.csv"))
        # anomaly detection on response time of service invocation
        anomalies = birch_ad_with_smoothing(latency_df_detailed, self.ad_threshold, self.smoothing_window)

        # get the anomalous service
        anomaly_nodes = []
        for anomaly in anomalies:
            edge = anomaly.split('_')
            anomaly_nodes.append(edge[1])

        anomaly_nodes = set(anomaly_nodes)

        anomaly_score = self._anomaly_subgraph(DG, anomalies, latency_df_detailed, self.alpha)

        anomaly_scores = []
        for anomaly_target in anomaly_score:
            node = anomaly_target[0]
            if DG.nodes[node]['type'] == 'service':
                anomaly_scores.append(anomaly_target)
        return anomaly_scores