from typing import List, Tuple
import networkx as nx
from .utils import causality_graph, birch_ad_with_smoothing, rt_invocations, aggregate_latency
from .interface import RootCauseAnalysis


class MonitorRank(RootCauseAnalysis):

    def _similarity_score(self, latency_df, node):
        corr = latency_df[self.entry_point].corr(latency_df[node])
        return round(abs(corr), 3)  

    def _self_score(self, DG, latency_df, node):
        self_score = 0
        node_score = self._similarity_score(latency_df, node)
        max_child_score = 0
        if node != self.entry_point:
            for _, v in DG.out_edges(node):
                child_score = self._similarity_score(latency_df, v)
                if child_score > max_child_score:
                    max_child_score = child_score
            
            if (node_score - max_child_score) > self_score:
                self_score = node_score - max_child_score
        return self_score

    def _sub_graph(self, DG, anomaly_nodes, latency_df):
        anomaly_graph = nx.DiGraph()
        for node in anomaly_nodes:
            for u, v in DG.in_edges(node):
                if v != self.entry_point:
                    source = self.rou * self._similarity_score(latency_df, u)
                    destination = self._similarity_score(latency_df, v)
                    anomaly_graph.add_edge(u,v, weight=destination)
                    anomaly_graph.add_edge(v,u, weight=source)

            for u, v in DG.out_edges(node):
                if v != self.entry_point:
                    source = self.rou * self._similarity_score(latency_df, u)
                    destination = self._similarity_score(latency_df, v)
                    anomaly_graph.add_edge(u,v, weight=destination)
                    anomaly_graph.add_edge(v,u, weight=source)   

        personalization = {}

        for node in anomaly_graph.nodes():
            score = self._self_score(DG, latency_df, node)
            anomaly_graph.add_edge(node, node, weight=score)
            personalization[node] = self._similarity_score(latency_df, node)
        personalization[self.entry_point] = 0 

        anomaly_graph = anomaly_graph.reverse(copy=True)

        return self.pagerank(anomaly_graph, personalization)

    def execute(self) -> List[Tuple[str, float]]:
        latency_df_detailed = rt_invocations(self.base_dir.joinpath("latency_merged_90.csv"))
        # anomaly detection on response time of service invocation
        anomalies = birch_ad_with_smoothing(latency_df_detailed, self.ad_threshold, self.smoothing_window)
        
        anomaly_nodes = []
        for anomaly in anomalies:
            edge = anomaly.split('_')
            anomaly_nodes.append(edge[1])                
        
        anomaly_nodes = set(anomaly_nodes)

        latency_df_agg = aggregate_latency(latency_df_detailed)
        
        DG = causality_graph(self.base_dir.joinpath("mpg.csv"))
        return self._sub_graph(DG, anomaly_nodes, latency_df_agg)
                    
                    

