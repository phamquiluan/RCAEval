from .interface import RootCauseAnalysis
from .utils import aggregate_latency, birch_ad_with_smoothing, rt_invocations, causality_graph


class MicroScope(RootCauseAnalysis):
        
    def _scoring(self, node, candidate, latency_df):
        corr = latency_df[node].corr(latency_df[candidate])
        return corr
    
    def _cause_inference(self, DG, latency_df, entry_point, anomaly_nodes):
        candidates = []
        
        stack = [entry_point]
        visited = set()  # Track visited nodes

        while len(stack) > 0:
            stack = set(stack)
            node = stack.pop()
            stack = list(stack)
            
            if node in visited:
                continue
            
            visited.add(node)  # Mark node as visited
            
            neighbors = []
            for n in DG.neighbors(node):
                neighbors.append(n)
            
            if len(neighbors) == 0:
                candidates.append(node)
                continue
            
            children = []
            for neighbor in neighbors:
                if neighbor in anomaly_nodes and neighbor not in visited:
                    children.append(neighbor)
                    stack.append(neighbor)
            if len(children) == 0:
                candidates.append(node)

        candidates_score = {}
        for candidate in candidates:
            candidates_score[candidate] = self._scoring(node, candidate, latency_df)

        candidates_score = sorted(candidates_score.items(), key=lambda x: x[1], reverse=True)
        return candidates_score 
    
    def execute(self):
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
        return self._cause_inference(DG, latency_df_agg, self.entry_point, anomaly_nodes)