from typing import List, Tuple
import networkx as nx
import time
import pathlib

class RootCauseAnalysis:
    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = base_dir
        self.entry_point = "nginx-proxy"
        self.smoothing_window = 10
        self.ad_threshold = 0.045
        self.rou = 0.5

    def execute(self) -> List[Tuple[str, float]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def compute_scores(self):
        anomaly_scores = self._execute()
        return anomaly_scores

    def pagerank(self, anomaly_graph, personalization):
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            try:
                anomaly_score = nx.pagerank(
                    anomaly_graph,
                    alpha=0.85,
                    personalization=personalization,
                    max_iter=10000
                )
                anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)
                return anomaly_score
            except nx.exception.PowerIterationFailedConvergence as e:
                retry_count += 1
                time.sleep(1)
        raise Exception("Failed to compute pagerank")
