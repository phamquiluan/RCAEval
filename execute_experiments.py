from argparse import Namespace
from main import main

if __name__ == "__main__":
    # Reproduce ASE paper experiments / the metric experiments for RCAEval paper
    methods = [
        "dummy",
        "pc_pagerank",
        "pc_randomwalk",
        "fci_pagerank",
        "fci_randomwalk",
        "granger_pagerank",
        "granger_randomwalk",
        "lingam_pagerank",
        "lingam_randomwalk",
        # "ges_pagerank", apparently legacy
        # "ges_randomwalk", apparently legacy
        "ntlr_pagerank",
        "ntlr_randomwalk",
        "causalrca",
        "causalai",
        # "run", fix later, currently not working due to short time series
        "microcause",
        "e_diagnosis",
        "baro",
        "rcd",
        "circa",
        "nsigma",
    ]
    
    for method in methods:
        try:
            print(f"[Data-Source=Metric] Running {method}...")
            args = Namespace(
                method=method,
                dataset="llm-ref-stack-dp",
                dataset_root="/app/data_raw",
                length=20,
                tdelta=0,
            test=False
                )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Metric] Error running {method}: {e}")

    methods = [
        "microrca",
        "microscope",
        "monitorrank",
    ]
    
    # Now also do the experiments for MicroRCA, MicroScope, and MonitorRank
    for method in methods:
        try:
            print(f"[Data-Source=Metric] Running {method}...")
            args = Namespace(
                method=method,
                dataset="llm-ref-stack-dp",
                dataset_root="/app/data_raw",
                length=20,
                tdelta=0,
            test=False
                )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Metric] Error running {method}: {e}")