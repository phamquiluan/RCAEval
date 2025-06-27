from argparse import Namespace
import traceback
from main import main

if __name__ == "__main__":
    DATASET = "llm-ref-stack-dp"
    DATASET_ROOT = "/app/data_raw"

    # Reproduce ASE paper experiments / the metric experiments for RCAEval paper
    ase_methods = [
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
        # "run", # fix later, currently not working due to short time series, and generally annoying
        "microcause",
        "e_diagnosis",
        "baro",
        "rcd",
        "circa",
        "nsigma",
    ]
    # Now also do the experiments for MicroRCA, MicroScope, and MonitorRank
    new_methods = [
        "microrca",
        "microscope",
        "monitorrank",
    ]
    
    for method in ase_methods + new_methods:
        try:
            print(f"[Data-Source=Metric] Running {method}...")
            args = Namespace(
                method=method,
                dataset=DATASET,
                dataset_root=DATASET_ROOT,
                length=20,
                tdelta=0,
                test=False
                )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Metric] Error running {method}: {e}")

    # now trace methods and multi-source methods from WWW paper
    trace_methods = [
        "microrank",
        "tracerca"
    ]
    for method in trace_methods:
        try:
            print(f"[Data-Source=Trace] Running {method}...")
            args = Namespace(
                method=method,
                dataset=DATASET,
                dataset_root=DATASET_ROOT,
                length=20,
                tdelta=0,
                test=False
            )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Trace] Error running {method}: {e}")
            traceback.print_exc()
