#!/bin/bash
ln -fs "$PWD/lib/pyAgrum/lib/image.py" "$PWD/env-rcd/lib/python3.8/site-packages/pyAgrum/lib/"
ln -fs "$PWD/lib/causallearn/search/ConstraintBased/FCI.py" "$PWD/env-rcd/lib/python3.8/site-packages/causallearn/search/ConstraintBased/"
ln -fs "$PWD/lib/causallearn/utils/Fas.py" "$PWD/env-rcd/lib/python3.8/site-packages/causallearn/utils/"
ln -fs "$PWD/lib/causallearn/utils/PCUtils/SkeletonDiscovery.py" "$PWD/env-rcd/lib/python3.8/site-packages/causallearn/utils/PCUtils/"
ln -fs "$PWD/lib/causallearn/graph/GraphClass.py" "$PWD/env-rcd/lib/python3.8/site-packages/causallearn/graph/"