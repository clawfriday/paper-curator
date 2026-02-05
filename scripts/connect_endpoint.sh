#!/bin/bash

# # binds login node port to compute node port
# ssh hopper \
#   'ssh -f -N -L 8001:localhost:8001 hopper-45'

# ssh hopper \
#   'ssh -f -N -L 8002:localhost:8002 hopper-34'

# ssh hopper \
#   'ssh -f -N -L 8004:localhost:8004 hopper-34'

# binds local PC port 8001 to hopper login node port 8001, doing it in background
ssh -f -N -L 8000:localhost:8005 hopper

ssh -f -N -L 8001:localhost:8001 hopper
ssh -f -N -L 8002:localhost:8002 hopper
ssh -f -N -L 8004:localhost:8004 hopper

# binds local PC port 3000 to hopper login node port 3000
ssh -f -N -L 3000:localhost:3000 hopper

############################################################
# test the port forwarding via curl
############################################################
echo "--- LLM ---" && \
# curl -s http://localhost:8001/v1/models
export OPENAI_API_BASE="http://localhost:8001"
export OPENAI_API_BASE="https://transmissively-conidial-fredrick.ngrok-free.dev"
curl -s ${OPENAI_API_BASE}/v1/models | \
python3 -c "import sys, json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))"

echo && echo "--- VLM ---" && \
# curl -s http://localhost:8002/v1/models
# export OPENAI_API_BASE2="http://hopper-34:8002"
export OPENAI_API_BASE2="http://localhost:8002"
curl -s ${OPENAI_API_BASE2}/v1/models | \
python3 -c "import sys, json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))"

echo && echo "--- Embedding ---" && \
# curl -s http://localhost:8004/v1/models
export OPENAI_API_BASE3="http://localhost:8004"
curl -s ${OPENAI_API_BASE3}/v1/models | \
python3 -c "import sys, json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))"