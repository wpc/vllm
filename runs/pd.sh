#!/bin/bash

# Container settings (override as needed)
: "${CONTAINER_IMAGE:=/home/dmoss/scratch/images/vllm-custom-kimi.sqsh}"
: "${CONTAINER_MOUNTS:=/home/dmoss/scratch:/scratch}"
: "${CONTAINER_WORKDIR:=/scratch/runs}"

echo "Partition: $SLURM_JOB_PARTITION"
echo "$SLURM_JOB_NUM_NODES nodes allocated"
if [ "$SLURM_JOB_NUM_NODES" -le 6 ]; then
    echo "Error: This script requires more than 6 nodes, but only $SLURM_JOB_NUM_NODES node(s) allocated"
    exit 1
fi

# Print the node layout
echo "Node layout:"
echo "SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
echo "  -- RANK 0: Router + Benchmark"
echo "  -- RANK 1: Prefill - DEP8 - 1/2"
echo "  -- RANK 2: Prefill - DEP8 - 2/2"
echo "  -- RANK 3: Prefill - DEP8 - 1/2"
echo "  -- RANK 4: Prefill - DEP8 - 2/2"
echo "  -- RANK 5: Decode - DEP8 - 1/2"
echo "  -- RANK 6: Decode - DEP8 - 2/2"


# Get all node hostnames and extract their numbers (improved parsing)
ALL_NODE_NUMS=()

# export nsys=true

while IFS= read -r hostname; do
  [ -n "$hostname" ] && ALL_NODE_NUMS+=("$hostname")
done < <(scontrol show hostnames "$SLURM_NODELIST" 2>/dev/null)

PREFILL_1_HOSTNAME="${ALL_NODE_NUMS[1]}"
PREFILL_2_HOSTNAME="${ALL_NODE_NUMS[2]}"
PREFILL_3_HOSTNAME="${ALL_NODE_NUMS[3]}"
PREFILL_4_HOSTNAME="${ALL_NODE_NUMS[4]}"
DECODE_1_HOSTNAME="${ALL_NODE_NUMS[5]}"
DECODE_2_HOSTNAME="${ALL_NODE_NUMS[6]}"

ALL_NODE_NUMS_STR="${ALL_NODE_NUMS[*]}"
DPS="4"
srun --segment $SLURM_JOB_NUM_NODES --ntasks-per-node=1 sudo nvidia-smi -ac 3996,2062

srun --segment $SLURM_JOB_NUM_NODES \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$CONTAINER_WORKDIR" \
  --mpi=pmix \
  bash -c "
RANK=0
for node in $ALL_NODE_NUMS_STR; do
    if [ \"\$node\" = \"\$(hostname -s)\" ]; then
        break
    fi
    RANK=\$((RANK + 1))
done
if [ \"\$RANK\" = 1 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_1_HOSTNAME: $PREFILL_1_HOSTNAME RANK: \$RANK DPS: 8\"
    just prefill-master $PREFILL_1_HOSTNAME 8
elif [ \"\$RANK\" = 2 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_2_HOSTNAME: $PREFILL_2_HOSTNAME RANK: \$RANK DPS: 8\"
    just prefill-worker $PREFILL_1_HOSTNAME 4 8
elif [ \"\$RANK\" = 3 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_3_HOSTNAME: $PREFILL_3_HOSTNAME RANK: \$RANK DPS: 8\"
    just prefill-master $PREFILL_3_HOSTNAME 8
elif [ \"\$RANK\" = 4 ]; then
    echo \"HOSTNAME: \$(hostname -s)  PREFILL_4_HOSTNAME: $PREFILL_4_HOSTNAME RANK: \$RANK DPS: 8\"
    just prefill-worker $PREFILL_3_HOSTNAME 4 8
elif [ \"\$RANK\" = 5 ]; then
    echo \"HOSTNAME: \$(hostname -s)  DECODE_1_HOSTNAME: $DECODE_1_HOSTNAME RANK: \$RANK DPS: 8\"
    just decode-master $DECODE_1_HOSTNAME 8
elif [ \"\$RANK\" = 6 ]; then
    echo \"HOSTNAME: \$(hostname -s)  DECODE_2_HOSTNAME: $DECODE_2_HOSTNAME RANK: \$RANK DPS: 8\"
    just decode-worker $DECODE_1_HOSTNAME 4 8
elif [ \"\$RANK\" = 0 ]; then
    while ! curl -s http://$PREFILL_1_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$PREFILL_2_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$PREFILL_3_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$PREFILL_4_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$DECODE_1_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    while ! curl -s http://$DECODE_2_HOSTNAME:8087/health > /dev/null 2>&1; do
        sleep 1
    done
    echo \"vLLM is ready, creating router server...\"
    just router 4 1 http://$PREFILL_1_HOSTNAME:8087 http://$PREFILL_2_HOSTNAME:8087 http://$PREFILL_3_HOSTNAME:8087 http://$PREFILL_4_HOSTNAME:8087 http://$DECODE_1_HOSTNAME:8087 http://$DECODE_2_HOSTNAME:8087 &
    echo \"router is ready, running accuracy...\"
    just accuracy 0.0.0.0  8192
    echo \"router server is ready, running benchmark...\"
    just bench 0.0.0.0 8192 5120 2048 8192 1024
else
    echo \"Something went wrong...\"
fi
"


