#!/bin/bash

# =============================================================================
# rocprofv3 Profiling Script for Llama 3.1 Training on AMD MI300X
# =============================================================================
#
# Usage:
#   ./run_rocprof.sh                          # Default: trace mode
#   ./run_rocprof.sh trace                    # HIP API + kernel tracing
#   ./run_rocprof.sh counters                 # Hardware performance counters
#   ./run_rocprof.sh trace configs/llama_8b.yaml   # With custom config
#
# Prerequisites:
#   - ROCm 6.x+ installed
#   - rocprofv3 available in PATH
# =============================================================================

set -e

MODE="${1:-trace}"
CONFIG="${2:-configs/llama_8b.yaml}"
OUTPUT_DIR="profiling_output/rocprof"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "rocprofv3 Profiling for Llama 3.1 Training"
echo "============================================================"
echo "  Mode:       ${MODE}"
echo "  Config:     ${CONFIG}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Timestamp:  ${TIMESTAMP}"
echo ""

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    echo "ERROR: rocprofv3 not found in PATH."
    echo "Please ensure ROCm is installed and rocprofv3 is available."
    echo "Try: export PATH=\$PATH:/opt/rocm/bin"
    exit 1
fi

echo "ROCm version:"
rocprofv3 --version 2>/dev/null || echo "  (version check not supported)"
echo ""

# Small profiling run (fewer steps to keep trace manageable)
PROFILE_CMD="python profile_training.py --config ${CONFIG} --num-steps 3 --warmup-steps 2 --output-dir ${OUTPUT_DIR}/pytorch_prof"

case "${MODE}" in
    trace)
        echo "=== TRACING MODE ==="
        echo "Collecting HIP API calls and GPU kernel dispatches..."
        echo ""

        # Trace HIP runtime API and kernel activity
        rocprofv3 \
            --hip-trace \
            --kernel-trace \
            --output-directory "${OUTPUT_DIR}/trace_${TIMESTAMP}" \
            -- ${PROFILE_CMD}

        echo ""
        echo "Trace output saved to: ${OUTPUT_DIR}/trace_${TIMESTAMP}/"
        echo ""
        echo "To view results:"
        echo "  - Load the .json files in https://ui.perfetto.dev/"
        echo "  - Or use: rocprofv3 --output-format csv ... for CSV output"
        ;;

    counters)
        echo "=== COUNTER COLLECTION MODE ==="
        echo "Collecting hardware performance counters..."
        echo ""

        # Create counter input file for MI300X-relevant metrics
        COUNTER_FILE="${OUTPUT_DIR}/counters_input.txt"
        cat > "${COUNTER_FILE}" << 'EOF'
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS
pmc: SQ_INSTS_GDS SQ_INSTS_FLAT_LDS_ONLY SQ_INSTS_FLAT_GLOBAL
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_WRREQ_sum TCC_EA_RDREQ_sum
pmc: TA_FLAT_READ_WAVEFRONTS_sum TA_FLAT_WRITE_WAVEFRONTS_sum
pmc: TCP_TCC_READ_REQ_sum TCP_TCC_WRITE_REQ_sum TCP_TCC_ATOMIC_WITH_RET_REQ_sum
pmc: GRBM_COUNT GRBM_GUI_ACTIVE SPI_CSN_WAVE SPI_CSN_NUM_THREADGROUPS
EOF

        echo "Counter config written to: ${COUNTER_FILE}"
        echo ""

        rocprofv3 \
            --counter-file "${COUNTER_FILE}" \
            --output-directory "${OUTPUT_DIR}/counters_${TIMESTAMP}" \
            -- ${PROFILE_CMD}

        echo ""
        echo "Counter results saved to: ${OUTPUT_DIR}/counters_${TIMESTAMP}/"
        ;;

    *)
        echo "ERROR: Unknown mode '${MODE}'."
        echo "Supported modes: trace, counters"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Profiling complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Analyze results:  python analyze_profile.py --input ${OUTPUT_DIR}"
echo "  2. View Chrome trace: open https://ui.perfetto.dev/ and load the .json traces"
echo "  3. Optimize: use custom kernels for identified hotspots"
