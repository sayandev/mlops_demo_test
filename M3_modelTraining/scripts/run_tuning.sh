# =============================================================================
# scripts/run_tuning.sh
# =============================================================================
#!/bin/bash
set -e

echo "🔍 Starting hyperparameter tuning..."

# Default values
DATA_PATH="/app/data/kaggle_fraud.csv"
NUM_SAMPLES=20
MAX_EPOCHS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if services are running
if ! docker compose ps | grep -q "Up"; then
    echo "🚢 Starting services..."
    docker compose up -d
    sleep 10
fi

echo "⏳ Waiting for MLflow to be ready..."
for i in {1..20}; do
    if docker compose exec ray-head curl -s http://mlflow:5050 > /dev/null; then
        echo "✅ MLflow is up!"
        break
    fi
    sleep 2
done

# Run tuning
echo "🎯 Tuning with parameters:"
echo "  Data: $DATA_PATH"
echo "  Number of trials: $NUM_SAMPLES"
echo "  Max epochs: $MAX_EPOCHS"

docker compose exec ray-head python /app/tune_ray.py \
    --data "$DATA_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --max_epochs "$MAX_EPOCHS"

echo "✅ Hyperparameter tuning completed!"
echo "📊 View results at: http://localhost:5050"
echo "🔍 View Ray dashboard at: http://localhost:8265"