#!/bin/bash
# setup_gcp.sh - Script to set up GCP environment for the pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${1:-"civic-nation-461607-p6"}
REGION=${2:-"us-central1"}
CLUSTER_NAME=${3:-"iris-ml-cluster"}
REPO_NAME=${4:-"ml-models"}

echo -e "${GREEN}Setting up GCP environment for ML CI/CD Pipeline${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Cluster Name: $CLUSTER_NAME"
echo "Repository Name: $REPO_NAME"

# Verify gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}gcloud CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}Setting project to $PROJECT_ID${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com

# Create Artifact Registry repository
echo -e "${YELLOW}Creating Artifact Registry repository...${NC}"
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for ML models" || echo "Repository might already exist"

# Configure Docker authentication for Artifact Registry
echo -e "${YELLOW}Configuring Docker authentication...${NC}"
gcloud auth configure-docker $REGION-docker.pkg.dev

# Create GKE cluster with smaller configuration to avoid quota issues
echo -e "${YELLOW}Creating GKE cluster (this may take several minutes)...${NC}"
gcloud container clusters create $CLUSTER_NAME \
    --zone=$REGION-a \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=3 \
    --machine-type=e2-medium \
    --disk-type=pd-standard \
    --disk-size=50GB \
    --enable-autorepair \
    --enable-autoupgrade \
    --enable-network-policy \
    --enable-ip-alias || echo "Cluster might already exist"

# Get cluster credentials
echo -e "${YELLOW}Getting cluster credentials...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$REGION-a

# Create namespace for the application
echo -e "${YELLOW}Creating Kubernetes namespace...${NC}"
kubectl create namespace iris-classifier || echo "Namespace might already exist"
kubectl config set-context --current --namespace=iris-classifier

# Create service account for GitHub Actions (optional)
echo -e "${YELLOW}Creating service account for CI/CD...${NC}"
gcloud iam service-accounts create github-actions-sa \
    --description="Service account for GitHub Actions CI/CD" \
    --display-name="GitHub Actions SA" || echo "Service account might already exist"

# Grant necessary permissions to the service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.clusterAdmin"

# Create and download service account key
echo -e "${YELLOW}Creating service account key...${NC}"
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com

# Set up firewall rules (if needed)
echo -e "${YELLOW}Setting up firewall rules...${NC}"
gcloud compute firewall-rules create allow-iris-api \
    --allow tcp:8200,tcp:80,tcp:443 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow access to Iris API" || echo "Firewall rule might already exist"

echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - GCP_PROJECT_ID: $PROJECT_ID"
echo "   - GCP_SA_KEY: Contents of github-actions-key.json"
echo ""
echo "2. Update your workflow file with correct project ID and region"
echo ""
echo "3. Commit and push your code to trigger the pipeline"
echo ""
echo -e "${YELLOW}Important files created:${NC}"
echo "- github-actions-key.json (add this content to GitHub secrets)"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "- View cluster: gcloud container clusters list"
echo "- View repositories: gcloud artifacts repositories list"
echo "- View services: kubectl get services"
echo "- View pods: kubectl get pods"
echo ""
echo -e "${YELLOW}Cluster Configuration:${NC}"
echo "- Machine Type: e2-medium (1 vCPU, 4GB RAM)"
echo "- Disk: 50GB standard persistent disk per node"
echo "- Nodes: 2 (autoscaling 1-3)"
echo "- Total Disk Usage: ~100GB (well within your 250GB quota)"
