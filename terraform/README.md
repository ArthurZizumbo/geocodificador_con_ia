# Terraform Infrastructure for INE Geocoding System

This directory contains Terraform configurations for deploying the INE Geocoding System infrastructure to Google Cloud Platform.

## Architecture Overview

The infrastructure follows Infrastructure as Code principles with a two-phase deployment approach:

1. **Bootstrap Phase**: Creates the foundational resources needed for remote state management
2. **Main Phase**: Deploys the application infrastructure using remote state

## Two-Phase Deployment Process

### Prerequisites

- Google Cloud SDK installed and authenticated (`gcloud auth login`)
- Terraform >= 1.0 installed
- Project configured in `bootstrap/terraform.tfvars`

### Phase 1: Bootstrap

The bootstrap phase creates:
- GCS bucket for Terraform remote state
- Service account for Terraform operations
- Required IAM bindings
- Configuration files for the main deployment

```bash
cd bootstrap/
terraform init
terraform plan
terraform apply
```

**Outputs**:
- `../terraform-sa-key.json` - Service account credentials
- `../backend-config.hcl` - Backend configuration
- `../set-terraform-env.bat` - Environment variables

### Phase 2: Main Infrastructure

The main phase creates:
- VPC network and subnet
- Cloud NAT for outbound connectivity
- Firewall rules for health checks
- GCS bucket for application data

```bash
cd .. # Back to terraform/ directory
set-terraform-env.bat
terraform init -backend-config=backend-config.hcl
terraform plan
terraform apply
```

## Directory Structure

```
terraform/
├── bootstrap/           # Bootstrap configuration
│   ├── main.tf         # Bootstrap resources
│   ├── outputs.tf      # Bootstrap outputs
│   ├── terraform.tfvars # Bootstrap variables
│   └── *.tpl           # Template files
├── main.tf             # Main infrastructure
├── variables.tf        # Variable definitions
├── outputs.tf          # Infrastructure outputs
├── terraform.tfvars    # Main variables (gitignored)
└── README.md           # This file
```

## Security Considerations

- Service account follows principle of least privilege
- GCS buckets have versioning enabled
- VPC uses private subnets with Cloud NAT
- Sensitive files are excluded from version control

## Clean Up

To destroy the infrastructure:

```bash
# Destroy main infrastructure first
terraform destroy

# Then destroy bootstrap (if needed)
cd bootstrap/
terraform destroy
```

**Warning**: Destroying the bootstrap will delete the Terraform state bucket. Ensure you have backups if needed.

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure your user has sufficient permissions or use the generated service account
2. **Backend Bucket Not Found**: Run bootstrap phase first
3. **Resource Conflicts**: Check for existing resources with same names

### Useful Commands

```bash
# Check current state
terraform state list

# Import existing resources
terraform import google_storage_bucket.example bucket-name

# Refresh state
terraform refresh
```