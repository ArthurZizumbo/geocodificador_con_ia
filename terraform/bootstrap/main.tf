# bootstrap/main.tf
# Bootstrap Terraform configuration to create GCS backend and initial resources
# This needs to be applied first with local state, then migrated to remote state

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Initially uses local state, will be migrated to GCS after first apply
}

# Variables
variable "project_id" {
  description = "ID del proyecto de Google Cloud"
  type        = string
}

variable "region" {
  description = "Región de Google Cloud"
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "Nombre del bucket para el estado de Terraform"
  type        = string
}

variable "service_account_name" {
  description = "Nombre de la cuenta de servicio para Terraform"
  type        = string
  default     = "terraform-backend-sa"
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "storage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "compute.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy        = false
}

# Create GCS bucket for Terraform state
resource "google_storage_bucket" "terraform_state" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false

  # Prevent accidental deletion of this bucket
  lifecycle {
    prevent_destroy = true
  }

  # Enable versioning
  versioning {
    enabled = true
  }

  # Enable uniform bucket-level access
  uniform_bucket_level_access = true

  # Server-side encryption (using Google-managed keys)
  # encryption block is optional for Google-managed encryption

  # Public access prevention
  public_access_prevention = "enforced"

  depends_on = [google_project_service.required_apis]
}

# Create service account for Terraform
resource "google_service_account" "terraform_sa" {
  account_id   = var.service_account_name
  display_name = "Terraform Backend Service Account"
  description  = "Service account for managing Terraform state and infrastructure"

  depends_on = [google_project_service.required_apis]
}

# IAM bindings for Terraform service account
resource "google_project_iam_member" "terraform_sa_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.terraform_sa.email}"
}

resource "google_project_iam_member" "terraform_sa_compute_admin" {
  project = var.project_id
  role    = "roles/compute.admin" 
  member  = "serviceAccount:${google_service_account.terraform_sa.email}"
}

resource "google_project_iam_member" "terraform_sa_service_account_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.terraform_sa.email}"
}

resource "google_project_iam_member" "terraform_sa_cloud_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.terraform_sa.email}"
}

# Create service account key
resource "google_service_account_key" "terraform_sa_key" {
  service_account_id = google_service_account.terraform_sa.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# Store the key in a local file (for manual use)
resource "local_file" "terraform_sa_key" {
  content         = base64decode(google_service_account_key.terraform_sa_key.private_key)
  filename        = "../terraform-sa-key.json"
  file_permission = "0600"
}

# Create backend configuration file
resource "local_file" "backend_config" {
  content = templatefile("${path.module}/backend-config.tpl", {
    bucket_name = google_storage_bucket.terraform_state.name
  })
  filename = "../backend-config.hcl"
}

# Create environment variables file
resource "local_file" "terraform_env" {
  content = templatefile("${path.module}/terraform-env.tpl", {
    project_id = var.project_id
    region     = var.region
  })
  filename = "../set-terraform-env.bat"
}