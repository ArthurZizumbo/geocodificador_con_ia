# variables.tf
# Variable definitions for the main infrastructure

variable "project_id" {
  description = "ID del proyecto de Google Cloud"
  type        = string
  validation {
    condition     = length(var.project_id) > 0
    error_message = "El project_id no puede estar vacío."
  }
}

variable "region" {
  description = "Región de Google Cloud donde desplegar los recursos"
  type        = string
  default     = "us-central1"
  
  validation {
    condition = contains([
      "us-central1", "us-east1", "us-west1", "us-west2", "us-west3", "us-west4",
      "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west6",
      "asia-east1", "asia-northeast1", "asia-southeast1", "asia-south1"
    ], var.region)
    error_message = "La región debe ser una región válida de Google Cloud."
  }
}

variable "environment" {
  description = "Ambiente de deployment"
  type        = string
  default     = "prod"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "El ambiente debe ser dev, staging o prod."
  }
}

variable "app_name" {
  description = "Nombre base para los recursos de la aplicación"
  type        = string
  default     = "geocodificador-ine"
  
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,61}[a-z0-9]$", var.app_name))
    error_message = "El nombre de la aplicación debe seguir las convenciones de nomenclatura de GCP."
  }
}

# Optional variables for future expansion
variable "machine_type" {
  description = "Tipo de máquina para instancias de Compute Engine"
  type        = string
  default     = "e2-medium"
}

variable "min_replicas" {
  description = "Número mínimo de réplicas para autoescalado"
  type        = number
  default     = 1
  
  validation {
    condition     = var.min_replicas >= 0
    error_message = "El número mínimo de réplicas debe ser mayor o igual a 0."
  }
}

variable "max_replicas" {
  description = "Número máximo de réplicas para autoescalado"
  type        = number
  default     = 10
  
  validation {
    condition     = var.max_replicas >= 1
    error_message = "El número máximo de réplicas debe ser mayor o igual a 1."
  }
}