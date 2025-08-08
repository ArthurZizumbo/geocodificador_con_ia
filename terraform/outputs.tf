# outputs.tf
# Main infrastructure outputs

output "project_id" {
  description = "ID del proyecto"
  value       = var.project_id
}

output "region" {
  description = "Región configurada"
  value       = var.region
}

output "environment" {
  description = "Ambiente actual"
  value       = var.environment
}

output "vpc_network" {
  description = "Nombre de la red VPC"
  value       = google_compute_network.vpc.name
}

output "subnet_name" {
  description = "Nombre de la subred"
  value       = google_compute_subnetwork.subnet.name
}

output "app_data_bucket" {
  description = "Nombre del bucket para datos de la aplicación"
  value       = google_storage_bucket.app_data.name
}

output "vpc_network_id" {
  description = "ID completo de la red VPC"
  value       = google_compute_network.vpc.id
}

output "subnet_id" {
  description = "ID completo de la subred"
  value       = google_compute_subnetwork.subnet.id
}