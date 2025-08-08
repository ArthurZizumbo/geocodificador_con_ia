# bootstrap/outputs.tf
# Outputs for bootstrap resources

output "project_id" {
  description = "ID del proyecto de Google Cloud"
  value       = var.project_id
}

output "terraform_state_bucket" {
  description = "Nombre del bucket para el estado de Terraform"
  value       = google_storage_bucket.terraform_state.name
}

output "terraform_sa_email" {
  description = "Email de la cuenta de servicio de Terraform"
  value       = google_service_account.terraform_sa.email
}

output "terraform_sa_key_file" {
  description = "Ruta al archivo de clave de la cuenta de servicio"
  value       = local_file.terraform_sa_key.filename
  sensitive   = true
}

output "backend_config_file" {
  description = "Ruta al archivo de configuración del backend"
  value       = local_file.backend_config.filename
}

output "next_steps" {
  description = "Siguientes pasos después del bootstrap"
  value = <<EOT
1. Ejecuta: cd ..
2. Ejecuta: set-terraform-env.bat
3. Ejecuta: terraform init -backend-config=backend-config.hcl
4. Ejecuta: terraform plan
5. Ejecuta: terraform apply
EOT
}