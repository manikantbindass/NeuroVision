variable "region" {
  type = string
}
variable "name" {
  type = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,29}$", var.name))
    error_message = "Use a short lowercase project identifier."
  }
}
variable "vpc_id" {
  type = string
}
variable "private_subnet_ids" {
  type = list(string)
  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "Supply private subnets in at least two availability zones."
  }
}
variable "kubernetes_version" {
  type        = string
  description = "Choose an EKS-supported Kubernetes version for the region."
}
variable "cpu_instance_types" {
  type    = list(string)
  default = ["t3.large"]
}
variable "enable_gpu_nodes" {
  type    = bool
  default = false
}
variable "gpu_instance_types" {
  type    = list(string)
  default = ["g5.xlarge"]
}
variable "redis_auth_token" {
  type      = string
  sensitive = true
  validation {
    condition     = length(var.redis_auth_token) >= 32
    error_message = "Supply an operator-managed Redis token with at least 32 characters."
  }
}
variable "artifact_bucket_name" {
  type = string
}
variable "operator_role_arn" {
  type        = string
  description = "Existing IAM operator role authorized to administer this research cluster."
}
