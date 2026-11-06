output "cluster_name" {
  value = aws_eks_cluster.research.name
}
output "database_endpoint" {
  value = aws_db_instance.metadata.address
}
output "database_password_secret_arn" {
  value = aws_db_instance.metadata.master_user_secret[0].secret_arn
}
output "redis_endpoint" {
  value = aws_elasticache_replication_group.rate_limits.primary_endpoint_address
}
output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.id
}
