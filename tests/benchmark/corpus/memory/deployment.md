# Deployment Procedures

## Staging Deployment

Staging deployments are automatic on every merge to main. The process:
1. GitHub Actions builds the Docker image
2. Image is pushed to ECR with the commit SHA tag
3. ArgoCD detects the new image and rolls out
4. Smoke tests run automatically
5. Slack notification is sent to #deployments

## Production Deployment

Production requires manual approval from a tech lead. Steps:
1. Create a release tag (semantic versioning)
2. Approve the deployment in GitHub Actions
3. Blue-green swap begins
4. Health checks run for 5 minutes
5. If healthy, old deployment is terminated
6. If unhealthy, automatic rollback triggers

## Rollback Procedure

To rollback production:
1. Navigate to ArgoCD dashboard
2. Select the previous healthy revision
3. Click "Rollback"
4. Verify health checks pass
5. Post-mortem within 24 hours

## Environment Variables

Critical environment variables for deployment:
- DATABASE_URL: PostgreSQL connection string
- REDIS_URL: Redis connection string
- JWT_SECRET: RS256 private key path
- API_RATE_LIMIT: requests per minute
- LOG_LEVEL: debug/info/warn/error
