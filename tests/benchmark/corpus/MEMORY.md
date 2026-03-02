# Project Memory

## Redis Configuration

We use Redis as our primary cache layer. The maxmemory-policy is set to allkeys-lru,
which evicts the least recently used keys when memory is full. The max memory is 2GB
in production and 512MB in staging.

Connection pool settings:
- max_connections: 50
- timeout: 5 seconds
- retry_on_timeout: true
- socket_keepalive: true

## Authentication System

Authentication uses JWT tokens with RS256 signing. Access tokens expire after 15
minutes. Refresh tokens expire after 7 days. Token rotation is enabled — each
refresh request invalidates the old refresh token and issues a new pair.

The auth middleware validates tokens on every request. Invalid tokens return 401.
Expired tokens return 401 with a `token_expired` error code so the client knows
to attempt a refresh.

## Database Schema

PostgreSQL 15 is the primary database. We use connection pooling via PgBouncer
with a pool size of 25. The schema uses UUID primary keys generated client-side
to avoid round-trips.

Key tables:
- users: id, email, name, created_at, updated_at
- projects: id, owner_id, name, description, created_at
- tasks: id, project_id, assignee_id, title, status, priority, due_date

## API Rate Limiting

Rate limiting is implemented at the API gateway level using a token bucket
algorithm. Default limits:
- Anonymous: 60 requests/minute
- Authenticated: 300 requests/minute
- Admin: 1000 requests/minute

Rate limit headers are included in every response:
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset

## Logging and Observability

We use structured JSON logging with correlation IDs. Every request gets a unique
trace_id that propagates through all downstream service calls. Logs are shipped
to Elasticsearch via Filebeat.

Key log fields:
- trace_id: UUID correlation ID
- service: originating service name
- level: debug/info/warn/error
- timestamp: ISO 8601 format
- duration_ms: request duration

Prometheus metrics are exposed on /metrics endpoint. Grafana dashboards cover:
- Request latency (p50, p95, p99)
- Error rates by status code
- Cache hit ratio
- Database connection pool utilization

## Deployment Pipeline

CI/CD is handled by GitHub Actions. The pipeline stages are:
1. Lint and type check
2. Unit tests
3. Integration tests
4. Build Docker image
5. Push to ECR
6. Deploy to staging (automatic)
7. Deploy to production (manual approval required)

Blue-green deployments are used in production. The load balancer switches traffic
after health checks pass on the new deployment. Rollback is automatic if the
error rate exceeds 5% in the first 10 minutes.

## Search Infrastructure

Full-text search is powered by Elasticsearch 8. Documents are indexed with
custom analyzers for multilingual support. The search pipeline includes:
- Query expansion with synonyms
- Fuzzy matching (edit distance 2)
- Field boosting (title: 3x, description: 1x, tags: 2x)
- Highlighting with <mark> tags

Reindexing runs nightly via a scheduled job. Zero-downtime reindexing uses
the alias swap pattern: write to a new index, then atomically switch the alias.

## Error Handling Strategy

All API errors follow RFC 7807 (Problem Details for HTTP APIs). Error responses
include:
- type: URI identifying the error type
- title: human-readable summary
- status: HTTP status code
- detail: detailed explanation
- instance: URI of the specific occurrence

Custom error codes are documented in the error catalog. Client SDKs map
error codes to localized messages.

## Performance Optimization Notes

Database query optimization findings:
- Added composite index on tasks(project_id, status) — reduced query time from 450ms to 12ms
- Enabled prepared statements for frequently executed queries
- Added read replicas for reporting queries

Frontend optimizations:
- Code splitting reduced initial bundle from 2.1MB to 380KB
- Image lazy loading with intersection observer
- Service worker caching for static assets
