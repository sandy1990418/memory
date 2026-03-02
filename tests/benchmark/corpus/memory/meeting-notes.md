# Meeting Notes

## 2026-01-20: Architecture Review

Attendees: Sarah, Marcus, Yuki, Alex

Discussion points:
- Microservices boundary review: agreed to merge user-service and auth-service
- API gateway migration from Kong to Envoy planned for Q2
- Database sharding strategy: range-based sharding by tenant_id
- Monitoring: adding distributed tracing with Jaeger

Action items:
- Alex: POC for Envoy migration by Feb 1
- Sarah: Document merged service API contract
- Marcus: Benchmark sharding performance

## 2026-02-05: Sprint Retrospective

What went well:
- Deployment automation saved 4 hours per release
- New testing framework reduced flaky tests by 60%
- Cross-team code review improved code quality

What to improve:
- Documentation lagging behind code changes
- Need better error messages for API consumers
- Load testing should be part of CI pipeline
