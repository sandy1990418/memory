# Active Projects

## Project Alpha: User Dashboard Redesign

Status: In Progress (70% complete)
Lead: Sarah Chen
Target: Q1 2026

The dashboard redesign focuses on:
- Responsive layout with CSS Grid
- Dark mode support using CSS custom properties
- Widget-based customizable layout
- Performance: lazy-loaded widgets with skeleton screens

Tech stack changes:
- Migrating from Redux to Zustand for state management
- Replacing moment.js with date-fns (90% bundle size reduction)
- Adding React Query for server state

## Project Beta: Notification System

Status: Planning
Lead: Marcus Johnson
Target: Q2 2026

Multi-channel notification system supporting:
- In-app notifications (WebSocket push)
- Email notifications (SendGrid integration)
- SMS notifications (Twilio integration)
- Webhook notifications (custom endpoints)

Each notification type has configurable:
- Delivery schedule (immediate, batched hourly, daily digest)
- Priority levels (critical, high, normal, low)
- User preference overrides

## Project Gamma: Data Analytics Pipeline

Status: Completed
Lead: Yuki Tanaka

Built a real-time analytics pipeline using:
- Apache Kafka for event streaming
- Apache Flink for stream processing
- ClickHouse for analytics storage
- Grafana for visualization

Pipeline processes 50K events/second with < 2 second end-to-end latency.
