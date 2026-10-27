# Synthetic Atlas gateway policy

This document contains fictional rules written solely for regression tests.

## Production policy 2025

For the Atlas gateway production policy version 2025, the timeout threshold is 30 ms.

## Production policy 2026

For the Atlas gateway production policy version 2026, the timeout threshold is 45 ms.
In production, the Atlas gateway must not write request bodies to Redis.

## Staging policy 2026

In staging, the Atlas gateway may write synthetic request bodies to Redis for debugging.
This staging permission does not apply to production.
