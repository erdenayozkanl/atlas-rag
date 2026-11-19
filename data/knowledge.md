# NovaDesk Developer Platform

## Authentication
NovaDesk API requests use a bearer token in the Authorization header. Never put the API key in a URL. An API key can be rotated in the settings page.

## Rate limits
The NovaDesk API allows 120 requests per minute per API key. When a request receives HTTP 429, retry with exponential backoff and jitter. Do not retry forever.

## Webhooks
Webhook deliveries are signed using HMAC-SHA256. Verify the signature against the raw request body before processing an event. Events may be delivered more than once, so handlers must be idempotent.

## Billing
The Starter plan includes 10,000 events each month. The Pro plan includes 100,000 events each month. Overage is billed separately.

## Security
Do not log tokens, payment identifiers, or personal information. Rotate compromised tokens immediately and restrict permissions using least privilege.
