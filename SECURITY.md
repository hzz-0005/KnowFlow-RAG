# Security

This is a single-host research application. Compose publishes services on
127.0.0.1. A remote deployment needs a TLS reverse proxy, network access policy,
separate database credentials, resource limits and an operator-managed backup.

Create `.env` with `python scripts/init_env.py`; it generates a unique JWT secret
and refuses to replace an existing file. Never commit keys, uploads, database
dumps, conversation exports or evaluation results. `APP_ENV=production` rejects
the example JWT secret and secrets shorter than 32 characters. Development mode
exists for local tests, not permission to expose the example configuration.

All document, conversation, task and memory APIs check ownership. Documents and
retrieved passages are untrusted inputs. Markdown/diagrams are sanitized; model
output is not executed as Python/SQL/shell. Formula caches remain unverified.
Model requests can transmit selected document text to configured providers; use
providers and data you are authorized to process. Offline test stubs do not
transmit data to external models.

Do not disclose credentials or private documents in public issues. Report a
vulnerability through the repository's private vulnerability-reporting channel
when enabled; otherwise contact the repository owner privately. No contact
address is invented here. Permission checks and scanners reduce specific risks;
they are not an independent security audit or a multi-tenant production claim.
