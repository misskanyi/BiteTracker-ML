# Releasing roles, Google sign-in and live notifications

This runbook covers the `feature/rbac` release. It applies to the EC2 deployment described in
[README.md](README.md): Cloudflare → Nginx → Gunicorn → Django, with SQLite under `/var/lib/bitetracker`.

> **Merging into `main` deploys.** `.github/workflows/deploy.yml` deploys every push to `main`.
> Finish steps 1–3 before you merge.

## What changes for existing users

| Before | After | Effect |
|---|---|---|
| School Operator | **School Manager** | Same schools. They can now upload data for their schools. Before, uploads needed a Django permission that nobody had. |
| Programme Admin | **Resource Planning Stakeholder** | Same schools, but **read-only**. They lose user management, upload review and the audit log. |
| System Admin | **System Administrator** | Unchanged. The existing administrator account is kept. |

Other access changes:
- **Model performance** (`/performance/`) is for System Administrators only.
- Upload records and files are visible only to the uploader and to System Administrators.
- Migration `accounts.0003` renames the stored role values. No accounts, schools, forecasts,
  uploads or model files are changed.

If someone who was a Programme Admin still needs to manage users, make them a System
Administrator in *Administration → User management* after the release.

## 1. Google Cloud Console (one-time)

1. Go to <https://console.cloud.google.com/> and create or select a project, e.g. "BiteTracker".
2. Open **APIs & Services → OAuth consent screen**:
   - **User type:** External. Choose Internal only if every user has an account in your own Google Workspace.
   - **App name:** BiteTracker. **Support email:** your address. **App domain / homepage:** `https://bitetracker.xyz`.
   - **Authorised domains:** `bitetracker.xyz`
   - **Scopes:** `openid`, `.../auth/userinfo.email` and `.../auth/userinfo.profile` only. These are non-sensitive, so Google doesn't need to verify the app.
   - **Publishing status:** while the app is set to "Testing", only listed test users can sign in. Select **Publish app** when you're ready.
3. Open **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   - **Application type:** Web application. **Name:** BiteTracker production.
   - **Authorised JavaScript origins:** `https://bitetracker.xyz`
   - **Authorised redirect URIs:** exactly `https://bitetracker.xyz/auth/google/login/callback/`, including `https` and the trailing slash.
   - If you also serve `www.bitetracker.xyz`, add `https://www.bitetracker.xyz/auth/google/login/callback/` as a second redirect URI. That host must also be in `DJANGO_ALLOWED_HOSTS`.
4. Copy the **Client ID** and **Client secret**. Don't paste them into Git, chat, tickets or shell history.

For local testing, create a separate client with the redirect URI
`http://localhost:8000/auth/google/login/callback/`, and put its values in your untracked `.env`.

## 2. Server environment (one-time)

Add two lines to `/etc/bitetracker/bitetracker.env`. The file is `root:bitetracker`, mode 640.
Edit it with `sudoedit` so the secret never appears in your shell history:

```
sudo sudoedit /etc/bitetracker/bitetracker.env
```

```
GOOGLE_OAUTH_CLIENT_ID=<client id>.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=<client secret>
```

You don't need to change anything else:
- `DJANGO_ALLOWED_HOSTS` must already include `bitetracker.xyz`.
- `DJANGO_CSRF_TRUSTED_ORIGINS` must already include `https://bitetracker.xyz`.
- When `DJANGO_DEBUG=False`, the redirect URI is always built with `https://`, even if Cloudflare
  reaches Nginx over plain HTTP. `DJANGO_OAUTH_PROTOCOL` overrides this; don't set it in production.

If both values are blank, the release still works: the Google button is hidden, the `/auth/google/…`
routes return 404, and password login works as before.

## 3. Cloudflare, Nginx, Gunicorn

- **Cloudflare:** no change is required.
  - Leave the `/auth/` and `/notifications/feed/` paths uncached. Cloudflare's default does not cache HTML or JSON, and the feed also sends `Cache-Control: no-store, private`.
  - If you have a "Cache Everything" page rule, add a bypass for `/auth/*` and `/notifications/*`.
  - Bot Fight Mode and WAF rules must not challenge `GET /auth/google/login/callback/`, because Google's redirect lands there.
- **Nginx:** no change. It already forwards `Host` and `X-Forwarded-Proto`. The callback URL
  (`?code=…&state=…`) appears in access logs. The code is single-use, expires within minutes, and
  is useless without the PKCE verifier held in the server session, so it is not a credential.
- **Gunicorn:** no change. Each open tab polls `/notifications/feed/` every 20 seconds, which is a few
  small reads. The current 2 workers × 2 threads handle that comfortably. If long 30-day forecasts
  make the site feel slow while many people are online, raise `GUNICORN_WORKERS` to 3.
- **New Python dependency:** `django-allauth[socialaccount]==65.19.7`, which brings in `requests`,
  `PyJWT` and `cryptography`. `deploy.sh` installs it from `requirements.txt`.

## 4. Deploy

Merge the pull request into `main`. The deploy workflow runs the tests. `deploy.sh` then backs up
the state directory, runs `collectstatic`, runs `migrate`, switches the release and checks `/health/`.

To apply the migrations by hand instead, run them as root on the instance:

```
sudo bash /opt/bitetracker/current/deploy/scripts/backup.sh pre-rbac
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh migrate --plan     # review
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh migrate --noinput
```

These migrations are applied:
- `accounts.0003_project_roles_and_access_requests`: role rename plus the AccessRequest table.
- `audit.0003`, `notifications.0003` (adds `event_key` and its unique constraint) and `schools.0003`
  (adds the "Clarification requested" status).
- allauth's own `account` and `socialaccount` tables.

## 5. Check after deploying

1. `curl -s https://bitetracker.xyz/health/` returns `"status": "ok"`.
2. Sign in with the existing administrator's password. *Administration → Access requests* is the first tab.
3. In a private window, choose **Continue with Google** with a test Google account. You should see
   "Your registration request has been sent to the administrator…". The administrator gets a toast
   within about 20 seconds, and the bell count goes up.
4. Approve the request as a School Manager for one school. In the private window, choose
   **Continue with Google** again: you land on the dashboard and see only that school.
5. Suspend the test account in *User management*. Its open session stops working on the next request.
6. Check `journalctl -u bitetracker --since "10 min ago"` for errors. Google failures are logged
   as the error type only, never with codes or tokens.

## Rollback

> **Warning:** the older code can't read the new role names. A code-only `rollback.sh` would break
> pages for every user who isn't a superuser. Reverse the migrations first.

**A. Keep the data written since the release (recommended).** Run this with the new release still
active, because it contains the reverse migrations:

```
sudo bash /opt/bitetracker/current/deploy/scripts/backup.sh pre-rbac-rollback
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh migrate notifications 0002
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh migrate schools 0002
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh migrate audit 0002
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh migrate accounts 0002
sudo bash /opt/bitetracker/current/deploy/scripts/rollback.sh
```

- Roles go back to School Operator, Programme Admin and System Admin.
- The access-request table is dropped. The audit log keeps the "Registration …" entries.
- People who registered with Google stay **inactive**, so they remain blocked. Delete those accounts in Django Admin if you don't want them.
- The allauth tables remain and are unused by the old code.

**B. Restore the pre-deploy database.** This loses everything written since the deploy:

```
sudo bash /opt/bitetracker/current/deploy/scripts/rollback.sh <previous-release-id> --restore-db /var/backups/bitetracker/<pre-deploy-backup>
```

To turn off only Google sign-in, blank the two `GOOGLE_OAUTH_*` lines and run
`sudo systemctl restart bitetracker`. Roles and notifications keep working.

## SQLite risks

- **One writer at a time.** Approvals, uploads and notifications are short transactions, and the
  feed only reads. Two simultaneous writes can still return "database is locked" after Django's
  5-second wait. This is unlikely at programme scale, but move to PostgreSQL if many people will
  write at once. Set `DJANGO_DB_ENGINE=postgresql`; the settings are already in place.
- **Weaker concurrency guarantees.** Approval uses a conditional `UPDATE … WHERE status='pending'`,
  so a double-click or two administrators can't decide the same request twice. A unique
  `(recipient, event_key)` constraint backs the "notify once" rule. Both hold on SQLite.
- **Single file on one disk.** The database lives on the instance's EBS volume. Keep `BACKUP_S3_URI`
  or EBS snapshots enabled. Accounts, approvals and audit records are now business records too.
- **Migrations lock the file** for a moment. `deploy.sh` runs them during the switch, so a few requests can wait.
