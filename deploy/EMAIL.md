# Account emails

BiteTracker emails a small set of account events. Everything else (forecast ready, uploads,
calendar, announcements, support) is shown in-app only.

| Event | Recipient | Can be switched off? |
|---|---|---|
| Registration received, awaiting approval | Applicant | No (essential) |
| New registration request | Each active System Administrator, one message each | Yes: Settings → Notifications |
| Registration approved / not approved | Applicant | No |
| Account suspended / reactivated | The account holder | No |
| Role or school assignments changed | The account holder | No |

Emails include the status, role and assigned schools where relevant, and a sign-in or review link.
They never include passwords, tokens, internal decision notes or school data.

**Branding:** the HTML part's header shows the BiteTracker icon, a PNG attached to the email itself
(`static/images/branding/email/logo-icon-96.png`, Content-ID `bitetracker-logo`, alt text
"BiteTracker logo"), next to the brand name as real text. No remote or localhost image URL is used,
and the header still reads "BiteTracker" when images are blocked. This branding is only in the email
body. It does not change the sender avatar Gmail shows; that would need BIMI, which isn't set up.

**Gmail's "Unsubscribe" link** comes from a `List-Unsubscribe` header that Brevo adds to every
email it relays, including transactional ones. BiteTracker doesn't set it. Leave it in place; Brevo
only offers removing it (in exchange for a `List-Help` header) on Enterprise plans, through Brevo
support. If a recipient uses it, Brevo may block later account emails to that address, even though
the outbox shows them as `sent`. Check Brevo's transactional logs and blocked contacts when someone
reports missing emails, and unblock an address only at that person's request.

## How delivery works

1. The account action records one `EmailDelivery` row per recipient (the outbox) **in the same
   database transaction**. A rollback removes the rows, so nothing is sent for changes that didn't
   happen.
2. Right after the commit, the same web request sends every email that transaction queued, as one
   batch over one SMTP connection. There is no background worker.
3. If sending fails, the account action still stands. The row is marked `failed`, and the log gets
   one line with only the delivery id, the kind of email and the exception class. Once a connection
   attempt has failed, the rest of the batch is marked `failed` without waiting on the network again.
4. `send_pending_emails` sends rows left `pending` (time budget used up) and retries `failed` rows,
   up to `EMAIL_MAX_ATTEMPTS` (5) times and only while they are less than
   `EMAIL_RETRY_MAX_AGE_HOURS` (48) old. Run it on a schedule (see below). Without a schedule,
   failed and deferred emails are not sent.
5. **No duplicates per event:** each (recipient, event) pair can only be stored once. This covers
   repeat approvals, page refreshes, repeat Google sign-ins and resubmitted suspend forms. Old
   accounts and requests are never emailed after the fact.
6. **No duplicates between senders:** a row is claimed with one conditional database `UPDATE`
   (`pending`/`failed` → `sending`) immediately before it is sent. If the web request and the retry
   command, or two retry runs, reach the same row, exactly one claim succeeds and the other skips
   the row. A claim left in `sending` for more than 15 minutes is treated as abandoned (the process
   died) and can be retried. The one remaining duplicate case: the process dies after the provider
   accepted the message but before the row was marked `sent`.
7. **Delivery mode is stamped on each row** (`channel`: `console`, `smtp` or `disabled`). A row is
   only ever sent under the mode it was queued in. Emails recorded while printing to the console
   are never sent later through SMTP, and switching modes never sends a backlog.
8. **Recipient allow-list:** when `EMAIL_ALLOWED_RECIPIENTS` is set, only those exact addresses
   are emailed. Every other email is stored as `suppressed` and is never sent, even after the list
   is cleared.

### How long a request can wait

Sending happens inside the request that caused the event: the Google sign-in callback for a new
registration, or the administrator's approve, suspend or edit request. `EMAIL_TIMEOUT` (T, default
5 s) limits each blocking network operation, not the whole send. The request stops *starting* new
emails after `EMAIL_REQUEST_BUDGET_SECONDS` (B, default 10 s), but it can't interrupt an
operation already in progress. Roughly, the request waits at most **max(B, C) + M**, where:

- C, connection setup: a TCP connect of up to T for each IP address the host resolves to (A), plus
  about 5 more round trips (greeting, EHLO, STARTTLS/TLS, EHLO, AUTH), each up to T. That is about
  (A + 5) × T.
- M, one message: about 4 round trips (MAIL, RCPT, DATA, end of data), each up to T. That is about
  4 × T.

| Situation | Typical wait (defaults) |
|---|---|
| Healthy provider: applicant plus 3 administrators (4 emails, 1 connection) | about 1–3 s |
| Port blocked or host unreachable (silently dropped) | A × T: 5 s per resolved address (provider hosts often resolve to 2–8 addresses), then the batch is marked failed |
| Connection refused | under 1 s |
| Server accepts, then stalls on every step (worst case) | about max(10, 30) + 20 = 50 s with A = 1 |

DNS lookup time is not covered by `EMAIL_TIMEOUT`. A server that keeps trickling bytes could in
theory stretch an operation further. Both are unlikely with a reputable provider.

Lower T or B to shorten the worst case. For example, `EMAIL_TIMEOUT=3` with
`EMAIL_REQUEST_BUDGET_SECONDS=5` gives roughly max(5, 18) + 12 = 30 s. The only true hard cap
would be sending outside the request (a queue worker), which is not implemented. Gunicorn's
120-second worker timeout is well above these figures.

## Configuration

| `BITETRACKER_EMAIL_DELIVERY` | Behaviour |
|---|---|
| `console` (default with `DJANGO_DEBUG=True`) | Prints emails to the terminal. Nothing is sent. |
| `disabled` (default with `DJANGO_DEBUG=False`) | Records rows as `disabled` and never sends them, even if delivery is enabled later. |
| `smtp` | Sends through `EMAIL_HOST`. Startup fails if `EMAIL_HOST` or `DEFAULT_FROM_EMAIL` is missing, if both TLS modes are set, or if `EMAIL_HOST_PASSWORD` equals the Google sign-in client secret. |

You can use any SMTP provider. Recipients can use Gmail or any other mailbox, and BiteTracker
never needs access to their inboxes. Before going live you need:

- a **sender address** on a domain you control, such as `no-reply@<your domain>`, set as
  `DEFAULT_FROM_EMAIL=BiteTracker <no-reply@<your domain>>`
- an **SMTP provider** account with that domain verified, and its SPF and DKIM DNS records
  published (add DMARC as well). Examples: Amazon SES (good fit on EC2), Brevo, Mailgun, Postmark,
  or the Google Workspace SMTP relay
- a **dedicated SMTP credential** (`EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_HOST_USER`,
  `EMAIL_HOST_PASSWORD`). Never use the Google sign-in client secret or a personal mailbox password.
- `BITETRACKER_SITE_URL=https://<your domain>` for the links in emails

Put these values only in `/etc/bitetracker/bitetracker.env`, never in Git.

## Retry schedule (optional, recommended)

Example root crontab entry (every 10 minutes):

```
*/10 * * * * bash /opt/bitetracker/current/deploy/scripts/manage.sh send_pending_emails >/dev/null 2>&1
```

The command prints counts only, never addresses. Add `--dry-run` to see how many emails are due.

## Controlled local SMTP test (one address you own)

All values go in the git-ignored `.env` in the project root. `.env` is matched by `*.env` in
`.gitignore`. Settings load it from the project folder whatever the current directory is, and real
environment variables take precedence over it. Quote the sender value if your editor complains:
`DEFAULT_FROM_EMAIL="BiteTracker <no-reply@your-domain>"`.

1. Apply the new tables to your local database: `python manage.py migrate`.
2. Add these lines to `.env`. Use a dedicated SMTP credential from your provider, never the Google
   sign-in client secret.
   ```
   BITETRACKER_EMAIL_DELIVERY=smtp
   EMAIL_ALLOWED_RECIPIENTS=you@example.com
   BITETRACKER_SITE_URL=http://localhost:8000
   DEFAULT_FROM_EMAIL=BiteTracker <no-reply@your-verified-domain>
   EMAIL_HOST=<provider SMTP host>
   EMAIL_PORT=587
   EMAIL_HOST_USER=<SMTP username>
   EMAIL_HOST_PASSWORD=<SMTP password>
   EMAIL_USE_TLS=True
   ```
   With the allow-list set, BiteTracker can only email `you@example.com`, even if the local database
   holds other people's real addresses. Their emails are stored as `suppressed` and never sent.
   (With Amazon SES in sandbox mode, the recipient must also be verified in SES, which adds a
   second safeguard.)
3. Check the configuration: `python manage.py check`. A missing host or sender, both TLS modes set,
   or reuse of the Google secret stops here with a clear message.
4. Test the connection with Django's built-in command, which sends one message to the address you
   name and touches no account: `python manage.py sendtestemail you@example.com`.
5. Check that nothing old is waiting: `python manage.py send_pending_emails --dry-run` should
   report `0 email(s) due`. Emails recorded in console mode can't be sent in smtp mode.
6. Run real events against an account that uses your address. As a local System Administrator:
   - Create a user with email `you@example.com` (creating a user sends no email).
   - Suspend it and expect one email. Submit the suspend form again and expect none.
   - Reactivate it and expect one email.
   - Change its role and schools and expect one "access has changed" email.
   - Optionally, if Google sign-in works locally, register with that Google address and expect a
     "registration received" email. The administrators' copies are suppressed unless their
     addresses are on the allow-list.
7. Inspect the results by status only (no addresses printed):
   `python manage.py shell -c "from web.notifications.models import EmailDelivery as E; from django.db.models import Count; print(list(E.objects.values_list('status').annotate(Count('pk'))))"`
8. Finish by setting `BITETRACKER_EMAIL_DELIVERY=console` again and removing the SMTP password
   from `.env` if you don't need it.

## Controlled first real-email test (production)

1. Keep `EMAIL_ALLOWED_RECIPIENTS` set to your own address(es) for the first run. While the
   provider is in sandbox mode (Amazon SES starts there), it also only delivers to verified
   addresses.
2. On the server, set `BITETRACKER_EMAIL_DELIVERY=smtp` plus the SMTP values, then restart:
   `systemctl restart bitetracker`. If the configuration is invalid, startup fails with a clear message.
3. Test the connection without touching any account:
   `deploy/scripts/manage.sh sendtestemail <your own address>`
4. Run the account events above with test identities you own. Expect exactly one email per event.
5. Check deliverability: the message should not land in spam, and SPF/DKIM should pass
   (Gmail: "Show original"). Check `journalctl -u bitetracker` for `Account email failed` lines.
6. Clear `EMAIL_ALLOWED_RECIPIENTS` to go live, and remove the test account.
