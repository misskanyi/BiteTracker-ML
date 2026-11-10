# Deploying BiteTracker to AWS EC2

Nothing in this folder creates AWS resources, and the GitHub deploy job stays switched off until the repository variable `DEPLOY_ENABLED` is `true`. Do the first-time steps below once. After that, every push to `main` that passes the tests deploys itself.

## How it fits together

```
GitHub push to main
  └─ .github/workflows/deploy.yml
       ├─ test job (Python 3.12, like the server): Django checks + pytest
       │    (a synthetic model is trained into a temp dir for the tests, never shipped)
       └─ deploy job (only when DEPLOY_ENABLED=true, environment "production")
            ├─ git archive → code-only tarball → checked for models/data/DBs → s3://<bucket>/releases/<sha>.tar.gz
            └─ SSM Run Command on the instance: download the tarball, take deploy/scripts from it, run deploy.sh
                 deploy.sh: check artifacts → prune old releases → venv (python3) → backup
                 → check --deploy → collectstatic → migrate → switch current → restart → GET /health/
                 → automatic rollback (code, and the DB if migrations ran) if /health/ fails

On the instance
  /opt/bitetracker/releases/<sha>/     code + .venv (newest 3 kept)
  /opt/bitetracker/current  → active release
  /var/lib/bitetracker/                BITETRACKER_STATE_DIR (never replaced by deploys)
      db.sqlite3  models/  data/processed/  data/uploads/  reports/metrics/
  /etc/bitetracker/bitetracker.env     secrets and settings (root:bitetracker, 640)
  /var/backups/bitetracker/            backups (before every deploy and nightly at 02:15)
```

| File | Purpose |
|---|---|
| `scripts/bootstrap.sh` | One-time, re-runnable server setup: packages, user, directories, env file, systemd, Nginx, backup cron |
| `scripts/install-artifacts.sh` | Installs the production model, metadata, feature columns, processed CSV and metrics into the state dir (refuses synthetic models) |
| `scripts/deploy.sh` | Deploys one release with backup, health check and automatic rollback (run by GitHub Actions) |
| `scripts/manage.sh` | Runs a Django management command in the active release as the app user |
| `scripts/rollback.sh` | Manual rollback to an earlier release (optionally restoring the DB) |
| `scripts/backup.sh` | Online SQLite backup + tarball of the state dir, optional S3 copy |
| `gunicorn.conf.py` | Gunicorn on 127.0.0.1:8000, 2 workers, logs to journald |
| `systemd/bitetracker.service` | Runs Gunicorn as user `bitetracker`; only the state dir is writable |
| `nginx/bitetracker.conf` | Port 80 reverse proxy for any host name (Django checks the host), serves `/static/`, blocks `/media/` |
| `bitetracker.env.example` | Template that `bootstrap.sh` turns into `/etc/bitetracker/bitetracker.env` |

## Requirements

- **Instance:** Ubuntu 24.04 with its default Python 3.12. The pinned libraries all ship Python 3.12 wheels, and the saved model loads and predicts identically on 3.11 and 3.12. Pickled scikit-learn models depend on the scikit-learn and NumPy versions, which `requirements.txt` pins, not on the Python minor version.
- **Disk:** at least 20 GB. On x86_64, each release's virtualenv is about 1.3 GB, because xgboost pulls in NVIDIA NCCL. `deploy.sh` keeps 3 releases and stops with a clear message if less than 2.5 GB is free.
- **Network:** port 80 open in the security group (443 later for HTTPS). Port 22 is not needed; use Systems Manager.
- **Instance role:** `AmazonSSMManagedInstanceCore` and `s3:GetObject` on `arn:aws:s3:::<deploy-bucket>/releases/*`.
- **GitHub deploy role:** `s3:PutObject` on `arn:aws:s3:::<deploy-bucket>/releases/*`, `ssm:SendCommand` on the `AWS-RunShellScript` document and the instance, and `ssm:GetCommandInvocation` on `*`. Its OIDC trust policy must allow the subject `repo:<owner>/<repo>:environment:production`, because the deploy job runs in the `production` environment.
- **GitHub variables:** `AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`, `DEPLOY_BUCKET`, `EC2_INSTANCE_ID`, and `DEPLOY_ENABLED=true` last.

## First deployment

### 1. Bootstrap the server (once; safe to repeat)

Put a copy of `deploy/` where the instance can read it, then run `bootstrap.sh` as root through SSM Run Command. From the laptop, in the repository, on the commit you are deploying:

```bash
git archive --format=tar.gz -o /tmp/bitetracker-deploy-files.tar.gz HEAD deploy
aws s3 cp --sse AES256 /tmp/bitetracker-deploy-files.tar.gz s3://<deploy-bucket>/releases/bootstrap/deploy-files.tar.gz
```

The command to run on the instance:

```sh
set -eu
work=$(mktemp -d)
aws s3 cp --only-show-errors s3://<deploy-bucket>/releases/bootstrap/deploy-files.tar.gz "$work/d.tar.gz"
tar -xzf "$work/d.tar.gz" -C "$work"
bash "$work/deploy/scripts/bootstrap.sh"
rm -rf "$work"
```

`bootstrap.sh` creates `/etc/bitetracker/bitetracker.env` only if it does not exist. It generates a new secret key (never printed) and sets `DJANGO_ALLOWED_HOSTS` to the instance's public IP and DNS name. It also uses plain-HTTP settings (`DJANGO_SECURE_COOKIES=False`, `DJANGO_SECURE_SSL_REDIRECT=False`, `DJANGO_SECURE_HSTS_SECONDS=0`) until HTTPS is set up. Extra host names can be passed as arguments on the first run.

If the instance has no Elastic IP, its public IP changes when it is stopped and started. Then add the new address to `DJANGO_ALLOWED_HOSTS` and run `sudo systemctl restart bitetracker`.

### 2. Copy the production model and data (once, and whenever the model is retrained)

These files come from the machine that trained the model on the Food4Education data. They are git-ignored and never go through GitHub. Make the tarball on the laptop, in the repository. `COPYFILE_DISABLE=1` stops macOS from adding `._*` files.

```bash
COPYFILE_DISABLE=1 tar -czf /tmp/bitetracker-state.tar.gz \
  models/final_best_model.joblib models/final_best_model_metadata.json models/feature_columns.json \
  data/processed/school_meals_processed.csv \
  reports/metrics/naive_baseline_metrics.csv reports/metrics/validation_model_selection.csv
aws s3 cp --sse AES256 /tmp/bitetracker-state.tar.gz s3://<deploy-bucket>/releases/bootstrap/bitetracker-state.tar.gz
rm /tmp/bitetracker-state.tar.gz
```

Install on the instance (SSM Run Command), then delete the S3 copy:

```sh
set -eu
work=$(mktemp -d)
aws s3 cp --only-show-errors s3://<deploy-bucket>/releases/bootstrap/deploy-files.tar.gz "$work/d.tar.gz"
aws s3 cp --only-show-errors s3://<deploy-bucket>/releases/bootstrap/bitetracker-state.tar.gz "$work/state.tar.gz"
tar -xzf "$work/d.tar.gz" -C "$work"
bash "$work/deploy/scripts/install-artifacts.sh" "$work/state.tar.gz"
rm -rf "$work"
```

```bash
aws s3 rm s3://<deploy-bucket>/releases/bootstrap/bitetracker-state.tar.gz
```

`install-artifacts.sh` takes only the files listed above from the tarball. It refuses a model whose metadata says it was trained on synthetic data and backs up any existing model before replacing it. It never touches the database or uploads.

### 3. First deploy from GitHub Actions

Set `DEPLOY_ENABLED=true` (*Settings → Secrets and variables → Actions → Variables*), then *Actions → Deploy → Run workflow* on `main`. The first deploy creates the SQLite database in `/var/lib/bitetracker`.

### 4. Schools and the first admin account

Load the schools from the processed CSV (SSM Run Command; safe to repeat):

```sh
bash /opt/bitetracker/current/deploy/scripts/manage.sh import_schools
```

Create the admin account from an interactive shell (EC2 console → the instance → *Connect* → *Session Manager*). This keeps the password out of SSM command history:

```bash
sudo bash /opt/bitetracker/current/deploy/scripts/manage.sh createsuperuser
```

## HTTPS (when you have a domain)

1. Point the domain's DNS at the instance's Elastic IP.
2. Open port 443 in the security group.
3. On the instance, add the domain to `DJANGO_ALLOWED_HOSTS` and use `https://<domain>` in `DJANGO_CSRF_TRUSTED_ORIGINS` in `/etc/bitetracker/bitetracker.env`.
4. Set `server_name <domain>;` in `/etc/nginx/sites-available/bitetracker`.
5. Run:
   ```bash
   sudo apt install -y certbot python3-certbot-nginx && sudo certbot --nginx -d <domain>
   ```
6. Set `DJANGO_SECURE_COOKIES=True`, `DJANGO_SECURE_SSL_REDIRECT=True` and `DJANGO_SECURE_HSTS_SECONDS=3600`, then run `sudo systemctl restart bitetracker`.

`bootstrap.sh` never overwrites an existing Nginx site, so certbot's changes survive a rerun.

## Roles, Google sign-in and notifications

See [GOOGLE_SIGNIN_AND_ACCESS.md](GOOGLE_SIGNIN_AND_ACCESS.md) for the Google Cloud Console setup, the two
`GOOGLE_OAUTH_*` environment variables, the migrations, and the rollback order. Old code can't read the new
role names, so reverse the migrations before rolling back.

## Day-to-day

- **Logs:** `journalctl -u bitetracker -f`, `/var/log/nginx/bitetracker.*.log`, `/var/log/bitetracker-backup.log`
- **Health:** `curl http://<public-ip>/health/` returns `{"status": "ok", "checks": {...}}`. Each check is only `ok` or `fail`. In production, `real_data_model` fails if the saved model's metadata says it was trained on synthetic data.
- **Roll back:** `sudo bash /opt/bitetracker/current/deploy/scripts/rollback.sh` (previous release), or `rollback.sh <sha> --restore-db /var/backups/bitetracker/<backup>`
- **Replace the model:** repeat step 2. The app reloads the model when the files change. Deploys never touch it.
- **Backups:** `/var/backups/bitetracker/` on the same disk. For off-instance copies, set `BACKUP_S3_URI` and give the instance role `s3:PutObject` on that location, or enable EBS snapshots.
