# Deploying BiteTracker to AWS EC2

Nothing in this folder creates AWS resources, and the GitHub deploy workflow stays switched off until you turn it on. Follow the steps below once. After that, every push to `main` that passes the tests deploys itself.

## How it fits together

```
GitHub push to main
  └─ .github/workflows/deploy.yml
       ├─ test job: Django checks + pytest (synthetic model trained into a temp dir, never shipped)
       └─ deploy job (only when DEPLOY_ENABLED=true)
            ├─ git archive → code-only tarball → checked for models/data/DBs → private S3 bucket
            └─ SSM Run Command on the instance → deploy/scripts/deploy.sh
                 backup → venv → check --deploy → collectstatic → migrate
                 → switch /opt/bitetracker/current → restart → GET /health/ → auto-rollback on failure

On the instance
  /opt/bitetracker/releases/<sha>/     code + .venv (last 5 kept)
  /opt/bitetracker/current  → active release
  /var/lib/bitetracker/                BITETRACKER_STATE_DIR (never touched by deploys)
      db.sqlite3  models/  data/processed/  data/uploads/  reports/metrics/
  /etc/bitetracker/bitetracker.env     secrets and settings (root:bitetracker, 640)
  /var/backups/bitetracker/            backups from deploy/scripts/backup.sh
```

| File | Purpose |
|---|---|
| `gunicorn.conf.py` | Gunicorn on 127.0.0.1:8000, 2 workers, logs to journald |
| `systemd/bitetracker.service` | Runs Gunicorn as user `bitetracker`. The code is read-only to it, and only the state dir is writable |
| `nginx/bitetracker.conf` | Public reverse proxy, serves `/static/`, blocks `/media/` |
| `bitetracker.env.example` | Template for `/etc/bitetracker/bitetracker.env` |
| `scripts/deploy.sh` | Deploys one release with backup, health check and automatic rollback |
| `scripts/rollback.sh` | Manual rollback to an earlier release (optionally restoring the DB) |
| `scripts/backup.sh` | Online SQLite backup + tarball of the state dir, optional S3 copy |

## Manual AWS steps

1. **S3 bucket for releases** (private, block all public access, SSE-S3 encryption). Example: `bitetracker-deploy-<account-id>`. Optionally add a lifecycle rule that deletes `releases/` after 30 days. If you want off-instance backups, use a second private bucket (or a `backups/` prefix).

2. **IAM role for the EC2 instance** (instance profile):
   - AWS managed policy `AmazonSSMManagedInstanceCore`.
   - `s3:GetObject` on `arn:aws:s3:::<deploy-bucket>/releases/*`.
   - Optional for backups: `s3:PutObject` on `arn:aws:s3:::<backup-bucket>/*`.

3. **EC2 instance**: Ubuntu 24.04 LTS, t3.small or larger, 20 GB gp3 disk. Attach the role from step 2. Security group inbound rules: 80 and 443 from anywhere, **no port 22**. Use Session Manager instead of SSH. Attach an Elastic IP and point your domain's DNS at it.

4. **Prepare the instance** (Session Manager shell):
   ```bash
   sudo apt update && sudo apt install -y nginx python3-venv python3-pip curl unzip certbot python3-certbot-nginx
   # Python 3.11 matches the saved model's environment (optional but recommended):
   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install -y python3.11 python3.11-venv
   # AWS CLI v2 (needed by deploy and backup scripts):
   curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscli.zip && unzip -q /tmp/awscli.zip -d /tmp && sudo /tmp/aws/install
   sudo useradd --system --home /var/lib/bitetracker --shell /usr/sbin/nologin bitetracker
   sudo install -d -o bitetracker -g bitetracker -m 750 /var/lib/bitetracker
   sudo install -d -m 755 /opt/bitetracker/releases
   sudo install -d -o root -g bitetracker -m 750 /etc/bitetracker
   sudo install -d -m 700 /var/backups/bitetracker
   ```

5. **Copy the deployment files from the repo** (for example, upload this folder via S3, or paste them in) and install them:
   ```bash
   sudo cp deploy/systemd/bitetracker.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable bitetracker
   sudo cp deploy/nginx/bitetracker.conf /etc/nginx/sites-available/bitetracker   # edit server_name first
   sudo ln -s /etc/nginx/sites-available/bitetracker /etc/nginx/sites-enabled/ && sudo rm -f /etc/nginx/sites-enabled/default
   sudo nginx -t && sudo systemctl reload nginx
   sudo install -o root -g bitetracker -m 640 deploy/bitetracker.env.example /etc/bitetracker/bitetracker.env
   sudo nano /etc/bitetracker/bitetracker.env     # set DJANGO_SECRET_KEY, hosts, etc.
   ```
   Until HTTPS works, set `DJANGO_SECURE_SSL_REDIRECT=False` and `DJANGO_SECURE_COOKIES=False`. Otherwise logins fail and pages redirect to an https:// address that does not exist yet.

6. **Copy the existing trained model and data into the state dir.** They never come from GitHub or CI. From your laptop, upload them to a private S3 location, then on the instance:
   ```bash
   # Needed: models/final_best_model.joblib, models/final_best_model_metadata.json,
   #         models/feature_columns.json, data/processed/school_meals_processed.csv, reports/metrics/*.csv
   # Optional: db.sqlite3 (your existing users/schools) and data/uploads/
   sudo aws s3 cp --recursive s3://<private-bucket>/initial-state/ /var/lib/bitetracker/
   sudo chown -R bitetracker:bitetracker /var/lib/bitetracker && sudo chmod -R o-rwx /var/lib/bitetracker
   ```
   Delete that S3 copy afterwards. If you start without a database, create an admin after the first deploy:
   `cd /opt/bitetracker/current && sudo -u bitetracker env $(sudo cat /etc/bitetracker/bitetracker.env | grep -v '^#' | xargs) .venv/bin/python manage.py createsuperuser`

7. **GitHub OIDC**: in IAM, add identity provider `token.actions.githubusercontent.com` (audience `sts.amazonaws.com`). Create role `bitetracker-github-deploy` with this trust policy (replace `OWNER/REPO`):
   ```json
   {"Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {"StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                                   "token.actions.githubusercontent.com:sub": "repo:OWNER/REPO:environment:production"}}}
   ```
   Permissions: `s3:PutObject` on `arn:aws:s3:::<deploy-bucket>/releases/*`; `ssm:SendCommand` on `arn:aws:ssm:<region>::document/AWS-RunShellScript` and on `arn:aws:ec2:<region>:<account-id>:instance/<instance-id>`; `ssm:GetCommandInvocation` and `ssm:ListCommandInvocations` on `*`.

8. **GitHub settings**: create environment `production` (optionally with required reviewers and a "main only" branch rule). Under *Settings → Secrets and variables → Actions → Variables*, add `AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`, `DEPLOY_BUCKET` and `EC2_INSTANCE_ID`. Add `DEPLOY_ENABLED=true` **last**, when you are ready. No AWS secrets are stored in GitHub.

9. **First deploy**: run the *Deploy* workflow manually (Actions → Deploy → Run workflow on `main`), then check `https://<your-domain>/health/`.

10. **HTTPS**: `sudo certbot --nginx -d <your-domain>`. Then set `DJANGO_SECURE_SSL_REDIRECT=True` and `DJANGO_SECURE_COOKIES=True` and run `sudo systemctl restart bitetracker`.

11. **Nightly backups**:
    ```bash
    echo '15 2 * * * root /opt/bitetracker/current/deploy/scripts/backup.sh nightly >>/var/log/bitetracker-backup.log 2>&1' | sudo tee /etc/cron.d/bitetracker-backup
    ```
    Also consider enabling AWS Backup or EBS snapshots for the instance volume.

## Day-to-day

- Logs: `journalctl -u bitetracker -f`, `/var/log/nginx/bitetracker.*.log`
- Health: `curl -H "Host: <your-domain>" http://127.0.0.1:8000/health/` returns `{"status": "ok", "checks": {...}}`. Each check is `ok` or `fail` only. In production `real_data_model` fails if the saved model's metadata says it was trained on synthetic data.
- Roll back: `sudo /opt/bitetracker/current/deploy/scripts/rollback.sh` (previous release) or `rollback.sh <sha> --restore-db /var/backups/bitetracker/<backup>`
- Replacing the model: copy the new `models/` files into `/var/lib/bitetracker/models/` (after a backup). The app reloads it automatically when the files change. Deploys never do this.
