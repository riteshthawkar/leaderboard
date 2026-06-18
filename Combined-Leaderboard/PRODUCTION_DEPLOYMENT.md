# Production Deployment Guide

## Overview

This guide covers deploying the Vision Leaderboard system to production using Gunicorn, PostgreSQL, and systemd.

## System Requirements

- Ubuntu 20.04 LTS (or equivalent)
- Python 3.8+
- PostgreSQL 12+
- 2+ GB RAM
- 10+ GB storage (for datasets and submissions)

## Pre-Deployment Checklist

- [ ] Test all fixes locally
- [ ] Review and update `.env.example` → `.env`
- [ ] Generate secure API tokens
- [ ] Set up PostgreSQL database
- [ ] Configure HTTPS/SSL certificates
- [ ] Set up monitoring and alerting
- [ ] Create database backups
- [ ] Plan rollback strategy

## Step 1: Install Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
sudo apt-get install -y postgresql postgresql-contrib
sudo apt-get install -y nginx supervisor
sudo apt-get install -y git wget curl

# Install libmagic for file type detection
sudo apt-get install -y libmagic1
```

## Step 2: Setup Application Directory

```bash
# Create application directory
sudo mkdir -p /opt/leaderboard
cd /opt/leaderboard

# Clone or copy the project
git clone <repository-url> .
# or
sudo cp -r /path/to/Combined-Leaderboard/* /opt/leaderboard/

# Set proper ownership
sudo chown -R www-data:www-data /opt/leaderboard
```

## Step 3: Create Virtual Environment

```bash
cd /opt/leaderboard

# Create venv
sudo -u www-data python3.9 -m venv venv

# Activate and upgrade pip
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt
pip install gunicorn psycopg2-binary
```

## Step 4: Setup PostgreSQL Database

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# In PostgreSQL:
CREATE DATABASE leaderboard_db;
CREATE USER leaderboard_user WITH PASSWORD 'your-secure-password';
ALTER ROLE leaderboard_user SET client_encoding TO 'utf8';
ALTER ROLE leaderboard_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE leaderboard_user SET default_transaction_deferrable TO on;
ALTER ROLE leaderboard_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE leaderboard_db TO leaderboard_user;
\q
```

## Step 5: Configure Environment

```bash
cd /opt/leaderboard

# Create .env file
sudo -u www-data cp .env.example .env

# Edit configuration
sudo -u www-data nano .env

# Important settings:
# DATABASE_URL=postgresql://leaderboard_user:your-secure-password@localhost/leaderboard_db
# FLASK_ENV=production
# LEADERBOARD_LOG_LEVEL=INFO
# API_TOKENS=<generate-with-: python -c "import secrets; print(secrets.token_hex(32))">
```

## Step 6: Create Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/leaderboard.service > /dev/null << 'EOF'
[Unit]
Description=Vision Leaderboard API Service
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/leaderboard

# Load environment from .env
EnvironmentFile=/opt/leaderboard/.env

# Run the application
ExecStart=/opt/leaderboard/venv/bin/gunicorn \
    --workers 4 \
    --worker-class sync \
    --bind unix:/opt/leaderboard/leaderboard.sock \
    --timeout 120 \
    --access-logfile /opt/leaderboard/logs/access.log \
    --error-logfile /opt/leaderboard/logs/error.log \
    --log-level info \
    backend.web:app

# Restart on failure
Restart=on-failure
RestartSec=10

# Security settings
PrivateTmp=yes
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/leaderboard

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload
```

## Step 7: Configure Nginx Reverse Proxy

```bash
# Create nginx config
sudo tee /etc/nginx/sites-available/leaderboard > /dev/null << 'EOF'
upstream leaderboard {
    server unix:/opt/leaderboard/leaderboard.sock fail_timeout=0;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=submit_limit:10m rate=1r/s;

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;

    client_max_body_size 50M;

    location / {
        # Rate limiting
        limit_req zone=api_limit burst=20;
        
        proxy_pass http://leaderboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }

    location /api/submit {
        # Stricter rate limiting for submissions
        limit_req zone=submit_limit burst=5;
        
        proxy_pass http://leaderboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Longer timeout for file uploads
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Static files
    location /static {
        alias /opt/leaderboard/frontend/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/leaderboard /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Start nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Step 8: Setup SSL/TLS with Let's Encrypt

```bash
# Install certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get certificate (requires DNS pointing to server)
sudo certbot certonly --nginx -d your-domain.com

# Auto-renewal is setup automatically
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

## Step 9: Create Log Directory and Permissions

```bash
cd /opt/leaderboard

# Create directories
sudo mkdir -p logs uploads results
sudo chown -R www-data:www-data logs uploads results
sudo chmod -R 755 logs uploads results
```

## Step 10: Start Services

```bash
# Start the leaderboard service
sudo systemctl start leaderboard
sudo systemctl enable leaderboard

# Check status
sudo systemctl status leaderboard

# View logs
sudo journalctl -u leaderboard -f

# Check Nginx
sudo systemctl status nginx
```

## Step 11: Setup Monitoring

```bash
# Install monitoring tools (optional)
sudo apt-get install -y prometheus prometheus-node-exporter grafana-server

# Add health check to monitoring
# Endpoint: /api/health (should return 200 with healthy status)
```

## Step 12: Database Backups

```bash
# Create backup script
sudo tee /opt/leaderboard/backup.sh > /dev/null << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/leaderboard/backups"
mkdir -p $BACKUP_DIR
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/leaderboard_$DATE.sql.gz"

pg_dump -U leaderboard_user -h localhost leaderboard_db | gzip > $BACKUP_FILE

# Keep only last 30 backups
find $BACKUP_DIR -name "leaderboard_*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE"
EOF

sudo chmod +x /opt/leaderboard/backup.sh
sudo chown www-data:www-data /opt/leaderboard/backup.sh

# Add to crontab
# 0 2 * * * /opt/leaderboard/backup.sh >> /opt/leaderboard/logs/backup.log 2>&1
```

## Step 13: Health Monitoring

```bash
# Test health endpoint
curl https://your-domain.com/api/health

# Should return: {"status": "healthy", ...}
```

## Rollback Procedure

```bash
# Stop current service
sudo systemctl stop leaderboard

# Restore from backup
pg_restore -U leaderboard_user -h localhost leaderboard_db < backup_file.sql

# Restart service
sudo systemctl start leaderboard
```

## Troubleshooting

### Check application logs
```bash
sudo journalctl -u leaderboard -n 50
```

### Check database connection
```bash
psql -U leaderboard_user -h localhost -d leaderboard_db
```

### Restart services
```bash
sudo systemctl restart leaderboard
sudo systemctl restart nginx
```

### View Nginx errors
```bash
sudo tail -f /var/log/nginx/error.log
```

## Performance Tuning

### PostgreSQL (postgresql.conf)
```
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 32MB
maintenance_work_mem = 64MB
```

### Gunicorn workers
```
workers = (2 × CPU_cores) + 1
```

## Security Best Practices

1. ✓ Use HTTPS/TLS for all communications
2. ✓ Enable firewall and restrict ports
3. ✓ Use strong API tokens (rotate regularly)
4. ✓ Enable database backups and test restoration
5. ✓ Monitor logs for suspicious activity
6. ✓ Keep dependencies updated
7. ✓ Use environment variables for secrets
8. ✓ Enable audit logging
9. ✓ Restrict file upload sizes
10. ✓ Regular security audits

## Support & Maintenance

- Monitor `/api/health` endpoint regularly
- Review logs weekly
- Update dependencies monthly
- Test backups quarterly
- Conduct security audits bi-annually

