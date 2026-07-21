# Deployment Cost Estimate

Last reviewed: 2026-07-12

This estimate covers the current MS VISTA architecture: a static React frontend,
one Flask API instance, SQLite, Microsoft authentication, transactional email,
and retained backups. It excludes the unfinished spatial evaluation pipeline,
hosted model inference, engineering time, and domain registration.

## Usage assumptions

| Item | Assumption |
|---|---:|
| Monthly users | 200 to 500 |
| Transactional emails | 500 to 1,500 per month |
| Model submissions | Fewer than 100 per month |
| API instances | One |
| GPU inference | None |
| Current SQLite size | Approximately 2.1 MB |
| Current verified backup size | Approximately 0.2 MB |

## Hugging Face estimate

| Component | Public pilot | Paid Hugging Face deployment |
|---|---:|---:|
| Static frontend Space | $0 | $0 |
| Backend CPU | $0 on CPU Basic | $3 to $21.90 |
| Hugging Face PRO | $0 | $9 if required |
| Storage Buckets | $0 at the expected size | $0 at the expected size |
| Microsoft authentication | $0 | $0 |
| Verification and reset email | $0.05 to $0.40 | $0.05 to $0.40 |
| Optional monitoring | $0 | $0 to $5 |
| Estimated total | Less than $1 per month | Approximately $13 to $33 per month |

### Compute

Hugging Face CPU Basic provides 2 vCPUs and 16 GB RAM at no charge. Free
hardware sleeps after extended inactivity, so the first request after a sleep
period has a cold start. CPU Upgrade provides 8 vCPUs and 32 GB RAM for
$0.03 per running hour.

| Monthly running time | CPU Upgrade cost |
|---|---:|
| 100 hours | $3.00 |
| 300 hours | $9.00 |
| 730 hours | $21.90 |

Hugging Face bills upgraded hardware only while it is starting or running. A
custom sleep timeout can reduce this cost. A PRO subscription is $9 per month
and is required for a custom Space domain or protected Space visibility. A
public deployment using its `hf.space` address does not need a custom domain.

### Storage

Free Hugging Face accounts currently include 100 GB of private Hub storage,
and that allowance applies to Storage Buckets. The current database and backup
footprint is only a few megabytes. Even after retaining submissions and rolling
backups, this deployment should remain inside the free allowance for a long
time.

Hugging Face lists private storage above the included allowance at a base price
of $18 per TB per month. This overage should not apply at the expected usage.

### Authentication and email

Microsoft Entra External ID includes the first 50,000 monthly active users at
no cost. The expected 200 to 500 users therefore remain in its free tier.

Azure Communication Services Email currently lists $0.00025 per email plus
$0.00012 per MB transferred. At 500 to 1,500 small transactional emails, the
expected charge is approximately $0.13 to $0.40 per month.

## SQLite durability consideration

The live SQLite database should not run directly on a Hugging Face Storage
Bucket mount. Storage Buckets are remote object storage, while this application
uses SQLite WAL mode. SQLite does not support WAL reliably over network
filesystems.

A Hugging Face only deployment would need to keep the active database on the
container filesystem, restore a verified snapshot during startup, and upload
new verified snapshots frequently. A sudden restart can still lose changes
made after the latest snapshot. That recovery window makes this arrangement
appropriate for a pilot, but not the strongest production option.

## Recommended low traffic production deployment

A small virtual machine with local SSD storage is a better fit for the current
single instance SQLite architecture.

| Component | Budget configuration | Safer configuration |
|---|---:|---:|
| DigitalOcean Basic Droplet | 1 GB at $6 | 2 GB at $12 |
| Weekly infrastructure backup | $1.20 | $2.40 |
| Azure transactional email | Less than $0.50 | Less than $0.50 |
| Static frontend | Included on the same server | Included on the same server |
| Hugging Face offsite backup bucket | $0 | $0 |
| Estimated total before domain | Approximately $8 per month | Approximately $15 per month |

The 1 GB configuration should handle the expected traffic with one Gunicorn
worker. The 2 GB configuration provides more headroom during large submission
validation, backup creation, and operating system updates.

The recommended production arrangement is:

1. Serve the static frontend and Flask API from one small virtual machine.
2. Keep SQLite on the virtual machine's local SSD.
3. Keep `WEB_CONCURRENCY=1` while using SQLite.
4. Create verified local backups every 48 hours and after important operations.
5. Mirror backup archives to a private Hugging Face Storage Bucket.
6. Keep Microsoft authentication and Azure Communication Services Email.

This is both less expensive and more reliable for SQLite than keeping an
upgraded Hugging Face Space running continuously.

## Pricing references

* [Hugging Face pricing](https://huggingface.co/pricing)
* [Hugging Face Spaces hardware and sleep behavior](https://huggingface.co/docs/hub/spaces-gpus)
* [Hugging Face storage limits](https://huggingface.co/docs/hub/storage-limits)
* [Hugging Face custom domains](https://huggingface.co/docs/hub/en/spaces-custom-domain)
* [Microsoft Entra External ID pricing](https://learn.microsoft.com/en-ie/entra/external-id/external-identities-pricing)
* [Azure Communication Services pricing](https://azure.microsoft.com/en-us/pricing/details/communication-services)
* [DigitalOcean Droplet pricing](https://www.digitalocean.com/pricing/droplets)
* [SQLite WAL documentation](https://sqlite.org/wal.html)

Recheck provider pricing before deployment because rates and free tier limits
can change.
