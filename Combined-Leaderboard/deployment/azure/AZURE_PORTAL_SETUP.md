# MS-VISTA Azure Portal Resource Setup

Last verified: 2026-07-31

This runbook describes the Azure resources and identity configuration needed to
finish the production deployment of the MS-VISTA leaderboard.

## Deployment constants

Use these existing values unless the Azure portal shows that the VM belongs to a
different resource group or subscription:

| Setting | Value |
| --- | --- |
| Subscription | MSR-VISTA |
| Subscription ID | `4a7ebf13-fe6e-4bb8-b17f-52787e55401c` |
| Resource group | `leaderboard` |
| Virtual machine | `MSR-VISTA-Leaderboard` |
| Current public IP | `172.198.69.11` |
| Temporary hostname | `172-198-69-11.sslip.io` |

Use the VM's existing Azure region for disks, storage, monitoring, and backup
resources. Azure Communication Services uses a separate **Data location**
setting. Its Communication Services and Email Communication Services resources
must use the same data geography.

Do not paste any client secret, storage key, connection string, or private SSH
key into chat, tickets, source control, resource tags, or portal notes.

## 1. Establish organizational administration

Azure subscriptions do not generate a shared username and password. Use
individual Microsoft Entra accounts and Azure RBAC.

1. In the Azure portal, open **Microsoft Entra ID**.
2. Open **Users** and add or invite the organizational MSR-VISTA accounts.
3. Open **Groups** and select **New group**.
4. Select **Security** as the group type.
5. Name the group `MS-VISTA-Platform-Admins`.
6. Add at least two organizational administrators.
7. Open **Resource groups** and select `LEADERBOARD`.
8. Open **Access control (IAM)** and select **Add role assignment**.
9. Assign `Contributor` to `MS-VISTA-Platform-Admins`.
10. Assign `Role Based Access Control Administrator` to the designated access
    administrator only.
11. Sign in using one organizational account and verify that it can read and
    update `MSR-VISTA-Leaderboard`.
12. Keep the personal account temporarily. Remove its role assignment only after
    the organizational account and emergency recovery account have been tested.

Completion check:

- Two organizational accounts can access `LEADERBOARD`.
- MFA is enabled for both accounts.
- The personal account is no longer the only owner of the deployment.

For Azure CLI access, use the organizational tenant and subscription:

```bash
az logout
az login --tenant <MSR_VISTA_TENANT_ID> --use-device-code
az account set --subscription 4a7ebf13-fe6e-4bb8-b17f-52787e55401c
az account show \
  --query '{name:name,id:id,tenantId:tenantId,user:user.name}' \
  --output table
```

Do not run Azure deployment commands if `az account show` reports the personal
`Azure for Students` subscription.

## 2. Create and attach the application data disk

This disk will hold the database, submissions, generated manifests, and other
persistent application state.

1. Open **Virtual machines** and select `MSR-VISTA-Leaderboard`.
2. Under **Settings**, select **Disks**.
3. Under **Data disks**, select **Create and attach a new disk**.
4. Enter `ms-vista-prod-data` as the disk name.
5. Select `Standard SSD LRS` as the storage type.
6. Enter `64 GiB` as the size.
7. Select platform-managed encryption.
8. Set **Host caching** to `None`.
9. Leave the automatically assigned LUN, normally `0`.
10. Disable **Delete with VM** so accidental VM deletion does not delete the
    application data disk.
11. Select **Save**.
12. Wait until the disk row reports `Attached`.

Do not partition, format, or mount the disk from the portal. The host-side
migration must first identify the new device by Azure LUN and preserve the
existing `/srv/ms-vista` data.

Completion check:

- `ms-vista-prod-data` appears under the VM's data disks.
- Size is 64 GiB, SKU is Standard SSD LRS, and Delete with VM is disabled.

## 3. Verify the static public IP

The VM already has public IP `172.198.69.11`. Do not create a second public IP
unless the current resource is not Standard and Static.

1. Open `MSR-VISTA-Leaderboard`.
2. Select **Networking** and then the attached network interface.
3. Open **IP configurations** and select the primary IP configuration.
4. Open the associated public IP resource.
5. Confirm **SKU** is `Standard`.
6. Confirm **Assignment** or **Allocation method** is `Static`.
7. Record the public IP resource name.

If the public IP is not Standard and Static, stop here. Changing an attached
public IP can interrupt SSH and DNS, so it should be done in a maintenance
window.

Completion check:

- The portal still displays `172.198.69.11`.
- The public IP is Standard and Static.

## 4. Configure the network security group

Caddy terminates TLS on the VM. The frontend, API, database, and Docker ports
must not be directly exposed.

1. Open the VM and select **Networking**.
2. Open the network security group attached to the VM's network interface.
3. Select **Inbound security rules**.
4. Add an SSH rule:
   - Source: `IP Addresses`
   - Source IP: each administrator's public IP with `/32`
   - Source port: `*`
   - Destination: `Any`
   - Service: `SSH`
   - Action: `Allow`
   - Priority: `100`
   - Name: `Allow-SSH-Admins`
5. Add an HTTP rule:
   - Source: `Any`
   - Service: `HTTP`
   - Action: `Allow`
   - Priority: `200`
   - Name: `Allow-HTTP`
6. Add an HTTPS rule:
   - Source: `Any`
   - Service: `HTTPS`
   - Action: `Allow`
   - Priority: `210`
   - Name: `Allow-HTTPS`
7. Remove or disable any broader Internet-to-SSH rule after the restricted SSH
   rule has been tested in a second terminal.
8. Do not create inbound rules for `5173`, `8000`, `8021`, SQLite, Docker, or
   internal health-check ports.

TCP 80 is required for Caddy's normal HTTP-to-HTTPS redirect and ACME
certificate challenge. UDP 443 is optional and not required for launch.

Completion check:

- Only TCP 80 and 443 are open publicly.
- TCP 22 is restricted to known administrator IP addresses.

## 5. Create the storage account

The storage account provides an off-VM Azure Files share for application-level
backups.

1. Search for **Storage accounts** and select **Create**.
2. On **Basics**, select the MSR-VISTA subscription.
3. Select resource group `LEADERBOARD`.
4. Enter a globally unique lowercase name such as
   `msvistaprod<unique-suffix>`.
5. Select the same region as `MSR-VISTA-Leaderboard`.
6. Select `Standard` performance.
7. Select `Locally-redundant storage (LRS)`.
8. On **Advanced**, enable **Secure transfer required**.
9. Set **Minimum TLS version** to `1.2`.
10. Disable anonymous blob access.
11. Leave hierarchical namespace, SFTP, and NFS disabled.
12. Keep storage account key access enabled for the initial Linux SMB mount.
13. On **Networking**, select **Enabled from selected virtual networks and IP
    addresses**.
14. Add the VM's existing virtual network and subnet. Allow the portal to enable
    the `Microsoft.Storage` service endpoint if prompted.
15. Do not create a public blob container.
16. Add tags such as `service=ms-vista`, `environment=production`, and
    `owner=<organizational-team>`.
17. Select **Review + create**, validate, and select **Create**.

Completion check:

- The storage account is in `LEADERBOARD` and the VM's region.
- Secure transfer and TLS 1.2 are enabled.
- Anonymous blob access is disabled.
- The VM subnet is allowed by the storage firewall.

## 6. Create the Azure Files backup share

1. Open the new storage account.
2. Under **Data storage**, select **File shares**.
3. Select **+ File share**.
4. Enter `ms-vista-backups`.
5. Select the SMB protocol.
6. Select `Transaction optimized` as the access tier.
7. Set the quota to `50 GiB`.
8. Create the share.
9. Open **Data protection** on the storage account.
10. Enable soft delete for file shares with at least 14 days retention.
11. Return to the share and confirm its status is available.

Do not copy the storage key into this repository. The VM will store it in a
root-only credentials file outside the application directory and mount the share
at `/mnt/ms-vista-backups` using SMB 3.1.1.

Completion check:

- Share name is exactly `ms-vista-backups`.
- Soft delete is enabled.
- The share is not exposed through anonymous access.

## 7. Create a Recovery Services vault

Azure Files stores logical application backups. A Recovery Services vault
provides separate VM-level disaster recovery.

1. Search for **Recovery Services vaults** and select **Create**.
2. Select the MSR-VISTA subscription and `LEADERBOARD`.
3. Enter `ms-vista-backup-vault`.
4. Select the same region as the VM.
5. Select **Review + create** and create the vault.
6. Open the vault and select **Properties**.
7. Choose `GRS` redundancy if organizational policy and budget permit it;
   otherwise use `LRS`.
8. Keep soft delete enabled.
9. Do not permanently lock vault immutability until a restore drill and
   retention policy have been approved.
10. Open **Backup**, choose **Azure Virtual Machine**, and create a policy.
11. Use a daily schedule during the VM's lowest-traffic period.
12. Retain daily recovery points for at least 14 days and weekly recovery points
    for at least 4 weeks.
13. Select `MSR-VISTA-Leaderboard` and enable backup.
14. Run **Backup now** only after the new data disk has been mounted and the
    application migration has completed.

Completion check:

- The VM is listed as a protected item.
- The policy includes the OS and attached data disks.
- The first post-migration backup completes successfully.

## 8. Create Email Communication Services

This resource owns the verified sender domain. It is separate from the
Communication Services resource that sends messages.

1. Search for **Email Communication Services**.
2. Select **Create**.
3. Select the MSR-VISTA subscription and `LEADERBOARD`.
4. Enter `ms-vista-email-prod`.
5. Select an organization-approved **Data location**.
6. Record the chosen data location because the Communication Services resource
   must use the same geography.
7. Select **Review + create** and create the resource.
8. Open the resource and select **Provision domains**.

For immediate functional testing:

1. Select **Add free Azure subdomain**.
2. Create the Azure-managed domain.
3. Wait until its status is ready.
4. Record its default `DoNotReply@<generated-domain>.azurecomm.net` MailFrom
   address.

For production branding:

1. Use a dedicated subdomain such as `notifications.ms-vista.org`; do not use the
   temporary `sslip.io` hostname.
2. Select **Add custom domain**.
3. Enter the complete email subdomain.
4. Copy the portal-provided domain ownership TXT record into the authoritative
   DNS provider.
5. Add the portal-provided SPF TXT record.
6. Add both portal-provided DKIM CNAME records.
7. Add a DMARC TXT record at `_dmarc.<email-subdomain>`. Begin with an
   organization-approved monitoring policy such as `p=none`, then tighten it
   after delivery monitoring.
8. Return to Azure and run each verification action.
9. Wait until domain ownership, SPF, and DKIM all show verified.
10. Add sender username `no-reply` with display name `MS-VISTA Leaderboard` if
    the custom domain supports sender usernames.

Completion check:

- At least one domain is fully ready.
- The exact MailFrom address is recorded.
- A custom production domain has verified SPF and DKIM before public use.

## 9. Create Azure Communication Services

1. Search for **Communication Services** and select **Create**.
2. Select the MSR-VISTA subscription and `LEADERBOARD`.
3. Enter `ms-vista-acs-prod`.
4. Select the same data geography used by `ms-vista-email-prod`.
5. Select **Review + create** and create the resource.
6. Open the resource.
7. Under **Email**, select **Domains**.
8. Select **Connect domain**.
9. Filter by the MSR-VISTA subscription, `LEADERBOARD`, and
   `ms-vista-email-prod`.
10. Select the fully verified domain and choose **Connect**.
11. Open **Overview** or **Keys** and record only the endpoint:
    `https://ms-vista-acs-prod.communication.azure.com`.
12. Do not copy the connection string unless a temporary diagnostic fallback is
    explicitly required.
13. Open **Try Email**, choose the connected sender, and send a message to a
    controlled mailbox.
14. Confirm that the operation and actual mailbox delivery succeed.

Completion check:

- The domain appears under Connected domains.
- Try Email delivers a message.
- The endpoint and MailFrom address are recorded.

## 10. Configure ACS connection-string authentication

This deployment uses the ACS connection string because the deployment operator
does not have permission to create Azure role assignments. No VM managed
identity or ACS IAM role assignment is required for this mode.

1. Open `ms-vista-acs-prod`.
2. Under **Settings**, select **Keys**.
3. Locate the primary connection string.
4. Select **Show** and copy the complete value.
5. Do not paste the value into chat, source control, portal notes, or a shell
   history entry.
6. Store it only in the root-owned production environment file on the VM:
   `/srv/ms-vista/app/deployment/azure/production.env`.
7. Set `ACS_ENDPOINT` to an empty value. When both values are configured, the
   application prioritizes the connection string, so leaving the endpoint empty
   makes the selected mode explicit.
8. Set `ACS_SENDER_ADDRESS` to the exact MailFrom address from the connected
   verified domain.
9. Ensure the production environment file remains owned by root with mode
   `0600`.

The production application will use:

```dotenv
ACS_CONNECTION_STRING=endpoint=https://<resource>.communication.azure.com/;accesskey=<secret>
ACS_ENDPOINT=
ACS_SENDER_ADDRESS=<exact-MailFrom-address>
```

The placeholder above illustrates the format only. Never place a real
connection string in this document or any tracked file.

Completion check:

- `ACS_CONNECTION_STRING` and `ACS_SENDER_ADDRESS` are configured.
- `ACS_ENDPOINT` is empty.
- The API email readiness check passes.
- A registration-verification email and password-reset email are delivered.

For key rotation, switch the application to the secondary connection string,
restart and test email, regenerate the primary key, and then optionally switch
back. Never regenerate the key currently used by the running application.

## 11. Create the Microsoft sign-in app registration

This registration controls optional "Sign in with Microsoft" user login. It is
tenant-scoped and separate from ACS email authentication.

1. Open **Microsoft Entra ID**.
2. Select **App registrations** and **New registration**.
3. Enter `MS-VISTA Leaderboard Production`.
4. For a public leaderboard, select accounts in any organizational directory
   plus personal Microsoft accounts. Use single tenant only if sign-in must be
   limited to the MSR-VISTA organization.
5. Under **Redirect URI**, choose `Web`.
6. Add:
   `https://172-198-69-11.sslip.io/api/auth/oauth/microsoft/callback`
7. Select **Register**.
8. Record the Application (client) ID and Directory (tenant) ID.
9. Open **Authentication** and confirm the redirect is under the Web platform,
   not the Single-page application platform.
10. Do not enable implicit access-token or ID-token grants; the backend uses the
    authorization-code flow with PKCE.
11. Open **API permissions**. Do not add application-level Microsoft Graph
    permissions. The application requests only `openid email profile`.
12. Open **Certificates & secrets** and create a client secret with the shortest
    operationally practical lifetime, no longer than 12 months.
13. Copy the secret **Value** once into the protected production secret store.
    The Secret ID is not the secret value.
14. Create an organizational reminder at least 30 days before expiration.
15. Open **Owners** and add at least two organizational owners.

When a permanent hostname is available, add its exact Web redirect before
removing the temporary redirect:

```text
https://leaderboard.<organization-domain>/api/auth/oauth/microsoft/callback
```

For public work and personal Microsoft accounts, the application configuration
uses:

```dotenv
MICROSOFT_CLIENT_ID=<application-client-id>
MICROSOFT_CLIENT_SECRET=<secret-value>
MICROSOFT_TENANT_ID=common
```

Use `organizations` instead of `common` when personal Microsoft accounts should
not be accepted.

Completion check:

- The Web redirect URI matches the deployed hostname and path exactly.
- Two organizational owners are assigned.
- No unnecessary application-level Graph permission was granted.

## 12. Configure DNS

Skip Azure DNS resource creation if the organization already manages its domain
through another DNS provider.

1. Decide on a permanent hostname such as `leaderboard.ms-vista.org`.
2. In the authoritative DNS provider, create an `A` record.
3. Set the record name to `leaderboard`.
4. Set the value to `172.198.69.11`.
5. Use a short TTL such as 300 seconds during cutover.
6. Wait for public resolution.
7. Add the permanent Microsoft OAuth redirect URI before switching the
   application to the hostname.
8. After DNS resolves, Caddy can request the public TLS certificate.

Completion check:

```bash
dig +short leaderboard.<organization-domain>
```

The command must return `172.198.69.11`.

## 13. Create monitoring and notification resources

Create these after an organizational operations mailbox exists.

1. Search for **Monitor** and open **Alerts**.
2. Select **Action groups** and **Create**.
3. Use `LEADERBOARD` and name the group `ms-vista-prod-alerts`.
4. Add an email receiver using an organizational operations address.
5. Test the action group.
6. Add a VM availability alert for `MSR-VISTA-Leaderboard`.
7. Add a CPU alert for sustained high utilization, for example above 85 percent
   for 15 minutes.
8. Enable VM Insights or the Azure Monitor Agent before creating guest disk-space
   alerts.
9. After the HTTPS endpoint is public, create an Application Insights standard
   availability test for a public health URL.
10. Route backup-failure alerts from `ms-vista-backup-vault` to the same action
    group.

Completion check:

- The test alert reaches the organizational mailbox.
- VM unavailability and backup failures have active alert rules.

## 14. Information to provide for deployment completion

The following values are safe to share:

- Managed disk resource name
- Storage account name
- File share name
- Recovery Services vault name
- ACS endpoint
- Email MailFrom address
- Microsoft Application client ID
- Microsoft tenant ID
- Permanent application hostname

Do not share:

- Microsoft client secret
- ACS connection string or access key
- Storage account key or SAS token
- SSH private key
- Flask `SECRET_KEY`

After the required resources are ready, the host-side completion sequence is:

1. Format and mount the managed disk.
2. Stop the application briefly and migrate `/srv/ms-vista`.
3. Mount `ms-vista-backups` persistently.
4. Update the protected production environment.
5. Start Caddy and the `ms-vista` systemd service.
6. Verify public TLS, API readiness, registration email, password reset, and
   Microsoft sign-in.
7. Run and verify an application backup.
8. Perform a restore drill.
9. Run the first VM-level backup.
