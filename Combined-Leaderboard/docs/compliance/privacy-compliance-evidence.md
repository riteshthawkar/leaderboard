# Privacy compliance evidence

This record maps the final two Privacy tasks in RC&ET ID 10466 to implementation and review evidence. It contains no credentials, user records, or submission data.

## 1. Include all mandatory disclosures

Status: implemented in the shared footer on every frontend route.

| Requirement | Public label or statement | Implementation evidence |
| --- | --- | --- |
| Microsoft Privacy Statement | Privacy & Cookies | `frontend/src/components/Layout.jsx` links to `https://go.microsoft.com/fwlink/?LinkId=521839`. |
| Consumer Health Privacy Statement | Consumer Health Privacy | `frontend/src/components/Layout.jsx` links to `https://go.microsoft.com/fwlink/?linkid=2259814`. |
| Microsoft trademark notice | Trademarks | `frontend/src/components/Layout.jsx` links to `https://www.microsoft.com/trademarks`. |
| Website terms | Terms of Use | `frontend/src/components/Layout.jsx` links to `https://go.microsoft.com/fwlink/?LinkID=206977`. |
| Current copyright notice near the terms link | Copyright [current year] Microsoft | The year is generated at runtime and rendered in the legal-disclosures navigation region beside Terms of Use. |
| Service-specific notice | Leaderboard Data Notice | The footer links to the local `/privacy` route by default or to `VITE_PRIVACY_POLICY_URL` when an approved external notice is configured. |
| About our ads | Not applicable | MS VISTA does not display third-party advertising. No advertising SDK, ad placement, or advertising component is present in the frontend. Reassess this item before introducing advertising. |

The non-applicability determination was checked with a repository search for common advertising SDK and network markers; no matches were found in the frontend or backend application sources.

Automated evidence:

- `frontend/tests/components/Layout.test.jsx` asserts every required label and exact destination.
- `frontend/tests/pages/Profile.test.jsx` verifies the account export and deletion controls described by the data notice.
- `tests/backend/test_auth_data_rights.py` verifies authenticated export and irreversible identity anonymisation across separate authentication and submission databases while preserving published research results.

Verification recorded on 2026-09-05:

- Backend suite: 438 passed, 4 skipped.
- Frontend suite: 47 passed.
- Frontend lint: passed with zero warnings.
- Production frontend build: passed with an explicit HTTPS API origin and `/leaderboard/` asset base path.

Rendered evidence:

- `docs/compliance/evidence/privacy-disclosures-desktop.png` (1440 x 900; SHA-256 `7c17c0502723d598b3a45605734f4437dbb96dc4425b53a9ea6556c37344e1a9`)
- `docs/compliance/evidence/privacy-disclosures-mobile.png` (390 x 844; SHA-256 `c40ce9f1eec814bedbea2cd9bb28caee10b4e18e6ce2837df59b870fec3924c2`)

Both screenshots were captured on 2026-09-05 from the isolated release-candidate commit. Browser inspection confirmed all five link destinations, the current copyright year, a working local `/privacy` route, and no console errors or warnings at either viewport.

Regenerate the screenshots from the deployed production URL whenever the footer content, hosting path, or canonical domain changes.

## 2. Register the site

Status: external operator action required. Code changes cannot complete or truthfully evidence Web Compliance Portal registration.

Use the following registration values:

| Field | Value |
| --- | --- |
| Site top-level URL | **TBD: final canonical production URL. Do not register the development repository URL.** |
| Hosting domain | **TBD: derived from the final production URL.** |
| Verified FTE owner 1 | Required from the MS VISTA team |
| Verified FTE owner 2 | Required from the MS VISTA team |
| Service Tree ID | Required from the owning Microsoft service |
| Privacy contact | `PrivCon_TnR@microsoft.com` |

Completion procedure:

1. Open the Web Compliance Portal at `https://aka.ms/wcp` using the appropriate Microsoft corporate account.
2. Create a site record, or locate the existing record and select **Edit Site Details**.
3. Enter the final user-facing production URL, two verified FTE owners, Service Tree ID, privacy contact, and the remaining required ownership metadata. Do not use a temporary development or source-repository URL.
4. Save the record and copy the resulting site-details URL.
5. Enter that URL in the RC&ET field **Privacy: Website Registration**.
6. Capture a screenshot showing the saved site URL and registration status. Do not include access tokens, credentials, or unrelated personal data.

Registration evidence remains incomplete until the Web Compliance Portal record URL and screenshot are attached to RC&ET. Do not mark the task complete based only on this repository record.

## Verification commands

```bash
cd Combined-Leaderboard
.venv/bin/python -m pytest -q

cd frontend
npm run lint
npm run test:run
VITE_API_BASE_URL=https://api.example.test \
VITE_BASE_PATH=/leaderboard/ \
npm run build
```
