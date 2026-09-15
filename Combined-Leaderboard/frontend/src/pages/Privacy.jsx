import { WorkspacePage } from "@/components/WorkspacePage";
import { AUTH_TRANSPORT } from "@/lib/api";

const sections = [
  {
    title: "Information collected",
    body: "Account records include your email address, verification status, sign in provider, and a password hash when email authentication is used. Submission records include model metadata, final answers, scores, file hashes, request identifiers, and limited network information used for security and quotas.",
  },
  {
    title: "How information is used",
    body: "Information is used to operate accounts, verify email addresses, process evaluation-module submissions, publish approved results, investigate failures, prevent abuse, and maintain reproducible evaluation records.",
  },
  {
    title: "Submission files",
    body: "Visual capability response files are validated and scored in memory, with normalized final answers retained for owner and administrator audit. Reasoning-analysis submissions are different: the entire submitted Track 3 ZIP, including final answers, compressed model outputs, provenance, scores, and hashes, is retained and made public with the published result. Model outputs may contain reasoning text. Do not include personal data, contact details, credentials, or confidential material anywhere in the package. Public downloads can be copied by others; removing a result cannot recall those copies.",
  },
  {
    title: "Storage and retention",
    body: "Account and submission records are stored in a restricted database. Verified backups use a separate restricted storage volume and rotate according to the deployment retention policy. Deletion from the live service does not immediately erase older backups; those copies remain until they expire. Published evaluation records may remain while the research leaderboard is active. Restoring a backup requires the operator to reapply subsequent account-deletion requests before reopening the service.",
  },
  {
    title: "Service providers",
    body: "The current deployment uses GitHub Pages for the frontend and Oracle Cloud Infrastructure for the backend, database, and backup storage. Azure Communication Services delivers verification and password reset email. Selecting Microsoft sign in sends you to Microsoft's identity service. These providers may process connection and security information under their own terms. This hosting description is not a claim of Microsoft privacy or compliance approval.",
  },
  {
    title: "Children and minimum age",
    body: "This research leaderboard is intended for researchers and practitioners aged 16 or older and is not directed to children. Sign in verifies account access; it does not independently verify a person's age or establish parental consent. If you believe a child has created an account, contact the project administrator for review and removal. The service owner must review any applicable age, region, or parental-consent requirements before a public production launch.",
  },
  {
    title: "Cookies and security",
    body: "MS-VISTA does not include advertising or analytics trackers. Essential browser storage supports sign in, session recovery, request protection, and theme preference. GitHub Pages uses bearer authentication rather than cross-site session cookies; the same-origin backend frontend can use a secure HttpOnly session cookie. Microsoft sign in may temporarily use an API-origin cookie to protect the redirect flow, and Microsoft controls storage on its own sign-in pages. Operational safeguards include access controls, request limits, encrypted transport, restricted backups, and audit records.",
  },
];

const storage = [
  ["lb_refresh_token_v1", "Session storage", "Rotating refresh token for bearer sign in; access tokens stay in memory.", "Cleared on logout or when the tab session ends; also expires server-side."],
  ["lb_user", "Local storage", "Cached signed-in email and session UI information; not authorization proof.", "Cleared on logout or invalid session; otherwise until browser data is cleared."],
  ["lb_csrf_token", "Local storage", "Request-protection token for cookie authentication.", "Cleared on logout or invalid session."],
  ["vci-theme", "Local storage", "Chosen light or dark appearance.", "Until changed or browser data is cleared."],
  ["ms_vista_session", "Secure HttpOnly cookie on the API origin", "Same-origin session and temporary OAuth redirect state; not required for GitHub Pages bearer API requests.", "Session or configured server expiry; cleared on logout."],
  ["vista_oauth_state", "Secure HttpOnly cookie on the API origin", "Binds Microsoft sign-in redirects to the browser that started them.", "Ten-minute maximum; removed when the redirect flow completes."],
];

export function Privacy() {
  return (
    <WorkspacePage
      eyebrow="Privacy"
      title="Privacy notice"
      description="This notice explains how MS VISTA handles account, submission, and operational information for the research leaderboard."
    >
      <div className="border-y border-border-strong">
        {sections.map((section, index) => (
          <section
            className="grid border-b border-border last:border-b-0 lg:grid-cols-[minmax(14rem,0.42fr)_minmax(0,1fr)]"
            key={section.title}
          >
            <div className="border-b border-border bg-surface-subtle px-6 py-6 lg:border-b-0 lg:border-r lg:px-8">
              <span className="mb-3 block text-sm font-medium text-faint">
                {String(index + 1).padStart(2, "0")}
              </span>
              <h2 className="font-display text-xl font-semibold text-foreground">
                {section.title}
              </h2>
            </div>
            <p className="m-0 max-w-4xl px-6 py-6 leading-7 text-muted lg:px-8">
              {section.body}
            </p>
          </section>
        ))}
      </div>
      <section id="browser-storage-inventory" className="border-b border-border-strong px-6 py-8 lg:px-8">
        <h2 className="font-display text-xl font-semibold">Browser storage inventory</h2>
        <p className="mt-3 text-sm text-muted">This frontend uses {AUTH_TRANSPORT === "bearer" ? "bearer" : "cookie"} authentication. This inventory covers application-controlled storage, not the contents of third-party sign-in pages.</p>
        <dl className="mt-5 divide-y divide-border">
          {storage.map(([name, type, purpose, lifetime]) => (
            <div key={name} className="py-4">
              <dt className="break-words font-mono text-sm font-semibold">{name}</dt>
              <dd className="mt-2 text-sm leading-6 text-muted">{type}. {purpose} {lifetime}</dd>
            </div>
          ))}
        </dl>
      </section>
      <div className="border-b border-border-strong px-6 py-8 lg:px-8">
        <p className="m-0 max-w-4xl leading-7 text-muted">
          You can download your account data and request account deletion from your profile. Deletion removes the live authentication record and anonymises the account linkage in retained submissions; published model metadata and evidence remain part of the research record. Personal information embedded by a submitter inside an uploaded file is not automatically scrubbed. For correction, evidence removal, or other privacy requests, contact the project administrator. This notice was last updated on 15 September 2026.
        </p>
      </div>
    </WorkspacePage>
  );
}
