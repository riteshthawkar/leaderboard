import { WorkspacePage } from "@/components/WorkspacePage";

const sections = [
  {
    title: "Information collected",
    body: "Account records include your email address, verification status, sign in provider, and a password hash when email authentication is used. Submission records include model metadata, final answers, scores, file hashes, request identifiers, and limited network information used for security and quotas.",
  },
  {
    title: "How information is used",
    body: "Information is used to operate accounts, verify email addresses, process benchmark submissions, publish approved leaderboard results, investigate failures, prevent abuse, and maintain reproducible evaluation records.",
  },
  {
    title: "Submission files",
    body: "Visual benchmark response files are validated and scored in memory, with normalized final answers retained for owner and administrator audit. Spatial submissions are different: the original ZIP, per sample final answer evidence, run manifest, aggregate report, scores, and integrity hashes are retained in the database and made public with the leaderboard result. Raw reasoning traces are not required or published.",
  },
  {
    title: "Storage and retention",
    body: "Account and submission records are stored in a protected SQLite database. Verified backup archives are retained according to the deployment retention policy. Records may remain while the research leaderboard is active or as needed to preserve published evaluation history.",
  },
  {
    title: "Service providers",
    body: "Hugging Face provides application hosting and private storage. Microsoft or Google may process identity information when their sign in option is selected. Azure Communication Services may deliver verification and password reset email.",
  },
  {
    title: "Cookies and security",
    body: "The application uses an essential secure session cookie and browser storage for cross site request protection, theme preference, and session recovery. Operational safeguards include access controls, request limits, encrypted transport, restricted backups, and audit records.",
  },
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
      <div className="border-b border-border-strong px-6 py-8 lg:px-8">
        <p className="m-0 max-w-4xl leading-7 text-muted">
          You may review your account and submission history through the application. Requests concerning correction, removal, or privacy should be directed to the project administrator. This notice was last updated on 14 July 2026.
        </p>
      </div>
    </WorkspacePage>
  );
}
