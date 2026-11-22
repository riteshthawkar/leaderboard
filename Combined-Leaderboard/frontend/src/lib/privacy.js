// Microsoft Privacy Statement: required on releases intended for people outside Microsoft.
// Its link text must read exactly "Privacy & Cookies" and point at this forward link.
export const MICROSOFT_PRIVACY_STATEMENT_URL = "https://go.microsoft.com/fwlink/?LinkId=521839";

// Service-specific notice describing what this leaderboard stores. It supplements the Microsoft
// Privacy Statement above; it does not replace it.
export const LEADERBOARD_DATA_NOTICE_URL = (import.meta.env.VITE_PRIVACY_POLICY_URL || "/privacy").trim();
export const LEADERBOARD_DATA_NOTICE_IS_EXTERNAL = /^https?:\/\//i.test(LEADERBOARD_DATA_NOTICE_URL);
