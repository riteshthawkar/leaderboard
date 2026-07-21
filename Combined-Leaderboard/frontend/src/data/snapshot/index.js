// Frozen API snapshot for the static (no-backend) build (VITE_STATIC=1).
// The static demo serves these instead of calling the Flask API, so charts and
// tables show the data captured at snapshot time.
//
// Regenerate all payloads against a running server with:
//   .venv/bin/python scripts/refresh_static_snapshots.py
import statisticsOverview from "./statistics-overview.json";
import leaderboardVisualCognition from "./leaderboard-visual-cognition.json";
import leaderboardSpatial from "./leaderboard-spatial.json";
import taskDoYouSeeMe from "./task-do_you_see_me.json";
import taskMindsEye from "./task-minds_eye.json";
import taskSpatial from "./task-spatial.json";
import authProviders from "./auth-providers.json";
import modelReports from "./model-reports.json";

export const snapshots = {
  "/api/statistics/overview": statisticsOverview,
  "/api/leaderboard/visual-cognition": leaderboardVisualCognition,
  "/api/leaderboard/spatial": leaderboardSpatial,
  "/api/tasks/do_you_see_me/info": taskDoYouSeeMe,
  "/api/tasks/minds_eye/info": taskMindsEye,
  "/api/tasks/spatial/info": taskSpatial,
  "/api/auth/providers": authProviders,
  ...modelReports,
};
