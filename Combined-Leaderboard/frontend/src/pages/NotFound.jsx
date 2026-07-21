import { ArrowLeft, Trophy } from "lucide-react";
import { Link } from "react-router-dom";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";

export function NotFound() {
  return (
    <WorkspacePage
      eyebrow="404"
      title="Page not found"
      description="The address does not match a page in this application. Use one of the links below to continue."
    >
      <div className="flex flex-wrap gap-3 border-y border-border-strong px-6 py-8 max-sm:flex-col lg:px-8">
        <Button asChild variant="brand">
          <Link to="/"><ArrowLeft size={16} />Return to overview</Link>
        </Button>
        <Button asChild variant="ghost">
          <Link to="/leaderboard"><Trophy size={16} />View leaderboard</Link>
        </Button>
      </div>
    </WorkspacePage>
  );
}
