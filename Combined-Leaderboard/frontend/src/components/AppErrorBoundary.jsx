import { Component } from "react";
import { Button } from "@/components/ui/button";

export class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Unhandled application render error", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="border-t border-border py-12">
        <div className="container">
          <div className="border border-negative border-l-[3px] bg-negative-soft p-4 text-negative" role="alert">
            <h1 className="mb-2 text-2xl font-semibold text-foreground">The page could not be displayed</h1>
            <p className="mb-4 leading-relaxed">An unexpected interface error occurred. Your account and submission data were not changed.</p>
            <Button type="button" variant="brand" onClick={() => window.location.reload()}>
              Reload page
            </Button>
          </div>
        </div>
      </main>
    );
  }
}
