import * as React from "react";
import { cn } from "@/lib/utils";

const Card = React.forwardRef(({ className, standalone = false, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("min-w-0 bg-surface text-foreground", standalone && "border border-border p-6 shadow-none", className)}
    {...props}
  />
));
Card.displayName = "Card";

export { Card };
