import * as React from "react";
import { cn } from "@/lib/utils";

const Card = React.forwardRef(({ className, standalone = false, ...props }, ref) => (
  <div ref={ref} className={cn("card", standalone && "standalone", className)} {...props} />
));
Card.displayName = "Card";

export { Card };