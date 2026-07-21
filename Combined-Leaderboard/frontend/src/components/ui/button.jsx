import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex h-10 min-h-10 items-center justify-center gap-2 border border-solid border-border-strong bg-surface px-4 py-0 font-sans text-sm font-semibold leading-5 text-foreground shadow-none transition-colors hover:border-foreground hover:bg-surface-subtle focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand disabled:pointer-events-none disabled:opacity-60 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "",
        brand: "border-border-strong bg-invert-bg text-invert-text hover:border-foreground hover:bg-invert-bg-hover",
        primary: "border-border-strong bg-invert-bg text-invert-text hover:border-foreground hover:bg-invert-bg-hover",
        ghost: "border-border-strong bg-surface text-foreground hover:border-foreground hover:bg-surface-subtle",
      },
      size: {
        default: "",
        sm: "h-9 min-h-9 px-3 py-0 text-xs",
        icon: "size-10 min-h-10 p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />;
});
Button.displayName = "Button";

export { Button, buttonVariants };
