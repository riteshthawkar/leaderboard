"use client";;
import { motion, useSpring } from "motion/react";
import { useChartConfig } from "../chart-config-context";
import { chartCssVars } from "../chart-context";

export function TooltipDot({
  x,
  y,
  visible,
  color,
  size = 5,
  strokeColor = chartCssVars.background,
  strokeWidth = 2,
  springConfig,
  animate = true
}) {
  const { tooltipSpring } = useChartConfig();
  const effectiveSpring = springConfig ?? tooltipSpring;
  const animatedX = useSpring(x, effectiveSpring);
  const animatedY = useSpring(y, effectiveSpring);

  if (animate) {
    animatedX.set(x);
    animatedY.set(y);
  }

  if (!visible) {
    return null;
  }

  if (!animate) {
    return (
      <circle
        cx={x}
        cy={y}
        fill={color}
        r={size}
        stroke={strokeColor}
        strokeWidth={strokeWidth} />
    );
  }

  return (
    <motion.circle
      cx={animatedX}
      cy={animatedY}
      fill={color}
      r={size}
      stroke={strokeColor}
      strokeWidth={strokeWidth} />
  );
}

TooltipDot.displayName = "TooltipDot";

export default TooltipDot;
