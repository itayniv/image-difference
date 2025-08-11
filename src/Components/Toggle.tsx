import React, { useEffect, useState, useCallback, forwardRef } from "react";

type SizeKey = "sm" | "md" | "lg";

export interface ToggleProps {
  checked?: boolean;
  defaultChecked?: boolean;
  onChange?: (checked: boolean) => void;
  disabled?: boolean;
  size?: SizeKey;
  "aria-label"?: string;
  className?: string;
}

const SIZE_CLASSES: Record<SizeKey, { track: string; knob: string; translate: string }> = {
  sm: {
    track: "w-10 h-5 p-[2px]",
    knob: "w-4 h-4",
    translate: "translate-x-5",
  },
  md: {
    track: "w-14 h-7 p-[2px]",
    knob: "w-6 h-6",
    translate: "translate-x-7",
  },
  lg: {
    track: "w-18 h-9 p-[2px]",
    knob: "w-8 h-8",
    translate: "translate-x-9",
  },
};

const Toggle = forwardRef<HTMLButtonElement, ToggleProps>(
  (
    {
      checked: controlledChecked,
      defaultChecked = false,
      onChange,
      disabled = false,
      size = "sm",
      className = "",
      ...rest
    },
    ref
  ) => {
    const isControlled = typeof controlledChecked === "boolean";
    const [internalChecked, setInternalChecked] = useState<boolean>(defaultChecked);
    const checked = isControlled ? (controlledChecked as boolean) : internalChecked;

    useEffect(() => {
      if (!isControlled) return;
    }, [controlledChecked, isControlled]);

    const toggle = useCallback(() => {
      if (disabled) return;
      const next = !checked;
      if (!isControlled) setInternalChecked(next);
      onChange?.(next);
    }, [checked, disabled, isControlled, onChange]);

    return (
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={rest["aria-label"] ?? "Toggle"}
        ref={ref}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            toggle();
          }
        }}
        disabled={disabled}
        className={`relative inline-flex items-center rounded-full transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
          disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer"
        } ${SIZE_CLASSES[size].track} ${
          checked ? "bg-blue-600 shadow-[0_0_0_6px_rgba(37,99,235,0.12)]" : "bg-gray-200"
        } ${className}`}
      >
        <span
          className={`bg-white rounded-full shadow-[0_2px_6px_rgba(0,0,0,0.12)] transform transition-transform duration-200 ${SIZE_CLASSES[size].knob} ${
            checked ? SIZE_CLASSES[size].translate : "translate-x-0"
          }`}
        />
      </button>
    );
  }
);

Toggle.displayName = "Toggle";

export default Toggle;
