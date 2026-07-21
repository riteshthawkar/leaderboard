import { useRef, useState } from "react";
import { UploadCloud, FileText, X } from "lucide-react";
import { cn } from "@/lib/utils";

function formatBytes(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

export function FileDropzone({ name, accept = "", required = false, hint, maxBytes }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const acceptList = accept
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);

  const isAllowed = (candidate) =>
    !acceptList.length ||
    acceptList.some((ext) => candidate.name.toLowerCase().endsWith(ext));

  const syncFromInput = () => {
    const picked = inputRef.current?.files?.[0];
    if (!picked) {
      setFile(null);
      return;
    }
    if (!isAllowed(picked)) {
      setError(`Unsupported file. Use ${acceptList.join(" or ")}.`);
      if (inputRef.current) inputRef.current.value = "";
      setFile(null);
      return;
    }
    if (maxBytes && picked.size > maxBytes) {
      setError(`File too large. Maximum size is ${formatBytes(maxBytes)}.`);
      if (inputRef.current) inputRef.current.value = "";
      setFile(null);
      return;
    }
    setError("");
    setFile({ name: picked.name, size: picked.size });
  };

  const clear = () => {
    if (inputRef.current) inputRef.current.value = "";
    setFile(null);
    setError("");
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div
        className={cn(
          "relative flex min-h-28 items-center justify-center border border-dashed border-border-strong bg-surface-subtle px-4 py-5 transition-colors hover:border-brand hover:bg-brand-soft",
          dragOver && "border-solid border-brand bg-brand-soft",
          file && "border-solid border-border-strong bg-surface",
          error && "border-negative",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          name={name}
          accept={accept}
          required={required}
          className="absolute inset-0 z-10 size-full cursor-pointer opacity-0"
          onChange={syncFromInput}
          onDragEnter={() => setDragOver(true)}
          onDragOver={() => setDragOver(true)}
          onDragLeave={() => setDragOver(false)}
          onDrop={() => setDragOver(false)}
        />
        {file ? (
          <div className="pointer-events-none relative z-20 flex w-full items-center gap-3">
            <FileText className="shrink-0 text-brand-strong" size={22} aria-hidden="true" />
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="truncate text-sm font-semibold text-foreground">{file.name}</span>
              <span className="text-xs text-faint">
                {formatBytes(file.size)} · ready to submit
              </span>
            </span>
            <button
              type="button"
              className="pointer-events-auto grid size-8 shrink-0 cursor-pointer place-items-center border border-border bg-surface text-muted transition-colors hover:border-negative hover:text-negative"
              onClick={clear}
              aria-label="Remove selected file"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        ) : (
          <div className="pointer-events-none relative z-20 flex flex-col items-center gap-1.5 text-center">
            <span className="grid size-11 place-items-center border border-border bg-surface text-brand-strong">
              <UploadCloud size={24} aria-hidden="true" />
            </span>
            <span className="text-sm text-muted">
              <strong className="font-semibold text-brand-strong">Click to upload</strong> or drag and drop
            </span>
            {hint && <span className="text-xs text-faint">{hint}</span>}
          </div>
        )}
      </div>
      {error && (
        <p className="m-0 text-xs text-negative" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
